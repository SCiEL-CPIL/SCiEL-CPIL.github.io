import numpy as np
import time

N = 400

def _compute_autoconv(f):
    """Compute full autoconvolution via FFT"""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, size)[:2 * n - 1]
    return auto

def _update_autoconv(auto, f, i, delta):
    """
    Update autoconvolution when f[i] changes by delta.
    """
    n = len(f)
    left = np.zeros(2*n-1)
    left[i:i+n] = f
    right = np.zeros(2*n-1)
    right[:n-i] = f[i:]
    diag = np.zeros(2*n-1)
    if 0 <= 2*i < 2*n-1:
        diag[2*i] = 1.0
    return auto + delta * (left + right) + delta * delta * diag

def _score_from_auto(auto, s, exclude_lag0=False):
    """Compute score from autoconvolution and sum"""
    if abs(s) < 1e-12:
        return float('inf')
    if exclude_lag0:
        # Exclude lag 0 (index 0)
        max_abs = np.max(np.abs(auto[1:]))
    else:
        max_abs = np.max(np.abs(auto))
    return max_abs / (s * s)

def _spectral_optimize(f, deadline, rng):
    """
    Optimize in the frequency domain: directly shape |F|^2 to minimize
    the inverse FFT's maximum absolute value, then round to +/- 1.
    
    Key insight: autoconv = ifft(|F|^2), so we want |F|^2 to have a flat
    inverse FFT (except DC). We iteratively:
    1. Compute current F
    2. Identify which frequency bins contribute most to autoconv peaks
    3. Reduce those bins' magnitudes
    4. Reconstruct time domain and round
    """
    current_f = f.copy()
    best_f = f.copy()
    best_auto = _compute_autoconv(best_f)
    best_s = np.sum(best_f)
    best_score = _score_from_auto(best_auto, best_s)
    
    size = 1
    while size < 2 * N - 1:
        size <<= 1
    
    # Number of frequency bins for rfft
    n_bins = size // 2 + 1
    
    for iteration in range(500):
        if time.time() > deadline - 20:
            break
            
        F = np.fft.rfft(current_f, size)
        mag = np.abs(F)
        phase = np.angle(F)
        
        # Current autoconv
        auto = np.fft.irfft(mag**2, size)[:2*N-1]
        
        # Find the peak lag (excluding DC)
        abs_auto = np.abs(auto)
        # We want to reduce the max of abs_auto[1:]
        peak_idx = np.argmax(abs_auto[1:]) + 1
        
        # The autoconv at lag k is ifft(|F|^2)[k] = sum_j |F[j]|^2 * exp(2*pi*i*j*k/size)
        # To reduce auto[peak_idx], we want to reduce |F[j]|^2 for j where 
        # exp(2*pi*i*j*peak_idx/size) has positive real part (constructive)
        # and increase where negative (destructive) - but we can only reduce magnitudes
        
        # Alternative approach: use gradient-based shaping
        # d(auto[k])/d(|F[j]|^2) = cos(2*pi*j*k/size)
        # To minimize auto[peak_idx], reduce |F[j]| where cos(2*pi*j*peak_idx/size) > 0
        
        cos_term = np.cos(2 * np.pi * np.arange(n_bins) * peak_idx / size)
        
        # Only modify non-DC bins (index 0)
        # Reduce bins where cos_term > 0 (they add positively to the peak)
        # Scale factor: how much to reduce
        scale = 0.95
        
        new_mag = mag.copy()
        positive_mask = (cos_term > 0.1) & (np.arange(n_bins) > 0)
        new_mag[positive_mask] *= scale
        
        # Also try to flatten: bring high-magnitude bins down
        target_mag = np.percentile(mag[1:], 30)
        high_mask = (mag > target_mag * 1.5) & (np.arange(n_bins) > 0)
        new_mag[high_mask] = np.minimum(new_mag[high_mask], target_mag * 1.2)
        
        new_F = new_mag * np.exp(1j * phase)
        new_f = np.fft.irfft(new_F, size)
        
        # Round to +/- 1
        new_f_rounded = np.sign(new_f[:N])
        zero_mask = (new_f_rounded == 0)
        new_f_rounded[zero_mask] = current_f[zero_mask]
        
        # Evaluate
        new_auto = _compute_autoconv(new_f_rounded)
        new_s = np.sum(new_f_rounded)
        new_score = _score_from_auto(new_auto, new_s)
        
        if new_score < best_score:
            best_score = new_score
            best_f = new_f_rounded.copy()
            best_auto = new_auto.copy()
            best_s = new_s
            current_f = new_f_rounded.copy()
        else:
            # Keep current_f for next iteration but don't update best
            # Sometimes accept worse moves to escape local minima
            if rng.random() < 0.1:
                current_f = new_f_rounded.copy()
    
    return best_f, best_auto, best_s, best_score

def _local_search(f, auto, s, current_score, best_score, best_f, best_auto, best_s, deadline, rng):
    """Greedy local search with incremental autoconv update."""
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline - 10:
                break
            old_val = f[i]
            delta = -2 * old_val
            new_auto = _update_autoconv(auto, f, i, delta)
            new_s = s + delta
            new_score = _score_from_auto(new_auto, new_s)
            if new_score < current_score - 1e-12:
                f[i] = old_val + delta
                auto = new_auto
                s = new_s
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    best_auto = auto.copy()
                    best_s = s
                improved = True
    return f, auto, s, current_score, best_score, best_f, best_auto, best_s

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    # Start with all +1s
    best_f = np.ones(N)
    best_auto = _compute_autoconv(best_f)
    best_s = np.sum(best_f)
    best_score = _score_from_auto(best_auto, best_s)
    
    # Phase 1: Spectral Optimization (NEW: direct frequency-domain shaping)
    rng = np.random.default_rng(42)
    best_f, best_auto, best_s, best_score = _spectral_optimize(best_f, deadline, rng)
    
    # Phase 2: Local Search to refine
    if time.time() < deadline - 10:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng = np.random.default_rng(123)
        
        f, auto, s, current_score, best_score, best_f, best_auto, best_s = _local_search(
            f, auto, s, current_score, best_score, best_f, best_auto, best_s, deadline, rng)
            
    # Phase 3: More spectral optimization from improved starting point
    if time.time() < deadline - 15:
        rng2 = np.random.default_rng(789)
        new_f, new_auto, new_s, new_score = _spectral_optimize(best_f, deadline, rng2)
        if new_score < best_score:
            best_score = new_score
            best_f = new_f.copy()
            best_auto = new_auto.copy()
            best_s = new_s
    
    # Final Polish: one last greedy pass
    if time.time() < deadline - 1:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng = np.random.default_rng(789)
        
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline - 1:
                break
            old_val = f[i]
            delta = -2 * old_val
            new_auto = _update_autoconv(auto, f, i, delta)
            new_s = s + delta
            new_score = _score_from_auto(new_auto, new_s)
            if new_score < current_score - 1e-12:
                f[i] = old_val + delta
                auto = new_auto
                s = new_s
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    best_auto = auto.copy()
                    best_s = s
                    
    return best_f
