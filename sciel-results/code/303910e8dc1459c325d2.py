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

def _spectral_shaping(f, deadline, rng):
    """
    Iteratively shape the spectrum to minimize side-lobes in autoconvolution.
    We keep the DC component (sum) high and flatten the rest of the spectrum.
    """
    current_f = f.copy()
    best_f = f.copy()
    best_auto = _compute_autoconv(best_f)
    best_s = np.sum(best_f)
    best_score = _score_from_auto(best_auto, best_s)
    
    size = 1
    while size < 2 * N - 1:
        size <<= 1
        
    for _ in range(300):
        if time.time() > deadline - 15:
            break
            
        F = np.fft.rfft(current_f, size)
        
        # Strategy: Reduce peaks in |F|^2 (excluding DC)
        # By reducing high-magnitude non-DC bins, we reduce the variance of the autoconvolution
        mag = np.abs(F)
        phase = np.angle(F)
        
        # Target: A low constant magnitude for non-DC bins
        # Using a low percentile to aggressively flatten
        target_mag = np.percentile(mag[1:], 20)
        
        new_mag = mag.copy()
        new_mag[1:] = target_mag
        
        new_F = new_mag * np.exp(1j * phase)
        
        # Inverse FFT
        new_f = np.fft.irfft(new_F, size)
        
        # Round to +/- 1
        # Use sign, but handle 0s by keeping previous value
        new_f_rounded = np.sign(new_f[:N])
        zero_mask = (new_f_rounded == 0)
        new_f_rounded[zero_mask] = current_f[zero_mask]
        
        # Update current_f
        current_f = new_f_rounded
        
        # Evaluate
        auto = _compute_autoconv(current_f)
        s = np.sum(current_f)
        score = _score_from_auto(auto, s)
        
        if score < best_score:
            best_score = score
            best_f = current_f.copy()
            best_auto = auto.copy()
            best_s = s
            
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
    
    # Phase 1: Spectral Shaping
    # This provides a good initial structure by flattening the spectrum,
    # which tends to reduce the max absolute autoconvolution.
    rng = np.random.default_rng(42)
    best_f, best_auto, best_s, best_score = _spectral_shaping(best_f, deadline, rng)
    
    # Phase 2: Local Search
    # Refine the spectrally shaped sequence using greedy flips.
    if time.time() < deadline - 10:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng = np.random.default_rng(123)
        
        f, auto, s, current_score, best_score, best_f, best_auto, best_s = _local_search(
            f, auto, s, current_score, best_score, best_f, best_auto, best_s, deadline, rng)
            
    # Phase 3: Simulated Annealing
    # Escape local minima found by the greedy search.
    if time.time() < deadline - 5:
        f = best_f.copy()
        current_score = best_score
        rng = np.random.default_rng(456)
        
        # Determine number of iterations based on remaining time
        # Each iteration involves a flip and score calculation (incremental or full?)
        # Using incremental update for speed.
        # However, SA needs to accept worse moves, so we need to track state carefully.
        # Let's use full recomputation for SA to be safe against drift, or careful incremental.
        # Incremental is much faster. Let's try incremental.
        
        auto = best_auto.copy()
        s = best_s
        
        start_temp = 0.1
        end_temp = 0.001
        # Estimate iterations: 150s limit, we are at ~10-20s elapsed maybe?
        # Let's just run until deadline - 2
        it = 0
        n_iters = 100000 # Cap, but time check is primary
        
        while it < n_iters and time.time() < deadline - 2:
            it += 1
            progress = it / n_iters
            temp = start_temp * (end_temp / start_temp) ** progress
            
            i = rng.integers(N)
            old_val = f[i]
            delta = -2 * old_val
            
            new_auto = _update_autoconv(auto, f, i, delta)
            new_s = s + delta
            new_score = _score_from_auto(new_auto, new_s)
            
            if new_score < current_score:
                # Accept
                f[i] = old_val + delta
                auto = new_auto
                s = new_s
                current_score = new_score
                
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    best_auto = auto.copy()
                    best_s = s
            else:
                # Reject or accept with probability
                delta_score = new_score - current_score
                if delta_score > 0 and temp > 0:
                    if rng.random() < np.exp(-delta_score / temp):
                        f[i] = old_val + delta
                        auto = new_auto
                        s = new_s
                        current_score = new_score
                    else:
                        # Revert (do nothing, state already unchanged)
                        pass
                else:
                    # Revert
                    pass
                    
    # Final Polish
    if time.time() < deadline - 1:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng = np.random.default_rng(789)
        
        # One last greedy pass
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
