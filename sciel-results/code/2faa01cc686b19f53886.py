"""
Spectral shaping + fast delta local search for signed autocorrelation minimization.
Objective: minimize max|autoconv(h)| / (sum h)^2 over signed step heights h of length N.
Strategy: construct h as the sign of the inverse DFT of a random unit-magnitude phase mask
(this is the structural spectral-shaping idea from the inspiration), then refine with an
O(N) delta-update local search (the refinement machinery from the current program) that
avoids full FFT re-computation on every flip.
"""
import time
import numpy as np

N = 400
_DEADLINE = 130.0  # safety margin under 150s kill


def _fft_autoconv(h):
    """Linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    C = np.fft.irfft(H * H, n=fft_size)
    return C[:size]


def _eval(h):
    """Return (value, c) where value = max|c| / (sum h)^2."""
    c = _fft_autoconv(h)
    m = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    denom = s * s
    if denom <= 1e-12:
        denom = 1e-12
    return m / denom, c


def _spectral_phase_init(seed=42, band_low=0.1, band_high=0.9):
    """
    Spectral shaping: construct h as the sign of the inverse DFT of a random
    unit-magnitude phase mask restricted to a band.
    """
    np.random.seed(seed)
    size = 1
    while size < 2 * N - 1:
        size *= 2
    phases = np.random.uniform(0, 2 * np.pi, size // 2 + 1)
    H = np.exp(1j * phases)
    freqs = np.arange(size // 2 + 1) / size
    band_mask = (freqs >= band_low) & (freqs <= band_high)
    H[~band_mask] = 0.0
    H[0] = float(np.real(H[0]))
    if size > 1:
        H[size // 2] = float(np.real(H[size // 2]))
    h_cont = np.fft.irfft(H, n=size)[:N]
    h = np.sign(h_cont)
    h[h == 0] = 1.0
    return h


def _golay_pair():
    """Golay complementary pair construction."""
    a = np.array([1.0])
    b = np.array([1.0])
    while len(a) < N:
        a = np.concatenate([a, b])
        b = np.concatenate([a, -b])
    a = a[:N]
    b = b[:N]
    return a, b


def _local_search(h, c, max_iter=50000, deadline=130.0):
    """
    Fast local search using delta updates to avoid full FFT recomputation.
    """
    start = time.time()
    current_val, _ = _eval(h)
    s = float(np.sum(h))
    denom = s * s if s * s > 1e-12 else 1e-12
    current_max = float(np.max(np.abs(c)))
    
    np.random.seed(42)
    order = np.random.permutation(N)
    oi = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    # Precompute Ls array
    Ls = np.arange(-(N - 1), N)
    zero_pos = N - 1
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve_count > max_no_improve:
            # Reset with new random order and break if stuck
            if it > 10000:
                break
            no_improve_count = 0
            order = np.random.permutation(N)
            oi = 0
            # Try a random perturbation
            idx = np.random.randint(N)
            h[idx] = -h[idx]
            c = _fft_autoconv(h)
            s = float(np.sum(h))
            denom = s * s if s * s > 1e-12 else 1e-12
            current_max = float(np.max(np.abs(c)))
            current_val = current_max / denom
            continue
        
        if oi >= N:
            oi = 0
            order = np.random.permutation(N)
        
        idx = int(order[oi])
        oi += 1
        target = -h[idx]
        d = target - h[idx]  # ±2
        
        # Compute delta_c efficiently
        valid1 = (idx + Ls >= 0) & (idx + Ls < N)
        valid2 = (idx - Ls >= 0) & (idx - Ls < N)
        # Use np.where to avoid index errors
        term1 = np.where(valid1, h[np.where(valid1, idx + Ls, 0)], 0.0)
        term2 = np.where(valid2, h[np.where(valid2, idx - Ls, 0)], 0.0)
        delta = d * (term1 + term2)
        delta[zero_pos] = 0.0
        
        new_c = c + delta
        new_s = s + d
        new_denom = new_s * new_s if new_s * new_s > 1e-12 else 1e-12
        new_max = float(np.max(np.abs(new_c)))
        new_val = new_max / new_denom
        
        if new_val < current_val - 1e-15:
            h[idx] = target
            c = new_c
            s = new_s
            denom = new_denom
            current_max = new_max
            current_val = new_val
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return h, current_val


def _greedy_peak_reduction(h, c, max_iter=20000, deadline=130.0):
    """
    Greedy peak reduction: at each step, find the lag with maximum |c|,
    then find the flip that most reduces that peak using the delta formula.
    """
    start = time.time()
    current_val, _ = _eval(h)
    s = float(np.sum(h))
    denom = s * s if s * s > 1e-12 else 1e-12
    current_max = float(np.max(np.abs(c)))
    
    Ls = np.arange(-(N - 1), N)
    zero_pos = N - 1
    
    np.random.seed(7)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find peak lag
        abs_c = np.abs(c)
        max_pos = int(np.argmax(abs_c))
        peak_val = c[max_pos]
        sign_peak = 1.0 if peak_val > 0 else -1.0
        
        # For each idx, compute how flipping affects the peak
        # delta_c at peak_pos for flip at idx:
        # L_peak = max_pos - (N-1)
        L_peak = max_pos - (N - 1)
        
        # delta_c[max_pos] = d * (h[idx + L_peak] if valid + h[idx - L_peak] if valid)
        # where d = ±2
        # We want to reduce |c[max_pos]|, so we want:
        # sign_peak * d * (h[idx+L_peak] + h[idx-L_peak]) < 0
        # Since d = -2*h[idx], this becomes:
        # sign_peak * (-2*h[idx]) * (h[idx+L_peak] + h[idx-L_peak]) < 0
        # i.e., sign_peak * h[idx] * (h[idx+L_peak] + h[idx-L_peak]) > 0
        
        if L_peak >= 0:
            # h[idx + L_peak] valid if idx + L_peak < N
            # h[idx - L_peak] valid if idx - L_peak >= 0
            term1 = np.zeros(N)
            term2 = np.zeros(N)
            v1 = np.arange(N) + L_peak < N
            v2 = np.arange(N) - L_peak >= 0
            term1[v1] = h[np.arange(N)[v1] + L_peak]
            term2[v2] = h[np.arange(N)[v2] - L_peak]
        else:
            L_abs = -L_peak
            term1 = np.zeros(N)
            term2 = np.zeros(N)
            v1 = np.arange(N) + L_abs < N
            v2 = np.arange(N) - L_abs >= 0
            term1[v1] = h[np.arange(N)[v1] + L_abs]
            term2[v2] = h[np.arange(N)[v2] - L_abs]
        
        neighbor_sum = term1 + term2
        flip_scores = sign_peak * h * neighbor_sum
        valid_flips = flip_scores > 0
        
        if not np.any(valid_flips):
            # No flip reduces the peak; try random perturbation
            if it % 2000 == 0:
                idx = np.random.randint(N)
                h[idx] = -h[idx]
                c = _fft_autoconv(h)
                s = float(np.sum(h))
                denom = s * s if s * s > 1e-12 else 1e-12
                current_max = float(np.max(np.abs(c)))
                current_val = current_max / denom
            continue
        
        valid_indices = np.where(valid_flips)[0]
        best_idx = valid_indices[int(np.argmax(flip_scores[valid_indices]))]
        
        # Evaluate the flip
        old_h = h[best_idx]
        h[best_idx] = -old_h
        d = -2 * old_h
        
        # Update c using delta
        idx = best_idx
        valid1 = (idx + Ls >= 0) & (idx + Ls < N)
        valid2 = (idx - Ls >= 0) & (idx - Ls < N)
        term1 = np.where(valid1, h[np.where(valid1, idx + Ls, 0)], 0.0)
        term2 = np.where(valid2, h[np.where(valid2, idx - Ls, 0)], 0.0)
        delta = d * (term1 + term2)
        delta[zero_pos] = 0.0
        
        new_c = c + delta
        new_s = s + d
        new_denom = new_s * new_s if new_s * new_s > 1e-12 else 1e-12
        new_max = float(np.max(np.abs(new_c)))
        new_val = new_max / new_denom
        
        if new_val < current_val - 1e-15:
            c = new_c
            s = new_s
            denom = new_denom
            current_max = new_max
            current_val = new_val
        else:
            h[best_idx] = old_h
    
    return h, current_val


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    best_h = None
    best_val = float('inf')
    best_c = None
    
    # Phase 1: Generate diverse candidates
    candidates = []
    
    # Spectral phase inits with different bands and seeds
    for seed in [42, 7, 123, 456, 789, 1000, 2000]:
        if time.time() > deadline - 30:
            break
        for bl, bh in [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.05, 0.95)]:
            h_spec = _spectral_phase_init(seed=seed, band_low=bl, band_high=bh)
            candidates.append(h_spec)
    
    # Golay pairs
    a, b = _golay_pair()
    candidates.append(a)
    candidates.append(b)
    
    # Evaluate all candidates
    for h_cand in candidates:
        if time.time() > deadline - 25:
            break
        val, c = _eval(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()
            best_c = c.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val, best_c = _eval(best_h)
    
    # Phase 2: Greedy peak reduction on best candidate
    if time.time() < deadline - 15:
        h_greedy, val_greedy = _greedy_peak_reduction(
            best_h.copy(), best_c.copy(), 
            max_iter=20000, deadline=deadline - 10
        )
        if val_greedy < best_val:
            best_val = val_greedy
            best_h = h_greedy
            _, best_c = _eval(best_h)
    
    # Phase 3: Fast delta local search
    if time.time() < deadline - 5:
        h_ls, val_ls = _local_search(
            best_h.copy(), best_c.copy(),
            max_iter=80000, deadline=deadline - 2
        )
        if val_ls < best_val:
            best_val = val_ls
            best_h = h_ls
    
    # Phase 4: Final short local search pass
    if time.time() < deadline - 1:
        _, c_final = _eval(best_h)
        h_final, val_final = _local_search(
            best_h.copy(), c_final.copy(),
            max_iter=20000, deadline=deadline
        )
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    return best_h
