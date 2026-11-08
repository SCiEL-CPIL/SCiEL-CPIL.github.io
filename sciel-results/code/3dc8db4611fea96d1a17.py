"""Spectral init + subgradient coordinate descent for signed autocorrelation minimization.
Objective: minimize max|autoconv(h)| / (sum h)^2 over signed step heights h of length N.
Structural change: use FFT-based spectral initialization and subgradient descent on peaks
instead of O(N^2) greedy flip local search.
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


def _spectral_init(seed=42):
    """
    Spectral initialization: start from random phase in frequency domain,
    then threshold to ±1. This gives a signal with controlled autoconvolution
    shape from the start.
    """
    np.random.seed(seed)
    size = 1
    while size < 2 * N - 1:
        size *= 2
    # Random frequency-domain signal with conjugate symmetry for real output
    H = np.random.randn(size // 2 + 1) + 1j * np.random.randn(size // 2 + 1)
    # Ensure H[0] is real (DC component)
    H[0] = float(np.real(H[0]))
    if size > 1:
        H[size // 2] = float(np.real(H[size // 2]))  # Nyquist
    h_cont = np.fft.irfft(H, n=size)[:N]
    # Threshold to ±1
    h = np.sign(h_cont)
    # Handle zeros
    h[h == 0] = 1.0
    return h


def _subgradient_descent(h, c, max_iter=50000, deadline=130.0, step_size=0.5):
    """
    Subgradient coordinate descent: at each step, identify the lag L* with maximum |c[L*]|.
    The subgradient of max|c| w.r.t. h[j] involves c's sensitivity to h[j].
    For autoconv c[L] = sum_k h[k]*h[k+L], d c[L]/d h[j] = h[j+L] + h[j-L] (with boundary handling).
    
    We use a projected subgradient method: update h in the direction that reduces
    the peak autoconvolution, then project back to ±1.
    
    Actually, for ±1 constraints, we use coordinate descent: pick the index j that
    most reduces the objective if flipped, based on the subgradient at the peak lag.
    """
    start = time.time()
    current_val, _ = _eval(h)
    s = float(np.sum(h))
    denom = s * s if s * s > 1e-12 else 1e-12
    current_max = float(np.max(np.abs(c)))
    max_pos = int(np.argmax(np.abs(c)))  # position in c array, L = max_pos - (N-1)
    
    np.random.seed(123)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find current peak lag
        abs_c = np.abs(c)
        max_pos = int(np.argmax(abs_c))
        L_peak = max_pos - (N - 1)  # actual lag
        peak_val = c[max_pos]
        sign_peak = np.sign(peak_val) if peak_val != 0 else 1.0
        
        # Subgradient: for each j, compute how flipping h[j] affects c at the peak lag
        # d c[L_peak] / d h[j] = h[j + L_peak] (if valid) + h[j - L_peak] (if valid)
        # Flipping h[j] changes c[L_peak] by -2 * h[j] * (h[j+L_peak] + h[j-L_peak])
        # We want to reduce |c[L_peak]|, so we want to flip j where:
        #   sign_peak * h[j] * (h[j+L_peak] + h[j-L_peak]) > 0
        # (this makes the change in c[L_peak] oppose the current sign)
        
        # Compute the subgradient vector for the peak lag
        L = L_peak
        sg = np.zeros(N)
        if L >= 0:
            # c[L] = sum_{j=0}^{N-1-L} h[j]*h[j+L]
            # d c[L] / d h[j] = h[j+L] (if j+L < N) + h[j-L] (if j-L >= 0)
            # For L > 0: d c[L] / d h[j] = h[j+L] (if j < N-L) + h[j-L] (if j >= L)
            for j in range(N):
                val = 0.0
                if j + L < N:
                    val += h[j + L]
                if j - L >= 0:
                    val += h[j - L]
                sg[j] = val
        elif L < 0:
            # By symmetry, same as |L|
            L_abs = -L
            for j in range(N):
                val = 0.0
                if j + L_abs < N:
                    val += h[j + L_abs]
                if j - L_abs >= 0:
                    val += h[j - L_abs]
                sg[j] = val
        
        # Score for flipping each j: we want sign_peak * h[j] * sg[j] to be positive
        # Flipping j changes c[L_peak] by -2*h[j]*sg[j]
        # New c[L_peak] = c[L_peak] - 2*h[j]*sg[j]
        # We want |new_c| < |c[L_peak]|, i.e., sign_peak * (-2*h[j]*sg[j]) < 0
        # i.e., sign_peak * h[j] * sg[j] > 0
        
        flip_scores = sign_peak * h * sg
        # Only consider flips that reduce the peak
        valid_flips = flip_scores > 0
        if not np.any(valid_flips):
            # No flip reduces the peak; try a random flip or break
            if it % 1000 == 0:
                # Random perturbation
                idx = np.random.randint(N)
                h[idx] = -h[idx]
                # Recompute c
                c = _fft_autoconv(h)
                s = float(np.sum(h))
                denom = s * s if s * s > 1e-12 else 1e-12
                current_max = float(np.max(np.abs(c)))
                current_val = current_max / denom
            continue
        
        # Pick the flip that most reduces the peak
        best_idx = int(np.argmax(flip_scores[valid_flips]))
        # Map back to actual index
        valid_indices = np.where(valid_flips)[0]
        best_idx = valid_indices[best_idx]
        
        # Evaluate the flip
        old_h = h[best_idx]
        h[best_idx] = -old_h
        new_c = _fft_autoconv(h)
        new_s = float(np.sum(h))
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
            # Revert
            h[best_idx] = old_h
    
    return h, current_val


def _coordinate_descent(h, c, max_iter=30000, deadline=130.0):
    """
    Coordinate descent: for each index j, compute the exact change in objective
    if h[j] is flipped, using FFT for the new autoconvolution. Accept if improved.
    This is O(N) per sweep with O(N log N) per FFT, so O(N^2 log N) total per sweep.
    """
    start = time.time()
    current_val, _ = _eval(h)
    s = float(np.sum(h))
    denom = s * s if s * s > 1e-12 else 1e-12
    
    np.random.seed(99)
    order = np.random.permutation(N)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = int(order[it % N])
        old_val = h[idx]
        h[idx] = -old_val
        
        new_c = _fft_autoconv(h)
        new_s = float(np.sum(h))
        new_denom = new_s * new_s if new_s * new_s > 1e-12 else 1e-12
        new_max = float(np.max(np.abs(new_c)))
        new_val = new_max / new_denom
        
        if new_val < current_val - 1e-15:
            c = new_c
            s = new_s
            denom = new_denom
            current_val = new_val
        else:
            h[idx] = old_val
    
    return h, current_val


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Try multiple spectral initializations
    for seed in [42, 7, 123, 456, 789]:
        if time.time() > deadline - 20:
            break
        h_spec = _spectral_init(seed=seed)
        val, c = _eval(h_spec)
        if val < best_val:
            best_val = val
            best_h = h_spec.copy()
            best_c = c.copy()
    
    # Also try simple Golay-like initialization
    a = np.array([1.0])
    b = np.array([1.0])
    while len(a) < N:
        a = np.concatenate([a, b])
        b = np.concatenate([a, -b])
    a = a[:N]
    b = b[:N]
    for h_g in [a, b]:
        val_g, c_g = _eval(h_g)
        if val_g < best_val:
            best_val = val_g
            best_h = h_g.copy()
            best_c = c_g.copy()
    
    # Subgradient descent on best
    if time.time() < deadline - 10 and best_h is not None:
        _, c_best = _eval(best_h)
        h_sg, val_sg = _subgradient_descent(best_h.copy(), c_best.copy(), 
                                             max_iter=20000, deadline=deadline - 5)
        if val_sg < best_val:
            best_val = val_sg
            best_h = h_sg
    
    # Coordinate descent on the result
    if time.time() < deadline - 5 and best_h is not None:
        _, c_cd = _eval(best_h)
        h_cd, val_cd = _coordinate_descent(best_h.copy(), c_cd.copy(),
                                            max_iter=10000, deadline=deadline - 2)
        if val_cd < best_val:
            best_val = val_cd
            best_h = h_cd
    
    # Final coordinate descent pass
    if time.time() < deadline - 2 and best_h is not None:
        _, c_final = _eval(best_h)
        h_final, val_final = _coordinate_descent(best_h.copy(), c_final.copy(),
                                                  max_iter=5000, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
