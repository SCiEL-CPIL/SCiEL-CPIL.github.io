"""Direct construction: Golay complementary pairs + Rudin-Shapiro, with fast delta local search.
Objective: minimize max|autoconv(h)| / (sum h)^2 over signed step heights h of length N.
For ±1 signals, lag-0 autoconv = N, so we want all other lags near zero (Golay-like).
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


def _golay_init():
    """
    Generate a Golay complementary pair of length >= N and truncate to N.
    Start with a=[1], b=[1] and iterate: a'=[a,b], b'=[a,-b].
    Return the truncated a sequence (or b if better).
    """
    a = np.array([1.0])
    b = np.array([1.0])
    while len(a) < N:
        a = np.concatenate([a, b])
        b = np.concatenate([a, -b])
    a = a[:N]
    b = b[:N]
    # Evaluate both and return the better one
    val_a, _ = _eval(a)
    val_b, _ = _eval(b)
    return a if val_a <= val_b else b


def _rudin_shapiro_init():
    """
    Generate a Rudin-Shapiro sequence of length >= N.
    RS sequence: r(0)=1, r(1)=1, r(2n)=r(n), r(2n+1)=-r(n+1)
    Truncate to N.
    """
    length = 1
    while length < N:
        length *= 2
    rs = np.zeros(length)
    rs[0] = 1.0
    rs[1] = 1.0
    for i in range(2, length):
        if i % 2 == 0:
            rs[i] = rs[i // 2]
        else:
            rs[i] = -rs[(i + 1) // 2]
    return rs[:N]


def _delta_update(c, h, idx, L):
    """
    Compute the change in c[L] when h[idx] is flipped.
    c[L] = sum_k h[k]*h[k+L]
    Flipping h[idx] changes c[L] by -2*h[idx]*(h[idx+L] if valid + h[idx-L] if valid)
    """
    n = len(h)
    delta = 0.0
    # h[idx] appears in c[L] when:
    # 1. k = idx, so we need idx + L < n: term h[idx]*h[idx+L]
    # 2. k+L = idx, so k = idx-L >= 0: term h[idx-L]*h[idx]
    if L >= 0:
        if idx + L < n:
            delta += h[idx + L]
        if idx - L >= 0:
            delta += h[idx - L]
    else:
        L_abs = -L
        if idx + L_abs < n:
            delta += h[idx + L_abs]
        if idx - L_abs >= 0:
            delta += h[idx - L_abs]
    return -2.0 * h[idx] * delta


def _local_search(h, c, max_sweeps=200, deadline=130.0):
    """
    Fast local search using delta updates instead of full FFT recomputation.
    For each flip, compute the exact change in the peak autoconvolution value.
    """
    start = time.time()
    current_val, _ = _eval(h)
    s = float(np.sum(h))
    denom = s * s if s * s > 1e-12 else 1e-12
    current_max = float(np.max(np.abs(c)))
    
    np.random.seed(42)
    
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        
        improved = False
        # Random order for this sweep
        order = np.random.permutation(N)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            # Find current peak lag
            abs_c = np.abs(c)
            max_pos = int(np.argmax(abs_c))
            L_peak = max_pos - (N - 1)
            
            # Compute delta for flipping h[idx] at the peak lag
            delta = _delta_update(c, h, idx, L_peak)
            
            # New peak value at L_peak
            new_peak = c[max_pos] + delta
            
            # Check if this reduces the max absolute value
            # We need to check if |new_peak| < current_max, and also if new_peak becomes the new max
            # For speed, only accept if it clearly reduces the peak
            if abs(new_peak) < abs(c[max_pos]) - 1e-10:
                # Apply flip
                old_val = h[idx]
                h[idx] = -old_val
                
                # Update c at affected lags
                # Flipping h[idx] affects c[L] for all L where idx is involved
                # c[L] changes by -2*h[idx]*(h[idx+L] + h[idx-L]) for each L
                # We can update c incrementally
                for L in range(-(N-1), N):
                    d = _delta_update(c, h, idx, L)
                    # Note: h[idx] is already flipped, so we need to be careful
                    # Actually, _delta_update uses current h, which is already flipped
                    # So the delta computed is for the NEW state
                    # We need the delta for the OLD state
                    # Let's recompute: before flip, delta_old = -2*old_val*(h[idx+L]+h[idx-L])
                    # But h[idx+L] and h[idx-L] haven't changed (only h[idx] changed)
                    # So delta_old = -2*old_val*(h[idx+L]+h[idx-L])
                    # And delta_new (if we were to flip back) = -2*(-old_val)*(h[idx+L]+h[idx-L]) = -delta_old
                    # So the change to c[L] is delta_old
                    # Let's just recompute c fully for correctness
                    pass
                
                # For correctness, recompute c fully (still fast with FFT)
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
                    improved = True
                else:
                    # Revert
                    h[idx] = old_val
                    # c is unchanged since we reverted
                    pass
        
        if not improved:
            break
    
    return h, current_val


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Generate initial candidates
    candidates = []
    
    # Golay init
    h_golay = _golay_init()
    val_g, c_g = _eval(h_golay)
    candidates.append((val_g, h_golay, c_g))
    
    # Rudin-Shapiro init
    h_rs = _rudin_shapiro_init()
    val_rs, c_rs = _eval(h_rs)
    candidates.append((val_rs, h_rs, c_rs))
    
    # Multiple spectral inits
    for seed in [42, 7, 123, 456, 789, 1000, 2000, 3000]:
        if time.time() > deadline - 60:
            break
        np.random.seed(seed)
        size = 1
        while size < 2 * N - 1:
            size *= 2
        H = np.random.randn(size // 2 + 1) + 1j * np.random.randn(size // 2 + 1)
        H[0] = float(np.real(H[0]))
        if size > 1:
            H[size // 2] = float(np.real(H[size // 2]))
        h_cont = np.fft.irfft(H, n=size)[:N]
        h_spec = np.sign(h_cont)
        h_spec[h_spec == 0] = 1.0
        val_s, c_s = _eval(h_spec)
        candidates.append((val_s, h_spec, c_s))
    
    # Sort by initial value
    candidates.sort(key=lambda x: x[0])
    
    # Local search on top candidates
    for val_init, h_init, c_init in candidates[:5]:
        if time.time() > deadline - 10:
            break
        h_copy = h_init.copy()
        c_copy = c_init.copy()
        h_opt, val_opt = _local_search(h_copy, c_copy, deadline=deadline - 5)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Final polish with more sweeps on the best
    if time.time() < deadline - 2 and best_h is not None:
        _, c_best = _eval(best_h)
        h_final, val_final = _local_search(best_h.copy(), c_best.copy(), deadline=deadline, max_sweeps=100)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
