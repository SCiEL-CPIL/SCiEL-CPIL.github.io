"""Combined: spectral subspace projection + FFT local search for signed step heights."""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# EVOLVE-BLOCK-START

def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2 * N - 1]

def _raw_value(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    return float(np.max(np.abs(ac))) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max|ac| via log-sum-exp. Returns (value, weights)."""
    a = np.abs(ac)
    m = np.max(a)
    if m < 1e-15:
        return 0.0, None
    scaled = beta * (a - m)
    scaled = np.clip(scaled, -60, 60)
    e = np.exp(scaled)
    log_sum = np.log(np.sum(e))
    weights = e / np.sum(e)
    return m + log_sum / beta, weights

def _smooth_objective(h, beta=50.0):
    """Smooth objective: soft_max_abs(ac) / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    sm, _ = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _smooth_gradient(h, beta=50.0):
    """
    Gradient of soft_max_abs(ac) / (sum h)^2 with respect to h.
    Uses FFT-based cross-correlation for the autoconvolution gradient.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)

    ac = _autoconv(h)
    sm, weights = _soft_max_abs(ac, beta)

    if weights is None:
        return np.zeros_like(h)

    # g_ac[m] = weights[m] * sign(ac[m])
    g_ac = weights * np.sign(ac)

    # Gradient of S = soft_max_abs(ac) w.r.t. h:
    # dS/dh[j] = sum_m g_ac[m] * (h[m-j] + h[j-m])
    # By symmetry of autoconv and g_ac, this equals 2 * cross_corr(g_ac, h)[j]
    # where cross_corr(a, b)[j] = sum_m a[m+j] * b[m]
    
    # Compute cross-correlation via FFT
    # We need C[j] = sum_m g_ac[m+j] * h[m] for j=0..N-1
    # g_ac has length 2N-1, h has length N
    # Max index in g_ac accessed: (N-1) + (N-1) = 2N-2, which is within 2N-1
    # Use FFT of size >= (2N-1) + N - 1 = 3N-2
    fsize_corr = 1
    while fsize_corr < 3 * N:
        fsize_corr *= 2

    V = np.fft.rfft(g_ac, fsize_corr)
    H = np.fft.rfft(h, fsize_corr)
    C = np.fft.irfft(V * np.conj(H), fsize_corr)
    
    # C[j] = sum_n g_ac[n+j] * conj(h[n]) = sum_n g_ac[n+j] * h[n] (h is real)
    # This is exactly what we want for j=0..N-1
    grad_S = 2.0 * C[:N]

    # f = S / s^2
    # df/dh[j] = (grad_S[j] * s - 2 * S) / s^3
    df = (grad_S * s - 2.0 * sm) / (s ** 3)

    return df

def _lbfgs_optimize(h, beta=50.0, max_iter=300, deadline=None):
    """L-BFGS optimization of the smooth surrogate."""
    h0 = h.copy()
    
    def fun(x):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _smooth_objective(x, beta)
    
    def jac(x):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(x)
        return _smooth_gradient(x, beta)
    
    try:
        res = minimize(fun, h0, method='L-BFGS-B', jac=jac,
                       options={'maxiter': max_iter, 'gtol': 1e-7, 'maxfun': 1000})
        return res.x, res.fun
    except Exception:
        return h0, _smooth_objective(h0, beta)

def _block_sign_flip_local_search(h, block_sizes=None, max_passes=5, deadline=None):
    """
    Local search: try flipping signs of blocks of consecutive elements.
    O(N log N) per evaluation via FFT.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    h = h.copy()
    best_val = _raw_value(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for bs in block_sizes:
            if deadline is not None and time.time() > deadline:
                break
            
            for start in range(0, N, bs):
                if deadline is not None and time.time() > deadline:
                    break
                
                end = min(start + bs, N)
                h[start:end] *= -1
                new_val = _raw_value(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:end] *= -1  # revert
        
        if not improved:
            break
    
    return h, best_val

def _single_flip_local_search(h, max_passes=10, deadline=None):
    """
    Local search: try flipping each individual element's sign.
    Uses incremental autoconvolution update for speed.
    """
    h = h.copy()
    s = np.sum(h)
    ac = _autoconv(h)
    best_val = float(np.max(np.abs(ac))) / (s * s)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            # Flipping h[i] -> -h[i] changes s -> s - 2*h[i]
            # Autoconvolution: ac_new[k] = ac_old[k] - 2*h[i]*(h[k-i] + h[i-k]) for valid indices
            # More precisely: ac_new = ac_old - 2*h[i]*(delta_i * h + h * delta_i)
            # where delta_i is unit vector at position i
            # ac_new[k] = ac_old[k] - 2*h[i]*h[k-i] - 2*h[i]*h[i-k] (with h[j]=0 for j<0 or j>=N)
            
            hi = h[i]
            s_new = s - 2.0 * hi
            
            # Compute delta ac
            # ac[k] = sum_j h[j]*h[k-j]
            # d(ac)[k] = -2*hi*(h[k-i] + h[i-k]) if indices valid
            # For k from 0 to 2N-2:
            #   term1: h[k-i] valid when 0 <= k-i < N, i.e., i <= k <= i+N-1
            #   term2: h[i-k] valid when 0 <= i-k < N, i.e., i-N+1 <= k <= i
            
            delta_ac = np.zeros(2 * N - 1)
            # term1: k in [i, i+N-1], h[k-i]
            k1_start = i
            k1_end = min(i + N, 2 * N - 1)
            if k1_start < k1_end:
                delta_ac[k1_start:k1_end] -= 2.0 * hi * h[0:k1_end - k1_start]
            # term2: k in [max(0,i-N+1), i], h[i-k]
            k2_start = max(0, i - N + 1)
            k2_end = i + 1
            if k2_start < k2_end:
                # h[i-k] for k=k2_start..k2_end-1 => indices i-k2_start .. i-k2_end+1 = i .. i-(k2_end-1)
                # i-k for k in [k2_start, k2_end-1] => i-k2_start down to i-(k2_end-1)
                idx = i - np.arange(k2_start, k2_end)
                delta_ac[k2_start:k2_end] -= 2.0 * hi * h[idx]
            
            ac_new = ac + delta_ac
            new_val = float(np.max(np.abs(ac_new))) / (s_new * s_new)
            
            if new_val < best_val:
                h[i] = -hi
                s = s_new
                ac = ac_new
                best_val = new_val
                improved = True
            # else: no change
        
        if not improved:
            break
    
    return h, best_val

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_sinc(n, beta=5.0):
    """Generate a sinc-like sequence."""
    i = np.arange(n)
    x = (i - n/2) / (n * beta)
    h = np.sinc(x)
    return h

def _gen_gaussian(n, sigma_frac=0.25):
    """Generate a Gaussian window."""
    i = np.arange(n)
    mu = n / 2
    sigma = n * sigma_frac
    return np.exp(-(i - mu)**2 / (2 * sigma**2))

def _gen_barker_like(n):
    """Generate a sequence with alternating pattern."""
    h = np.zeros(n)
    for i in range(n):
        if i % 4 in [0, 2]:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _raw_value(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian window
    inits.append(_gen_gaussian(N, sigma_frac=0.25))
    inits.append(_gen_gaussian(N, sigma_frac=0.15))
    
    # 4. Triangle window
    mu = N/2
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    inits.append(_gen_chirp(N))
    
    # 6. Sinc sequence
    inits.append(_gen_sinc(N, beta=3.0))
    inits.append(_gen_sinc(N, beta=5.0))
    
    # 7. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 8. Random smoothed sequences
    for seed in [42, 123, 456, 789, 101]:
        inits.append(_gen_random_smoothed(N, seed=seed))
        
    # Optimization loop using smooth surrogate with analytical gradient
    for h_init in inits:
        if time.time() > deadline - 25:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape
        h_norm = h_init / s0
        
        # Use smooth objective for gradient-based optimization
        try:
            # Start with moderate beta, increase for refinement
            for beta in [15, 30, 60, 100]:
                if time.time() > deadline - 15:
                    break
                
                h_opt, _ = _lbfgs_optimize(h_norm, beta=beta, max_iter=150, deadline=deadline - 10)
                
                # Check validity
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val = _raw_value(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
        except Exception:
            pass

    # Final local search refinement: single flip with incremental update (fastest)
    if time.time() < deadline - 15:
        h_refined, val_refined = _single_flip_local_search(best_h, max_passes=8, deadline=deadline-5)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Block flip local search for larger structural changes
    if time.time() < deadline - 10:
        h_block, val_block = _block_sign_flip_local_search(best_h, block_sizes=[4, 8, 16, 32], max_passes=3, deadline=deadline-5)
        if val_block < best_score:
            best_score = val_block
            best_h = h_block.copy()
    
    # Final single flip polish
    if time.time() < deadline - 5:
        h_final, val_final = _single_flip_local_search(best_h, max_passes=5, deadline=deadline-1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
