"""Combined program: sparse combinatorial support with alternating minimization + smooth surrogate + local search."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _max_abs_ac(h):
    return np.max(np.abs(_autoconv_full(h)))

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _max_abs_ac(h) / (s * s)

def _soft_max_abs(ac, beta):
    """Smooth approximation of max(|ac|) using log-sum-exp."""
    a = np.abs(ac)
    m = np.max(a)
    if m < 1e-15:
        return 0.0
    scaled = beta * (a - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _smooth_obj(h, beta):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_full(h)
    return _soft_max_abs(ac, beta) / (s * s)

def _local_search_flip(h, deadline, max_passes=10):
    """Local search: try flipping signs of individual elements."""
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline, block_size=4, max_passes=5):
    """Local search: try flipping signs of blocks of consecutive elements."""
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for start in range(0, n - block_size + 1, block_size):
            if time.time() > deadline:
                break
            orig = h[start:start+block_size].copy()
            h[start:start+block_size] = -orig
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[start:start+block_size] = orig
        if not improved:
            break
    return h, best_val

def _init_zadoff_chu_real(n):
    """Real part of Zadoff-Chu-like sequence, normalized."""
    i = np.arange(n)
    phase = np.pi * i * i / n
    h = np.cos(phase)
    s = np.sum(h)
    if abs(s) < 1e-9:
        h = h + 1e-6
        s = np.sum(h)
    return h / s

def _init_flat_top(n):
    """Flat-top-like window (Kaiser with high alpha)."""
    from scipy.special import i0
    x = np.arange(n)
    mid = (n - 1) / 2.0
    r = np.sqrt(1.0 - ((x - mid) / mid) ** 2)
    h = i0(14.0 * r) / i0(14.0)
    s = np.sum(h)
    return h / s

def _init_random_smooth(rng, n, smooth_len):
    """Random sequence, smoothed."""
    noise = rng.standard_normal(n)
    kernel = np.ones(smooth_len) / smooth_len
    h = np.convolve(noise, kernel, mode='same')
    s = np.sum(h)
    if abs(s) < 1e-6:
        h = h + 0.1
        s = np.sum(h)
    return h / s

def _sparse_comb_optimize(deadline, max_iter=50):
    """
    Sparse combinatorial support optimization.
    Represent h as sum of K shifted Dirac combs with continuous amplitudes.
    Alternating minimization over pulse locations and weights.
    """
    # Initialize with K sparse pulses
    K = 8  # Number of pulses
    rng = np.random.default_rng(123)
    
    # Random initial pulse locations
    pulse_locs = rng.integers(0, N, size=K)
    # Random initial amplitudes (positive for now)
    pulse_amps = rng.uniform(0.5, 1.5, size=K)
    
    best_val = np.inf
    best_h = None
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Build h from current pulses
        h = np.zeros(N)
        for k in range(K):
            loc = pulse_locs[k] % N
            h[loc] += pulse_amps[k]
        
        # Optimize amplitudes given locations (linear least squares)
        # We want to minimize max(|autoconv|) / (sum)^2
        # This is non-linear in amplitudes, so use gradient-based
        
        def amp_obj(amps):
            h_temp = np.zeros(N)
            for k in range(K):
                loc = pulse_locs[k] % N
                h_temp[loc] += amps[k]
            s = np.sum(h_temp)
            if abs(s) < 1e-9:
                return 1e10
            return _smooth_obj(h_temp, beta=50.0)
        
        try:
            res = minimize(amp_obj, pulse_amps, method='L-BFGS-B',
                           options={'maxiter': 50, 'disp': False})
            pulse_amps = res.x
        except Exception:
            pass
        
        # Optimize locations given amplitudes (combinatorial)
        # Try shifting each pulse by small amounts
        improved = False
        for k in range(K):
            if time.time() > deadline:
                break
            for delta in [-2, -1, 1, 2, -3, 3]:
                new_loc = (pulse_locs[k] + delta) % N
                old_loc = pulse_locs[k]
                pulse_locs[k] = new_loc
                
                h_temp = np.zeros(N)
                for kk in range(K):
                    h_temp[pulse_locs[kk] % N] += pulse_amps[kk]
                
                val = _smooth_obj(h_temp, beta=50.0)
                if val < best_val - 1e-6:
                    improved = True
                    best_val = val
                    best_h = h_temp.copy()
                else:
                    pulse_locs[k] = old_loc
        
        if not improved and iteration > 5:
            break
    
    # Normalize
    if best_h is not None:
        s = np.sum(best_h)
        if abs(s) > 1e-9:
            best_h = best_h / s
            val = _score(best_h)
            return best_h, val
    
    return None, np.inf

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin
    
    best_score = np.inf
    best_h = None
    
    def update_best(h):
        nonlocal best_score, best_h
        s = np.sum(h)
        if abs(s) < 1e-12:
            return
        val = _score(h)
        if val < best_score:
            best_score = val
            best_h = h.copy()
    
    # Baseline: uniform
    h_uniform = np.ones(N) / N
    update_best(h_uniform)
    
    # Initializations
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Kaiser window
    inits.append(_init_flat_top(N))
    
    # 2. Zadoff-Chu real
    inits.append(_init_zadoff_chu_real(N))
    
    # 3. Various Kaiser alphas
    from scipy.special import i0
    for alpha in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        x = np.arange(N)
        mid = (N - 1) / 2.0
        r = np.sqrt(1.0 - ((x - mid) / mid) ** 2)
        h = i0(alpha * r) / i0(alpha)
        s = np.sum(h)
        if abs(s) > 1e-9:
            inits.append(h / s)
    
    # 4. Random smooth
    for smooth_len in [5, 10, 20, 50]:
        inits.append(_init_random_smooth(rng, N, smooth_len))
    
    # 5. Chirp variants
    i = np.arange(N)
    for freq_scale in [1.0, 2.0, 0.5]:
        h = np.cos(np.pi * freq_scale * i * i / N)
        s = np.sum(h)
        if abs(s) > 1e-6:
            inits.append(h / s)
    
    # 6. Sparse comb initialization
    if time.time() < deadline - 30:
        h_sparse, val_sparse = _sparse_comb_optimize(deadline - 20)
        if h_sparse is not None:
            inits.append(h_sparse)
    
    # Optimize each initialization with multi-beta gradient optimization
    for h_init in inits:
        if time.time() > deadline - 15:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Multi-beta gradient optimization
        for beta in [10, 25, 50, 100, 200]:
            if time.time() > deadline - 10:
                break
            
            try:
                def obj(x, b=beta):
                    return _smooth_obj(x, b)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 200, 'disp': False, 'gtol': 1e-6, 'maxfun': 500})
                h_opt = res.x
                
                s_opt = np.sum(h_opt)
                if abs(s_opt) < 1e-6:
                    continue
                h_opt = h_opt / s_opt
                update_best(h_opt)
            except Exception:
                pass
    
    # Local refinement on best
    if best_h is not None and time.time() < deadline - 5:
        h_ref, val_ref = _local_search_flip(best_h, deadline - 2, max_passes=8)
        update_best(h_ref)
    
    # Block local search
    if best_h is not None and time.time() < deadline - 2:
        h_ref, val_ref = _local_search_block(best_h, deadline - 1, block_size=4, max_passes=4)
        update_best(h_ref)
    
    if best_h is None:
        return np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
