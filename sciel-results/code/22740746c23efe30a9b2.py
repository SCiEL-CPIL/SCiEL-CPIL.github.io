"""Rewritten program: signed step heights minimizing max|autoconv|/(sum h)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """
    Smooth objective for optimization: soft_max_abs(autoconv(h)) / (sum h)^2.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_barker_like(n):
    """
    Generate a Barker-like sequence: +1/-1 with good autocorrelation properties.
    We use a pseudo-random sign sequence smoothed slightly.
    """
    rng = np.random.default_rng(123)
    h = rng.choice([-1.0, 1.0], size=n)
    # Light smoothing to help gradient methods
    kernel = np.array([0.2, 0.3, 0.3, 0.2])
    h = np.convolve(h, kernel, mode='same')
    return h

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_gaussian(n):
    """Generate a Gaussian-shaped sequence."""
    x = np.arange(n)
    mu = n / 2
    sigma = n / 4
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gen_random_smooth(n, seed=42):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(8) / 8
    return np.convolve(noise, kernel, mode='same')

def _autoconv_incremental_flip(h, ac_base, i):
    """
    Compute autoconv after flipping sign at index i, given base autoconv.
    h' = h - 2*h[i]*e_i
    ac(h') = ac(h) - 2*h[i]*(e_i * h) - 2*h[i]*(h * e_i) + 4*h[i]^2*(e_i*e_i)
    But e_i*e_i = delta at 0, so the last term only affects ac[0].
    Actually: ac(h')_k = sum_j h'_j h'_{j+k}
    = sum_j (h_j - 2*h[i]*delta_{j,i})(h_{j+k} - 2*h[i]*delta_{j+k,i})
    = ac(h)_k - 2*h[i]*h_{i+k} - 2*h[i]*h_{i-k} + 4*h[i]^2*delta_{k,0}
    where h_{i+k} is 0 if out of range, and h_{i-k} similarly.
    """
    n = len(h)
    ac = ac_base.copy()
    hi = h[i]
    # For each lag k, the change is -2*hi*h[i+k] - 2*hi*h[i-k] + 4*hi^2*delta_{k,0}
    # h[i+k] for k in range: valid when 0 <= i+k < n
    # h[i-k] for k in range: valid when 0 <= i-k < n
    
    # Vectorized: for lag k from -(n-1) to (n-1), but autoconv is indexed 0..2n-2
    # ac index k corresponds to lag (k - (n-1)) in centered form
    # Let's just compute directly for the affected lags
    
    # h[i+k] is nonzero for k in [-(i), n-1-i], i.e., ac indices [n-1-i, n-1+i]
    # h[i-k] is nonzero for k in [i-(n-1), i], i.e., ac indices [n-1+(i-(n-1)), n-1+i] = [i, n-1+i]
    
    # Simpler: just compute the correction for all k
    # For lag k (ac index = k + n - 1):
    #   term1 = h[i + (k - (n-1))] if 0 <= i + k - (n-1) < n else 0  -- this is h[i+k-(n-1)]... 
    # Let me use the standard: ac[h]_m = sum_{j} h[j]*h[j+m-n+1] for m=0..2n-2
    # Actually let's just do it the straightforward way for the O(n) affected indices
    
    # The flip at position i affects ac at lags where i or i+lag is in [0,n-1]
    # For lag l (can be negative), ac index = l + n - 1
    # h[i] * h[i+l] contributes when both in range
    # h[i-l] * h[i] contributes when both in range (this is the same as lag -l)
    
    # Let me just compute the full correction vector
    # delta_ac[k] for k = 0..2n-2:
    #   = -2*hi * h[i + (k - n + 1)] (if valid) - 2*hi * h[i - (k - n + 1)] (if valid) + 4*hi*hi (if k==0)
    
    lags = np.arange(2 * n - 1) - (n - 1)  # lags from -(n-1) to (n-1)
    
    # h[i + lag] valid when 0 <= i + lag < n
    idx1 = i + lags
    mask1 = (idx1 >= 0) & (idx1 < n)
    contrib1 = np.zeros(2 * n - 1)
    contrib1[mask1] = h[idx1[mask1]]
    
    # h[i - lag] valid when 0 <= i - lag < n
    idx2 = i - lags
    mask2 = (idx2 >= 0) & (idx2 < n)
    contrib2 = np.zeros(2 * n - 1)
    contrib2[mask2] = h[idx2[mask2]]
    
    delta = -2 * hi * (contrib1 + contrib2)
    delta[0] += 4 * hi * hi  # the e_i * e_i term at lag 0
    
    ac += delta
    return ac

def _local_search(h, max_passes=30, deadline=None):
    """
    Local search with incremental autoconv update for sign flips.
    """
    h = h.copy()
    best_val = _score(h)
    ac_base = _autoconv_fft(h)
    s = np.sum(h)
    rng = np.random.default_rng(42)
    size = 2 * N - 1
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try all sign flips using incremental update
        best_i = -1
        best_flip_val = best_val
        
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            ac_new = _autoconv_incremental_flip(h, ac_base, i)
            s_new = s - 2 * h[i]
            if abs(s_new) < 1e-9:
                continue
            val = np.max(np.abs(ac_new)) / (s_new * s_new)
            if val < best_flip_val * (1 - 1e-12):
                best_flip_val = val
                best_i = i
        
        if best_i >= 0:
            # Apply the best flip
            ac_base = _autoconv_incremental_flip(h, ac_base, best_i)
            s = s - 2 * h[best_i]
            h[best_i] = -h[best_i]
            best_val = best_flip_val
            improved = True
        
        # Try small multiplicative perturbations
        for _ in range(N // 2):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.integers(N)
            orig = h[i]
            for factor in [1.03, 0.97, 1.06, 0.94]:
                if deadline is not None and time.time() > deadline:
                    break
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    ac_base = _autoconv_fft(h)
                    s = np.sum(h)
                    improved = True
                    break
                else:
                    h[i] = orig
        
        if not improved:
            break
    
    return h, best_val

def _epigraph_objective(params, beta=50.0):
    """
    Epigraph formulation: params = [h (N), t (1)]
    Minimize t subject to |ac_k| <= t * (sum h)^2 for all k.
    Smooth surrogate: sum_k softplus((|ac_k| - t*(sum h)^2) * beta) + t
    """
    h = params[:-1]
    t = params[-1]
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    bound = t * s * s
    violations = np.abs(ac) - bound
    # Soft constraint: sum of softplus(violations * beta)
    sp = np.logaddexp(0, beta * violations) / beta
    return t + np.sum(sp)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_h = np.ones(N)
    
    base_score = _score(best_h)
    best_score = base_score
    
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    inits.append(_gen_gaussian(N))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Barker-like
    inits.append(_gen_barker_like(N))
    
    # 5. Random smoothed
    inits.append(_gen_random_smooth(N, seed=1))
    
    # 6. Another random with different seed
    inits.append(_gen_random_smooth(N, seed=99))
    
    # 7. Alternating signs with envelope
    x = np.arange(N)
    envelope = np.exp(-0.5 * ((x - N/2) / (N/5))**2)
    inits.append(envelope * np.cos(np.pi * x / 5))
    
    # 8. Random binary smoothed
    rng8 = np.random.default_rng(7)
    h8 = rng8.choice([-1.0, 1.0], size=N)
    kernel8 = np.ones(6) / 6
    inits.append(np.convolve(h8, kernel8, mode='same'))
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Standard smooth objective optimization
        for beta in [50, 100]:
            if time.time() > deadline - 10:
                break
            
            try:
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 200, 'disp': False, 'gtol': 1e-6, 'ftol': 1e-8})
                h_opt = res.x
                
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Epigraph formulation
        if time.time() > deadline - 10:
            continue
        try:
            s_init = np.sum(h_norm)
            t_init = _autoconv_max_abs(h_norm) / (s_init * s_init) * 1.1  # slightly above
            params_init = np.concatenate([h_norm, [t_init]])
            
            def obj_epi(params):
                return _epigraph_objective(params, beta=50.0)
            
            res = minimize(obj_epi, params_init, method='L-BFGS-B',
                           options={'maxiter': 300, 'disp': False, 'gtol': 1e-6, 'ftol': 1e-8})
            h_opt = res.x[:-1]
            if abs(np.sum(h_opt)) < 1e-6:
                continue
            val = _score(h_opt)
            if val < best_score:
                best_score = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final local search on best found - more passes since incremental is faster
    if time.time() < deadline - 5:
        h_refined, val_refined = _local_search(best_h, max_passes=50, deadline=deadline - 2)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # High-precision polish: run L-BFGS-B with tighter tolerance on the best result
    if time.time() < deadline - 3:
        try:
            s0 = np.sum(best_h)
            if abs(s0) > 1e-9:
                h_norm = best_h / s0
                res = minimize(lambda x: _objective_smooth(x, beta=200.0), h_norm,
                               method='L-BFGS-B',
                               options={'maxiter': 500, 'disp': False, 'gtol': 1e-8, 'ftol': 1e-10})
                h_opt = res.x
                if abs(np.sum(h_opt)) > 1e-9:
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Another local search pass after polish
    if time.time() < deadline - 2:
        h_refined2, val_refined2 = _local_search(best_h, max_passes=30, deadline=deadline - 1)
        if val_refined2 < best_score:
            best_score = val_refined2
            best_h = h_refined2.copy()
    
    # Safety: if somehow we lost track, fall back
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
