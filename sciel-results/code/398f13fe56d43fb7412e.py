"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _epigraph_objective(x, beta=100.0):
    """
    Epigraph-formulation objective: minimize t subject to soft_max(|ac|) <= t*(sum h)^2.
    We use a penalty/soft-constraint approach: return t + penalty*(max(0, soft_max - t*s^2)).
    x = [h_0, ..., h_{N-1}, t]
    """
    h = x[:-1]
    t = x[-1]
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    constraint_violation = max(0.0, sm - t * s * s)
    # Large penalty for constraint violation
    return t + 100.0 * constraint_violation

def _epigraph_gradient(x, beta=100.0):
    """
    Gradient of the epigraph objective (numerical approximation for robustness).
    """
    h = x[:-1]
    t = x[-1]
    s = np.sum(h)
    if abs(s) < 1e-9:
        return np.ones_like(x)
    
    ac = _autoconv_fft(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    # Compute gradient of soft_max_abs w.r.t. h
    # d(soft_max)/dh_i = sum_j (beta * sign(ac_j) * exp(beta*(|ac_j|-m))) / sum_k exp(beta*(|ac_k|-m)) * dac_j/dh_i
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    weights = np.exp(scaled)
    w_sum = np.sum(weights)
    
    # dac_j/dh_i for autoconvolution: ac_j = sum_k h_k * h_{j-k}
    # Gradient is 2 * circulant convolution of h with delta at appropriate shifts
    # For efficiency, use FFT-based gradient
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # Gradient of autoconvolution w.r.t. h: 2 * ifft(H * fft(h_shifted))
    # More precisely, d(ac_j)/dh_i = h_{j-i} + h_{j-i} (by symmetry) = 2*h_{j-i} for valid indices
    # Using FFT: grad = 2 * ifft(H * conj(FFT(h_shifted)))
    H = np.fft.rfft(h, fsize)
    
    # Gradient of soft_max w.r.t. ac:
    grad_ac = weights * np.sign(ac) / w_sum
    # Extend grad_ac to full FFT size
    grad_ac_full = np.zeros(fsize // 2 + 1)
    grad_ac_full[:size] = grad_ac
    
    # d(soft_max)/dh = sum_j grad_ac_j * dac_j/dh
    # dac_j/dh_i = h_{j-i} (for linear autoconv, it's the convolution kernel)
    # This is equivalent to 2 * circular convolution of h with grad_ac (truncated)
    # Using FFT: grad_h = 2 * ifft(H * conj(FFT(grad_ac_full)))... 
    # Actually simpler: grad_h[i] = 2 * sum_j grad_ac[j] * h[j-i] (with proper indexing)
    
    # Let's use a direct but vectorized approach for the gradient
    # ac[j] = sum_k h[k] * h[j-k]
    # d(soft_max)/dh[i] = sum_j grad_ac[j] * (h[j-i] + h[j-i]) = 2 * sum_j grad_ac[j] * h[j-i]
    
    # For efficiency, use FFT-based cross-correlation
    grad_h = np.zeros(n)
    # grad_h[i] = 2 * sum_j grad_ac[j] * h[j-i] where j-i is in [0, n-1]
    # This is a convolution: grad_h = 2 * convolve(grad_ac, h, mode='valid') adjusted
    
    # Simpler: use the fact that autoconv gradient is 2*correlation
    # Let's just do it directly for N=400 (400*799 ~ 320k ops, fast enough)
    for i in range(n):
        # d(ac[j])/dh[i] = h[j-i] + h[j-i] for j where 0<=j-i<n and 0<=j<n... 
        # Actually ac[j] = sum_k h[k]*h[j-k], so d(ac[j])/dh[i] = h[j-i] if 0<=j-i<n else 0
        # Plus by symmetry of autoconv, also h[i-j]... no, autoconv is h*h not h*conj(h)
        # d(ac[j])/dh[i] = h[j-i] (if 0 <= j-i < n)
        # So grad_h[i] = sum_j grad_ac[j] * h[j-i] * (0<=j-i<n)
        # = sum_{k=0}^{n-1} grad_ac[i+k] * h[k] if i+k < size
        valid_j = np.arange(max(0, i), min(size, i + n))
        valid_k = valid_j - i
        grad_h[i] = 2.0 * np.sum(grad_ac[valid_j] * h[valid_k])
    
    # Gradient of objective w.r.t. h:
    # obj = t + 100 * max(0, sm - t*s^2)
    # If sm > t*s^2: d(obj)/dh = 100 * d(sm)/dh - 100*t*2*s*dh
    # else: d(obj)/dh = 0 (but t gradient is 1)
    
    constraint_active = sm > t * s * s
    if constraint_active:
        grad_h_obj = 100.0 * grad_h - 100.0 * t * 2.0 * s
    else:
        grad_h_obj = np.zeros(n)
    
    # Gradient w.r.t. t:
    if constraint_active:
        grad_t = 1.0 - 100.0 * s * s
    else:
        grad_t = 1.0
    
    grad = np.concatenate([grad_h_obj, [grad_t]])
    return grad

def _golden_section_search(func, a, b, tol=1e-6, max_iter=50):
    """1-D golden-section search to minimize func on [a, b]."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = func(c)
    fd = func(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = func(d)
    return (a + b) / 2.0

def _gen_zadoff_chu(n, rate=1.0):
    """
    Generate real part of Zadoff-Chu-like sequence with variable chirp rate.
    h[i] = cos(rate * pi * i^2 / n)
    """
    i = np.arange(n)
    phase = rate * np.pi * i**2 / n
    return np.cos(phase)

def _gen_barker_like(n):
    """
    Generate a sequence with good autocorrelation properties.
    """
    h = np.zeros(n)
    for i in range(n):
        if i % 4 == 0:
            h[i] = 1.0
        elif i % 4 == 1:
            h[i] = -1.0
        elif i % 4 == 2:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _block_sign_flip_local_search(h, max_passes=10, deadline=None, block_size=8):
    """
    Local optimization: try flipping blocks of consecutive signs, keep if improves score.
    More efficient than single flips.
    """
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for start in range(0, n - block_size + 1, block_size):
            if deadline is not None and time.time() > deadline:
                break
            # Flip a block of block_size consecutive elements
            h[start:start+block_size] = -h[start:start+block_size]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[start:start+block_size] = -h[start:start+block_size]
        # Also try single flips for fine-tuning
        for i in range(n):
            if deadline is not None and time.time() > deadline:
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

def _single_flip_local_search(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
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

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
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
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Triangle window
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Zadoff-Chu chirp with golden-section rate search
    def _chirp_score(rate):
        h = _gen_zadoff_chu(N, rate)
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        return _score(h)
    
    best_rate = _golden_section_search(_chirp_score, 0.1, 3.0, tol=0.01, max_iter=30)
    inits.append(_gen_zadoff_chu(N, best_rate))
    
    # Also add a few fixed-rate chirps
    for r in [0.5, 1.0, 1.5, 2.0]:
        inits.append(_gen_zadoff_chu(N, r))
    
    # 6. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 7. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.standard_normal(N)
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    # Optimization loop using epigraph formulation
    for h_init in inits:
        if time.time() > deadline - 25:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1
        h_norm = h_init / s0
        
        # Initial t estimate
        ac0 = _autoconv_fft(h_norm)
        t_init = np.max(np.abs(ac0)) + 0.1
        
        # Epigraph optimization
        try:
            x0 = np.concatenate([h_norm, [t_init]])
            
            for beta in [50, 100, 200]:
                if time.time() > deadline - 15:
                    break
                
                def obj(x):
                    return _epigraph_objective(x, beta=beta)
                
                res = minimize(obj, x0, method='L-BFGS-B', 
                               jac=_epigraph_gradient,
                               options={'maxiter': 80, 'disp': False, 'gtol': 1e-5})
                x_opt = res.x
                h_opt = x_opt[:-1]
                
                # Check validity and normalize
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
                        x0 = np.concatenate([h_opt, [max(0.01, val + 0.1)]])
                else:
                    # Fallback: try the raw h_opt
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
        except Exception:
            pass

    # Also try L-BFGS with soft-max objective as backup
    if time.time() < deadline - 30:
        for h_init in inits[:6]:  # Only first few inits
            if time.time() > deadline - 20:
                break
            s0 = np.sum(h_init)
            if abs(s0) < 1e-6:
                continue
            h_norm = h_init / s0
            try:
                for beta in [30, 60]:
                    if time.time() > deadline - 10:
                        break
                    def obj_soft(x):
                        s = np.sum(x)
                        if abs(s) < 1e-9:
                            return 1e10
                        ac = _autoconv_fft(x)
                        return _soft_max_abs(ac, beta) / (s * s)
                    
                    res = minimize(obj_soft, h_norm, method='L-BFGS-B', 
                                   options={'maxiter': 60, 'disp': False, 'gtol': 1e-5})
                    h_opt = res.x
                    s_opt = np.sum(h_opt)
                    if abs(s_opt) > 1e-6:
                        h_opt = h_opt / s_opt
                        val = _score(h_opt)
                        if val < best_score:
                            best_score = val
                            best_h = h_opt.copy()
            except Exception:
                pass

    # Block sign-flip local search refinement
    if time.time() < deadline - 10:
        h_refined, val_refined = _block_sign_flip_local_search(best_h, max_passes=8, deadline=deadline-5, block_size=8)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()

    # Final single-flip polish
    if time.time() < deadline - 3:
        h_polished, val_polished = _single_flip_local_search(best_h, max_passes=3, deadline=deadline-1)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
