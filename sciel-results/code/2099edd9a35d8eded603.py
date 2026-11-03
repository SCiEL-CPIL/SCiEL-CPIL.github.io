"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    # Autoconvolution is convolution of h with h
    # In frequency domain: H * H
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

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|).
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective for optimization.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _epigraph_objective(x, beta=50.0, lam=100.0):
    """
    Epigraph formulation: minimize t + lam * penalty
    where t >= max|ac| / s^2
    x = [h, t] with h in R^N, t in R
    """
    h = x[:N]
    t = x[N]
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    ratio = sm / (s * s)
    # Penalty for t < ratio
    violation = max(0.0, ratio - t)
    return t + lam * violation

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    # Use a quadratic phase
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_zadoff_chu(n, root=1):
    """Generate Zadoff-Chu-like sequence: real part of complex exponential."""
    i = np.arange(n)
    # Zadoff-Chu: x[k] = exp(-j*pi*root*k*(k+1)/n)
    phase = -np.pi * root * i * (i + 1) / n
    return np.cos(phase)

def _gen_zadoff_chu_optimized(n, root=1, max_iter=200):
    """
    Optimize phases of a complex exponential sum to minimize PAPR of autoconvolution.
    h[i] = cos(phi[i]) where phi is optimized.
    """
    i = np.arange(n)
    # Initial phases from Zadoff-Chu
    phi = -np.pi * root * i * (i + 1) / n
    
    def obj_phases(phi):
        h = np.cos(phi)
        s = np.sum(h)
        if abs(s) < 1e-9:
            return 1e10
        ac = _autoconv_fft(h)
        sm = _soft_max_abs(ac, beta=30.0)
        return sm / (s * s)
    
    try:
        res = minimize(obj_phases, phi, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'disp': False, 'gtol': 1e-5})
        if res.fun < obj_phases(phi):
            return np.cos(res.x)
    except Exception:
        pass
    return np.cos(phi)

def _gen_barker_like(n):
    """
    Generate a Barker-like sequence: maximize autocorrelation flatness.
    Use greedy construction.
    """
    h = np.zeros(n)
    h[0] = 1.0
    for k in range(1, n):
        # Try both signs, pick the one that minimizes max |ac|
        best_sign = 1.0
        best_val = np.inf
        for sign in [1.0, -1.0]:
            h[k] = sign
            ac = _autoconv_fft(h[:k+1])
            val = np.max(np.abs(ac))
            if val < best_val:
                best_val = val
                best_sign = sign
        h[k] = best_sign
    return h

def _gen_sinc(n):
    """Generate a windowed sinc sequence."""
    i = np.arange(n)
    center = n / 2
    x = (i - center)
    sinc = np.ones_like(x)
    nonzero = x != 0
    sinc[nonzero] = np.sin(np.pi * x[nonzero]) / (np.pi * x[nonzero])
    sigma = n / 4
    return np.exp(-0.5 * (x / sigma) ** 2)

def _local_search(h, max_passes=15, deadline=None):
    """
    Local search: try flipping signs or small perturbations.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try flipping signs
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
            # Try small perturbations
            for i in range(N):
                if deadline is not None and time.time() > deadline:
                    break
                orig = h[i]
                h[i] = orig * 1.05
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = orig
                h[i] = orig * 0.95
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = orig
        if not improved:
            break
            
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin: 5s before 150s limit
    
    best_score = np.inf
    best_h = np.ones(N) # Default: constant sequence
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    x = np.arange(N)
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-0.5 * ((x - mu) / sigma) ** 2))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Zadoff-Chu (root=1)
    inits.append(_gen_zadoff_chu(N, root=1))
    
    # 5. Zadoff-Chu (root=2) 
    inits.append(_gen_zadoff_chu(N, root=2))
    
    # 6. Sinc (windowed)
    inits.append(_gen_sinc(N))
    
    # 7. Barker-like
    inits.append(_gen_barker_like(N))
    
    # 8. Random smoothed
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(N)
    kernel = np.ones(10)/10
    smooth = np.convolve(noise, kernel, mode='same')
    inits.append(smooth)
    
    for idx, h_init in enumerate(inits):
        if time.time() > deadline - 25:  # Reserve 25s for final polish
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Optimize with smooth surrogate
        try:
            for beta in [30, 60]:
                if time.time() > deadline - 15:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B', 
                               options={'maxiter': 300, 'disp': False, 'gtol': 1e-5})
                h_opt = res.x
                
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
        except Exception:
            pass
        
        # Try epigraph formulation for the best candidate so far
        if idx < 4 and time.time() < deadline - 20:
            try:
                h_ep_init = np.concatenate([best_h, [best_score]])
                res_ep = minimize(_epigraph_objective, h_ep_init, method='L-BFGS-B',
                                  options={'maxiter': 100, 'disp': False, 'gtol': 1e-4})
                h_ep = res_ep.x[:N]
                if abs(np.sum(h_ep)) > 1e-6:
                    val_ep = _score(h_ep)
                    if val_ep < best_score:
                        best_score = val_ep
                        best_h = h_ep.copy()
            except Exception:
                pass

    # Final local search
    if time.time() < deadline - 8:
        h_refined, val_refined = _local_search(best_h, max_passes=12, deadline=deadline-5)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # High-precision polish: tighter tolerance
    if time.time() < deadline - 3:
        try:
            h_polish = best_h.copy()
            def obj_tight(x):
                return _objective_smooth(x, beta=100.0)
            res = minimize(obj_tight, h_polish, method='L-BFGS-B',
                           options={'maxiter': 500, 'disp': False, 'gtol': 1e-7})
            if abs(np.sum(res.x)) > 1e-6:
                val_polish = _score(res.x)
                if val_polish < best_score:
                    best_score = val_polish
                    best_h = res.x.copy()
        except Exception:
            pass

    # Final check
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
