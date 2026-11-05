"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _value(h):
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def construct():
    t_start = time.time()
    deadline = 120.0
    
    # Candidate 1: Raised cosine window
    i = np.arange(N)
    center = (N - 1) / 2.0
    # Full-width cosine: h[i] = 0.5 * (1 + cos(pi * (i - center) / (N/2)))
    # This is a Hann window
    h_hann = 0.5 * (1 + np.cos(np.pi * (i - center) / (N / 2.0)))
    # Ensure non-negative and finite
    h_hann = np.maximum(h_hann, 0)
    if np.sum(h_hann) == 0:
        h_hann = np.ones(N)
    v_hann = _value(h_hann)
    
    # Candidate 2: Parabolic window
    x = 2.0 * i / (N - 1) - 1.0  # in [-1, 1]
    h_para = np.maximum(1.0 - x**2, 0)
    if np.sum(h_para) == 0:
        h_para = np.ones(N)
    v_para = _value(h_para)
    
    # Candidate 3: Triangular (Bartlett) window
    h_tri = np.maximum(1.0 - np.abs(x), 0)
    if np.sum(h_tri) == 0:
        h_tri = np.ones(N)
    v_tri = _value(h_tri)
    
    # Candidate 4: Gaussian window with varying sigma
    best_gauss = None
    best_gauss_v = float('inf')
    for sigma_frac in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
        sigma = sigma_frac * N
        h_gauss = np.exp(-0.5 * ((i - center) / sigma) ** 2)
        v_g = _value(h_gauss)
        if v_g < best_gauss_v:
            best_gauss_v = v_g
            best_gauss = h_gauss.copy()
    
    # Track best so far
    candidates = [
        (v_hann, h_hann),
        (v_para, h_para),
        (v_tri, h_tri),
        (best_gauss_v, best_gauss),
        (2.0, np.ones(N)),  # seed
    ]
    candidates = [(v, h) for v, h in candidates if h is not None and np.all(np.isfinite(h)) and np.sum(h) != 0]
    best_v, best_h = min(candidates, key=lambda x: x[0])
    
    # Try scipy optimization on a parametric family if time permits
    if time.time() - t_start < deadline - 10.0:
        # Optimize a Gaussian-like shape: h[i] = exp(-a*(i-center)^2) * (1 + b*cos(2*pi*(i-center)/N))
        # Parameters: a (width), b (modulation)
        def obj(params):
            a, b = params
            if a < 0:
                return 1e10
            h = np.exp(-a * (i - center) ** 2) * (1 + b * np.cos(2 * np.pi * (i - center) / N))
            h = np.maximum(h, 0)
            if np.sum(h) == 0:
                return 1e10
            return _value(h)
        
        try:
            x0 = [1.0 / (0.3 * N) ** 2, 0.0]
            res = minimize(obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_v:
                a_opt, b_opt = res.x
                if a_opt >= 0:
                    h_opt = np.exp(-a_opt * (i - center) ** 2) * (1 + b_opt * np.cos(2 * np.pi * (i - center) / N))
                    h_opt = np.maximum(h_opt, 0)
                    if np.sum(h_opt) != 0 and np.all(np.isfinite(h_opt)):
                        best_v = res.fun
                        best_h = h_opt
        except Exception:
            pass
        
        # Try another family: h[i] = (1 + c1*cos(2*pi*i/N) + c2*cos(4*pi*i/N))
        if time.time() - t_start < deadline - 5.0:
            def obj2(params):
                c1, c2 = params
                h = 1.0 + c1 * np.cos(2 * np.pi * i / N) + c2 * np.cos(4 * np.pi * i / N)
                h = np.maximum(h, 0)
                if np.sum(h) == 0:
                    return 1e10
                return _value(h)
            
            try:
                x0_2 = [0.5, 0.1]
                res2 = minimize(obj2, x0_2, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res2.fun < best_v:
                    c1_opt, c2_opt = res2.x
                    h_opt2 = 1.0 + c1_opt * np.cos(2 * np.pi * i / N) + c2_opt * np.cos(4 * np.pi * i / N)
                    h_opt2 = np.maximum(h_opt2, 0)
                    if np.sum(h_opt2) != 0 and np.all(np.isfinite(h_opt2)):
                        best_v = res2.fun
                        best_h = h_opt2
            except Exception:
                pass
    
    # Safety: ensure valid output
    if best_h is None or not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
