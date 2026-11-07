"""Second autocorrelation inequality - optimized with structured multi-scale search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _h_from_params(params):
    """Convert log-params to normalized nonnegative heights via softmax."""
    p = np.asarray(params, dtype=float)
    p_shift = p - np.max(p)
    e = np.exp(p_shift)
    s = e.sum()
    if s < 1e-300:
        return np.ones(N) / N
    return e / s

def _neg_value(params):
    return -_compute_value(_h_from_params(params))

def _coordinate_descent(h, max_sweeps=15, tol=1e-12, deadline=None):
    """Coordinate descent with golden section search per coordinate."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 1e-12)
    h = h / h.sum()
    best = _compute_value(h)
    idx = np.arange(N)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 3:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 3:
                break
            old_val = _compute_value(h)
            old_hi = h[i]
            
            lo = 1e-10
            hi = max(3.0 * old_hi + 0.2, 1.0)
            
            gr = (np.sqrt(5) - 1) / 2
            a = lo + (1 - gr) * (hi - lo)
            b = lo + gr * (hi - lo)
            fa = _compute_value(np.where(idx == i, a, h))
            fb = _compute_value(np.where(idx == i, b, h))
            
            for _ in range(35):
                if deadline is not None and time.time() > deadline - 3:
                    break
                if fa < fb:
                    hi = b
                    b = a
                    fb = fa
                    a = lo + (1 - gr) * (hi - lo)
                    fa = _compute_value(np.where(idx == i, a, h))
                else:
                    lo = a
                    a = b
                    fa = fb
                    b = lo + gr * (hi - lo)
                    fb = _compute_value(np.where(idx == i, b, h))
            
            best_i = (a + b) / 2
            h_trial = np.where(idx == i, best_i, h)
            h_trial = h_trial / h_trial.sum()
            val_i = _compute_value(h_trial)
            
            if val_i > old_val + tol:
                h = h_trial
                improved = True
                best = val_i
        
        if not improved:
            break
    
    return h, best

def _generate_candidates(rng):
    """Generate a diverse set of candidate initial points."""
    candidates = []
    x = np.linspace(-1, 1, N)
    xh = np.linspace(-0.5, 0.5, N)
    idx = np.arange(N)
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Gaussian family (various widths)
    for a in [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        candidates.append(np.exp(-a * x**2))
    
    # Super-Gaussian / sub-Gaussian
    for p in [0.5, 1, 1.5, 2, 3, 4, 6, 8]:
        for a in [1, 2, 4, 8, 16]:
            candidates.append(np.exp(-a * np.abs(x)**p))
    
    # Polynomial / power shapes
    for k in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12]:
        candidates.append(np.maximum(0, 1 - np.abs(x)**k))
    
    # Cosine-based
    for mult in [0.5, 1, 1.5, 2, 2.5, 3]:
        candidates.append(np.maximum(0, np.cos(mult * np.pi * xh)))
        candidates.append(np.maximum(0, np.cos(mult * np.pi * xh)**2))
    
    # Sine-based
    for mult in [0.5, 1, 1.5, 2, 2.5, 3]:
        candidates.append(np.maximum(0, np.sin(mult * np.pi * xh)))
        candidates.append(np.maximum(0, np.sin(mult * np.pi * xh)**2))
    
    # Bimodal
    for c in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for a in [4, 8, 12, 20, 40, 80, 120]:
            candidates.append(np.exp(-a*(x-c)**2) + np.exp(-a*(x+c)**2))
    
    # Plateau with smooth edges
    for start in [0, 2, 5, 8, 10, 12, 15]:
        for end in [N-start, N-2, N-5, N-8, N-10]:
            if end > start + 4:
                plat = np.zeros(N)
                plat[start:end] = 1.0
                edge = 3
                for j in range(start, min(start+edge, N)):
                    plat[j] = (j - start + 1) / (edge + 1)
                for j in range(max(end-edge, 0), end):
                    plat[j] = (end - j) / (edge + 1)
                candidates.append(np.maximum(plat, 1e-8))
    
    # Triangle
    candidates.append(np.maximum(0, 1 - 2*np.abs(xh)))
    candidates.append(np.maximum(0, 1 - 4*np.abs(xh)))
    candidates.append(np.maximum(0, 1 - 1.5*np.abs(xh)))
    
    # Sech / Lorentzian-like
    for a in [1, 2, 3, 4, 6, 8, 12]:
        candidates.append(1.0 / np.cosh(a * x))
        candidates.append(1.0 / (1 + (a*x)**2))
    
    # Asymmetric
    for asym in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
        candidates.append(np.exp(-x**2 * (1 + asym*x)))
        candidates.append(np.exp(-x**2 * (1 - asym*x)))
    
    # W-shaped / multi-lobe
    for nlobes in [3, 4, 5]:
        for width in [0.1, 0.15, 0.2, 0.3, 0.4]:
            h_w = np.zeros(N)
            for k in range(nlobes):
                c = -1 + 2*k/(nlobes-1)
                h_w += np.exp(-(x-c)**2/(2*width**2))
            candidates.append(h_w)
    
    # Random smooth (low-frequency Fourier)
    for _ in range(40):
        nfreq = rng.integers(2, 10)
        coeffs = rng.standard_normal(nfreq)
        phases = rng.uniform(0, 2*np.pi, nfreq)
        h_f = np.ones(N)
        for k in range(1, nfreq+1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh + phases[k-1])
        candidates.append(np.abs(h_f) + 0.01)
    
    # Random (raw)
    for _ in range(30):
        h_r = np.abs(rng.standard_normal(N)) + 0.01
        candidates.append(h_r)
    
    # Concentrated center
    for width in [2, 3, 5, 7, 10, 15, 20]:
        center = N // 2
        candidates.append(np.exp(-((idx - center)**2) / (2*width**2)))
    
    # Edge-concentrated
    for width in [2, 3, 5, 7, 10]:
        candidates.append(np.exp(-((idx)**2) / (2*width**2)) + np.exp(-((idx - (N-1))**2) / (2*width**2)))
    
    # Single spike
    for pos in [0, 5, 10, 15, 20, 24, 25, 29, 35, 40, 45, 49]:
        spike = np.zeros(N)
        spike[pos] = 1.0
        candidates.append(spike + 1e-6)
    
    # Two spikes
    for p1 in [5, 10, 15, 20, 25]:
        for p2 in [30, 35, 40, 45, 49]:
            if p2 > p1:
                two = np.zeros(N)
                two[p1] = 1.0
                two[p2] = 1.0
                candidates.append(two + 1e-6)
    
    # Step functions
    for cut in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
        step = np.zeros(N)
        step[cut:] = 1.0
        candidates.append(step + 1e-6)
        step2 = np.zeros(N)
        step2[:cut] = 1.0
        candidates.append(step2 + 1e-6)
    
    # Power-law decay from center
    for alpha in [0.5, 1, 1.5, 2, 3, 4]:
        dist = np.abs(idx - (N-1)/2.0)
        candidates.append(1.0 / (1 + dist**alpha))
    
    # Alternating
    for freq in [1, 2, 3, 5]:
        alt = 1 + 0.5*np.cos(2*np.pi*freq*idx/N)
        candidates.append(np.maximum(alt, 1e-6))
    
    return candidates

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    best_h = np.ones(N) / N
    best_val = _compute_value(best_h)
    
    def try_optimize(params0, method='Nelder-Mead', maxiter=2000, **kwargs):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        try:
            if method == 'Nelder-Mead':
                opts = {'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            elif method == 'L-BFGS-B':
                opts = {'maxiter': maxiter, 'gtol': 1e-8}
            elif method == 'Powell':
                opts = {'maxiter': maxiter, 'xtol': 1e-10, 'ftol': 1e-13}
            elif method == 'CG':
                opts = {'maxiter': maxiter, 'gtol': 1e-8}
            else:
                opts = {}
            res = minimize(_neg_value, params0, method=method, options=opts)
            if res.fun < -best_val + 1e-15:
                best_val = -res.fun
                best_h = _h_from_params(res.x)
        except Exception:
            pass
    
    def try_coordinate(h0, max_sweeps=15):
        nonlocal best_h, best_val
        if time.time() > deadline - 3:
            return
        try:
            h_new, val_new = _coordinate_descent(h0, max_sweeps=max_sweeps, deadline=deadline)
            if val_new > best_val + 1e-15:
                best_val = val_new
                best_h = h_new
        except Exception:
            pass
    
    rng = np.random.default_rng(50)
    
    # Generate candidates
    candidates = _generate_candidates(rng)
    
    # Phase 1: Quick NM from diverse candidates (first ~60s)
    phase1_deadline = t0 + 60
    count = 0
    for h0 in candidates:
        if time.time() > phase1_deadline:
            break
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 1e-8)
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        try_optimize(params0, 'Nelder-Mead', maxiter=800)
        count += 1
    
    # Phase 2: Coordinate descent on best
    try_coordinate(best_h, max_sweeps=20)
    
    # Phase 3: Powell refinement
    if time.time() < deadline - 30:
        try_optimize(np.log(best_h), 'Powell', maxiter=400)
    
    # Phase 4: More random restarts with different seeds
    phase4_deadline = t0 + 95
    for seed in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]:
        if time.time() > phase4_deadline:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > phase4_deadline:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.01
            h0 = h0 / h0.sum()
            params0 = np.log(h0)
            try_optimize(params0, 'Nelder-Mead', maxiter=600)
    
    # Phase 5: Coordinate descent again
    try_coordinate(best_h, max_sweeps=20)
    
    # Phase 6: L-BFGS-B polish
    if time.time() < deadline - 15:
        try_optimize(np.log(best_h), 'L-BFGS-B', maxiter=800)
    
    # Phase 7: More structured restarts
    phase7_deadline = t0 + 125
    structured_seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 2020, 3030]
    for seed in structured_seeds:
        if time.time() > phase7_deadline:
            break
        rng3 = np.random.default_rng(seed)
        # Smooth random
        nfreq = rng3.integers(3, 8)
        coeffs = rng3.standard_normal(nfreq)
        phases = rng3.uniform(0, 2*np.pi, nfreq)
        xh = np.linspace(-0.5, 0.5, N)
        h_f = np.ones(N)
        for k in range(1, nfreq+1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh + phases[k-1])
        h_f = np.abs(h_f) + 0.01
        h_f = h_f / h_f.sum()
        try_optimize(np.log(h_f), 'Nelder-Mead', maxiter=500)
    
    # Phase 8: Coordinate descent + Powell polish
    try_coordinate(best_h, max_sweeps=20)
    if time.time() < deadline - 8:
        try_optimize(np.log(best_h), 'Powell', maxiter=200)
    
    # Phase 9: Final coordinate descent
    try_coordinate(best_h, max_sweeps=15)
    
    # Phase 10: Final NM
    if time.time() < deadline - 3:
        try_optimize(np.log(best_h), 'Nelder-Mead', maxiter=500)
    
    # Ensure nonnegativity and normalization
    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
# EVOLVE-BLOCK-END
