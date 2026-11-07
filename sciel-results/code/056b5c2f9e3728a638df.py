import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = _START + 145.0  # safety margin under 150s kill

def _eval(h):
    h = np.asarray(h, dtype=float)
    if h.size != N:
        return -np.inf
    if not np.all(np.isfinite(h)):
        return -np.inf
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    n1 = np.sum(g) / (m + 1)
    ninf = np.max(g)
    if n1 * ninf < 1e-30:
        return -np.inf
    return l2 / (n1 * ninf)

def _neg_eval(h):
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    v = _eval(h)
    if v == -np.inf:
        return 1e10
    return -v

def _lbfgs(h0, max_iter=300):
    try:
        bounds = [(0, 20)] * N
        res = minimize(_neg_eval, h0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _eval(h)
            if np.isfinite(v):
                return h, v
    except Exception:
        pass
    return h0, _eval(h0)

def _nelder_mead(h0, max_iter=200):
    try:
        res = minimize(_neg_eval, h0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-10})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _eval(h)
            if np.isfinite(v):
                return h, v
    except Exception:
        pass
    return h0, _eval(h0)

def _cd(h0, budget):
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    best = _eval(h)
    step = 0.5
    while time.time() < budget:
        improved = False
        for i in range(N):
            for d in (step, -step):
                cand = h.copy()
                cand[i] = max(0.0, cand[i] + d)
                v = _eval(cand)
                if v > best + 1e-12:
                    h = cand
                    best = v
                    improved = True
        if not improved:
            step *= 0.6
            if step < 1e-7:
                break
    return h, best

def _make_initial_guesses():
    t = np.linspace(-0.5, 0.5, N)
    guesses = []
    
    # Uniform
    guesses.append(np.ones(N))
    
    # Gaussian bumps at center
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
        guesses.append(np.exp(-t**2 / (2 * sigma**2)))
    
    # Cosine bumps
    for width in [0.3, 0.5, 0.7, 1.0, 1.5]:
        guesses.append(np.maximum(np.cos(np.pi * t / width), 0.0))
    
    # Tapered
    for a in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        guesses.append(np.maximum(1.0 - a * np.abs(t), 0.0))
        guesses.append(np.maximum(1.0 - a * t**2, 0.0))
    
    # Power law
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        guesses.append(np.maximum(1.0 - (2.0 * np.abs(t))**p, 0.0))
    
    # Exponential
    for a in [1.0, 2.0, 4.0, 8.0, 16.0]:
        guesses.append(np.exp(-a * np.abs(t)))
        guesses.append(np.exp(-a * t**2))
    
    # Lorentzian
    for c in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        guesses.append(1.0 / (1.0 + c * t**2))
    
    # Sine-based
    for a in [0.5, 1.0, 2.0, 3.0]:
        guesses.append(np.maximum(np.sin(np.pi * t) ** a, 0.0))
        guesses.append(np.maximum(np.cos(np.pi * t * 0.5) ** a, 0.0))
    
    # Two-bump shapes
    for d in [0.1, 0.2, 0.3]:
        guesses.append(np.exp(-t**2 / (2 * 0.15**2)) + np.exp(-(t-d)**2 / (2 * 0.15**2)))
        guesses.append(np.exp(-t**2 / (2 * 0.15**2)) + np.exp(-(t+d)**2 / (2 * 0.15**2)))
    
    # Random smooth guesses
    rng = np.random.RandomState(42)
    for _ in range(10):
        x = np.linspace(-1, 1, N)
        coeffs = rng.randn(5)
        poly = np.zeros(N)
        for k, c in enumerate(coeffs):
            poly += c * np.cos(k * np.pi * x)
        guesses.append(np.maximum(poly, 0.0))
    
    return guesses

def construct():
    # Phase 1: Evaluate all initial guesses
    guesses = _make_initial_guesses()
    best_h = np.ones(N)
    best_v = _eval(best_h)
    
    for g in guesses:
        g = np.asarray(g, dtype=float)
        if not np.all(np.isfinite(g)):
            continue
        g = np.maximum(g, 0.0)
        if np.max(g) < 1e-12:
            continue
        v = _eval(g)
        if v > best_v:
            best_v = v
            best_h = g.copy()
    
    # Phase 2: L-BFGS-B from best guess
    if time.time() < _DEADLINE - 60:
        h, v = _lbfgs(best_h, max_iter=400)
        if v > best_v:
            best_v, best_h = v, h
    
    # Phase 3: Coordinate descent refinement
    budget = min(time.time() + 25.0, _DEADLINE - 35.0)
    if time.time() < budget:
        h, v = _cd(best_h, budget)
        if v > best_v:
            best_v, best_h = v, h
    
    # Phase 4: L-BFGS-B again from refined point
    if time.time() < _DEADLINE - 20:
        h, v = _lbfgs(best_h, max_iter=300)
        if v > best_v:
            best_v, best_h = v, h
    
    # Phase 5: Nelder-Mead polish
    if time.time() < _DEADLINE - 10:
        h, v = _nelder_mead(best_h, max_iter=150)
        if v > best_v:
            best_v, best_h = v, h
    
    # Phase 6: Final coordinate descent
    budget = min(time.time() + 8.0, _DEADLINE - 3.0)
    if time.time() < budget:
        h, v = _cd(best_h, budget)
        if v > best_v:
            best_v, best_h = v, h
    
    # Final sanity check
    final = np.maximum(best_h, 0.0)
    final = np.where(np.isfinite(final), final, 0.0)
    if not np.all(np.isfinite(final)) or np.any(final < 0):
        final = np.ones(N)
    
    return final.tolist()
