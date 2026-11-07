import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 138.0  # safety margin under 150s kill

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
    n1 = np.sum(np.abs(g)) / (m + 1)
    ninf = np.max(np.abs(g))
    if n1 * ninf < 1e-30:
        return -np.inf
    return l2 / (n1 * ninf)

def _neg_eval(h):
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    return -_eval(h)

def _cd(h0, budget):
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    best = _eval(h)
    step = 0.2
    while time.time() < budget:
        improved = False
        # Use a larger step initially, then shrink
        for i in range(N):
            for d in (step, -step):
                cand = h.copy()
                cand[i] = max(0.0, cand[i] + d)
                v = _eval(cand)
                if v > best + 1e-10:
                    h = cand
                    best = v
                    improved = True
        if not improved:
            step *= 0.7
            if step < 1e-6:
                break
    return h, best

def _nm(h0, budget):
    try:
        # Use L-BFGS-B with bounds
        bounds = [(0, 10)] * N
        res = minimize(_neg_eval, h0, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _eval(h)
            return h, v
    except Exception:
        pass
    try:
        res = minimize(_neg_eval, h0, method="Nelder-Mead",
                       options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-9})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _eval(h)
            return h, v
    except Exception:
        pass
    return h0, _eval(h0)

def _gaussian_shapes(t, sigmas):
    shapes = []
    for s in sigmas:
        sh = np.exp(-t**2 / (2 * s**2))
        shapes.append(sh)
    return shapes

def _cosine_shapes(t, widths):
    shapes = []
    for w in widths:
        sh = np.maximum(np.cos(np.pi * t / w), 0.0)
        shapes.append(sh)
    return shapes

def _tapered_shapes(t, alphas):
    shapes = []
    for a in alphas:
        sh = np.maximum(1.0 - a * np.abs(t) * 2.0, 0.0)
        shapes.append(sh)
        sh2 = np.maximum(1.0 - a * t**2, 0.0)
        shapes.append(sh2)
    return shapes

def construct():
    t = np.linspace(-0.5, 0.5, N)
    
    # Start with a good initial guess
    best_h = np.ones(N)
    best_v = _eval(best_h)
    
    # 1) Analytic shape exploration
    all_shapes = []
    
    # Gaussian shapes
    sigmas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]
    all_shapes.extend(_gaussian_shapes(t, sigmas))
    
    # Cosine shapes
    widths = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    all_shapes.extend(_cosine_shapes(t, widths))
    
    # Tapered shapes
    alphas = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
    all_shapes.extend(_tapered_shapes(t, alphas))
    
    # Additional parametric families
    # Power-law tapers
    for p in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        sh = np.maximum(1.0 - (2.0 * np.abs(t))**p, 0.0)
        all_shapes.append(sh)
    
    # Exponential tapers
    for a in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        sh = np.exp(-a * np.abs(t))
        all_shapes.append(sh)
        sh2 = np.exp(-a * t**2)
        all_shapes.append(sh2)
    
    # Smooth bumps with different widths
    for c in [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
        sh = 1.0 / (1.0 + c * t**2)
        all_shapes.append(sh)
    
    # Sine-based shapes
    for a in [0.5, 1.0, 1.5, 2.0]:
        sh = np.maximum(np.sin(np.pi * t) ** a, 0.0)
        all_shapes.append(sh)
        sh2 = np.maximum(np.cos(np.pi * t * 0.5) ** a, 0.0)
        all_shapes.append(sh2)
    
    # Evaluate all shapes
    for sh in all_shapes:
        sh = np.asarray(sh, dtype=float)
        if not np.all(np.isfinite(sh)):
            continue
        sh = np.maximum(sh, 0.0)
        if np.max(sh) <= 1e-12:
            continue
        # Normalize to have max = 1 for fair comparison, but also try raw
        sh_norm = sh / np.max(sh)
        for candidate in [sh, sh_norm]:
            v = _eval(candidate)
            if v > best_v:
                best_v = v
                best_h = candidate.copy()
    
    # 2) Coordinate descent refinement
    budget = min(time.time() + 40.0, _DEADLINE - 70.0)
    h, v = _cd(best_h, budget)
    if v > best_v:
        best_v, best_h = v, h
    
    # 3) L-BFGS-B optimization
    budget2 = min(time.time() + 45.0, _DEADLINE - 30.0)
    h2, v2 = _nm(best_h, budget2)
    if v2 > best_v:
        best_v, best_h = v2, h2
    
    # 4) Another coordinate descent pass
    budget3 = min(time.time() + 20.0, _DEADLINE - 10.0)
    h3, v3 = _cd(best_h, budget3)
    if v3 > best_v:
        best_v, best_h = v3, h3
    
    # 5) Final Nelder-Mead polish
    budget4 = min(time.time() + 15.0, _DEADLINE - 5.0)
    h4, v4 = _nm(best_h, budget4)
    if v4 > best_v:
        best_v, best_h = v4, h4
    
    # Deadline guard: return best found
    final = np.maximum(best_h, 0.0)
    final = np.where(np.isfinite(final), final, 0.0)
    
    # Final sanity check
    if not np.all(np.isfinite(final)) or np.any(final < 0):
        final = np.ones(N)
    
    return final.tolist()
