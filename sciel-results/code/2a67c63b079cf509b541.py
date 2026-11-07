"""Program for Erdos minimum overlap task using multi-start optimization with diverse seeds.

Strategy:
1. Generate diverse initial seeds (anti-periodic, periodic, random, structured)
2. Use smooth surrogate (log-sum-exp) to optimize each seed
3. Local refinement with coordinate descent for fine-tuning
4. Return the best solution found within time budget
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 145.0
START = time.time()


def remaining():
    return DEADLINE - (time.time() - START)


def value(h):
    """The exact evaluator value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def value_smooth(h, beta):
    """Smooth surrogate for the max via log-sum-exp (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    cm = float(c.max())
    z = beta * (c - cm)
    z = np.clip(z, -500, 500)
    logsum = np.log(np.sum(np.exp(z))) + beta * cm
    return float(logsum) / N * 2.0


def make_seed_anti_periodic():
    """Anti-periodic seed: h[i+50] = 1 - h[i]"""
    i = np.arange(HALF)
    d = np.abs(i - (HALF - 1) / 2.0) / ((HALF - 1) / 2.0)
    tri = 1.0 - d
    first = 0.5 + (0.5 - tri.mean()) * tri
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def make_seed_periodic():
    """Periodic seed with high-frequency oscillation"""
    i = np.arange(N)
    # Mix of frequencies to spread correlation
    h = 0.5 + 0.3 * np.sin(2 * np.pi * i / 10) + 0.2 * np.sin(2 * np.pi * i / 7)
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def make_seed_random(seed_val):
    """Random seed with fixed seed for reproducibility"""
    rng = np.random.RandomState(seed_val)
    h = rng.rand(N)
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def make_seed_sawtooth():
    """Sawtooth wave pattern"""
    i = np.arange(N)
    h = 0.5 + 0.5 * (2 * (i % 20) / 20 - 1)
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def make_seed_bimodal():
    """Two peaks pattern"""
    i = np.arange(N)
    h = 0.5 + 0.4 * np.exp(-((i - 25) / 10)**2) + 0.4 * np.exp(-((i - 75) / 10)**2)
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def refine_smooth_full(h, beta, maxiter):
    """Optimize all N parameters to lower the smooth objective."""
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    bounds = [(1e-12, 1.0 - 1e-12)] * N

    def obj(p):
        return value_smooth(p, beta)

    try:
        res = minimize(obj, h0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-15,
                                "gtol": 1e-13, "maxls": 40})
        return np.clip(res.x, 0.0, 1.0)
    except Exception:
        return h0


def refine_exact_full(h, maxiter):
    """Optimize all N parameters to lower the exact objective."""
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    bounds = [(1e-12, 1.0 - 1e-12)] * N

    def obj(p):
        return value(p)

    try:
        res = minimize(obj, h0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-16,
                                "gtol": 1e-14, "maxls": 50})
        return np.clip(res.x, 0.0, 1.0)
    except Exception:
        return h0


def coordinate_descent(h, max_iters=2000):
    """Local refinement via coordinate descent"""
    h = np.array(h, dtype=np.float64)
    current_val = value(h)
    
    for it in range(max_iters):
        if remaining() < 5.0:
            break
        improved = False
        for i in range(N):
            # Try small perturbations
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h_new = h.copy()
                h_new[i] = np.clip(h[i] + delta, 0.0, 1.0)
                # Fix sum constraint
                h_new += (HALF - h_new.sum()) / N
                h_new = np.clip(h_new, 0.0, 1.0)
                v = value(h_new)
                if v < current_val:
                    h = h_new
                    current_val = v
                    improved = True
                    break
        if not improved:
            break
    
    # Final sum fix
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    t0 = time.time()
    
    # Generate diverse seeds
    seeds = [
        make_seed_anti_periodic(),
        make_seed_periodic(),
        make_seed_random(42),
        make_seed_random(123),
        make_seed_sawtooth(),
        make_seed_bimodal(),
    ]
    
    best = None
    best_val = float('inf')
    
    # Phase 1: Smooth optimization on each seed with moderate beta
    for seed_h in seeds:
        if remaining() < 30.0:
            break
        h = refine_smooth_full(seed_h, 50.0, 200)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v
    
    # Phase 2: Higher beta refinement on best
    if best is not None and remaining() > 20.0:
        h = refine_smooth_full(best, 200.0, 300)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v
    
    # Phase 3: Exact objective polish
    if best is not None and remaining() > 15.0:
        h = refine_exact_full(best, 400)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v
    
    # Phase 4: Coordinate descent local refinement
    if best is not None and remaining() > 10.0:
        h = coordinate_descent(best, max_iters=3000)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v
    
    # Phase 5: Final exact polish
    if best is not None and remaining() > 5.0:
        h = refine_exact_full(best, 200)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v
    
    # Ensure sum constraint
    if best is None:
        best = make_seed_anti_periodic()
    
    total = best.sum()
    if abs(total - HALF) > 1e-9 * N:
        best += (HALF - total) / N
        best = np.clip(best, 0.0, 1.0)
    
    return best
