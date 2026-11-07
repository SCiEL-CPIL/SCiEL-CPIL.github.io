"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _compute_value_fast(c0, c1, c2):
    """Compute r^2/(2*pi) using fast numpy polynomial operations."""
    c0f, c1f, c2f = float(c0), float(c1), float(c2)
    c3f = -(c0f + 12.0 * c1f + 16.0 * c2f) / 23040.0

    # q(y) where y = x^2, coefficients from y^5 down to y^0
    q_coeffs = np.array([
        4096.0 * c3f,
        -46080.0 * c3f,
        290304.0 * c3f + 256.0 * c2f,
        -774144.0 * c3f - 1344.0 * c2f,
        860160.0 * c3f + 2688.0 * c2f + 16.0 * c1f,
        -322560.0 * c3f - 1344.0 * c2f - 48.0 * c1f
    ])

    # Find roots
    roots_y = np.roots(q_coeffs)
    real_roots_y = [r.real for r in roots_y if abs(r.imag) < 1e-9 and r.real > 1e-10]

    if not real_roots_y:
        return np.inf

    # Check sign change at each real positive root
    valid_roots = []
    for r in real_roots_y:
        eps = 1e-8 * max(1.0, r)
        q_neg = np.polyval(q_coeffs, r - eps)
        q_pos = np.polyval(q_coeffs, r + eps)
        if q_neg * q_pos < 0:
            valid_roots.append(r)

    if not valid_roots:
        return np.inf

    r_max_y = max(valid_roots)
    return r_max_y / (2.0 * np.pi)


def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s kill

    # Seed
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_value_fast(seed[0], seed[1], seed[2])
    best_c = seed.copy()

    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60.0:
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        
        def de_objective(x):
            try:
                v = _compute_value_fast(x[0], x[1], x[2])
                return v if np.isfinite(v) else 1e6
            except:
                return 1e6

        de_result = differential_evolution(
            de_objective, bounds, maxiter=50, popsize=30,
            seed=42, tol=1e-10, mutation=(0.3, 1.0), recombination=0.7,
            polish=False, init='latinhypercube'
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_c = de_result.x.copy()

    # Phase 2: High-precision local optimization from best
    if time.time() < deadline - 30.0:
        def objective(x):
            try:
                return _compute_value_fast(x[0], x[1], x[2])
            except:
                return 1e10

        # Multiple local optimizations with different initial perturbations
        np.random.seed(123)
        for i in range(50):
            if time.time() > deadline - 20.0:
                break
            # Perturb best solution
            scale = max(0.01, 0.1 * np.exp(-i * 0.1))
            perturb = np.random.normal(0, scale, 3)
            x0 = best_c + perturb
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-18,
                                        'adaptive': True})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_c = res.x.copy()
            except:
                pass

    # Phase 3: Final polish with very tight tolerances
    if time.time() < deadline - 5.0:
        def objective_final(x):
            try:
                return _compute_value_fast(x[0], x[1], x[2])
            except:
                return 1e10

        for i in range(10):
            if time.time() > deadline - 2.0:
                break
            res = minimize(objective_final, best_c, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-18, 'fatol': 1e-20,
                                    'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_c = res.x.copy()

    return np.array(best_c)
# EVOLVE-BLOCK-END
