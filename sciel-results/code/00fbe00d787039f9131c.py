import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_genlaguerre, roots_laguerre
import time

def construct():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit

    m = 12
    # Precompute factorials for the objective function
    fact = np.ones(4 * m + 1)
    for i in range(1, 4 * m + 1):
        fact[i] = fact[i - 1] * i

    def objective(params):
        if not np.all(params > 0):
            return 1e10
        roots = np.sort(params)
        q_coeffs = np.poly(roots)
        q2 = np.convolve(q_coeffs, q_coeffs)
        q4 = np.convolve(q2, q2)
        powers = np.arange(4 * m, -1, -1)
        return np.sum(q4 * fact[powers])

    # Generate diverse initial candidates
    def get_candidates():
        candidates = []
        
        # 1. Seed: arithmetic progression 4, 8, ..., 48
        candidates.append(np.array([4.0 * (i + 1) for i in range(m)]))
        
        # 2. Standard Laguerre roots L_12^(0)
        try:
            r0, _ = roots_laguerre(m)
            candidates.append(r0.copy())
        except:
            pass
        
        # 3. Generalized Laguerre roots for various alpha
        for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            try:
                ra, _ = roots_genlaguerre(m, alpha)
                if np.all(ra > 0):
                    candidates.append(ra.copy())
            except:
                pass
        
        # 4. Perturbed arithmetic with different scales
        for scale in [3.0, 5.0, 6.0]:
            candidates.append(scale * np.arange(1, m + 1))
        
        # 5. Random perturbations of seed
        rng = np.random.default_rng(42)
        for _ in range(5):
            pert = rng.normal(0, 1.0, size=m)
            cand = 4.0 * np.arange(1, m + 1) + pert
            if np.all(cand > 0):
                candidates.append(cand)
        
        return candidates

    best_x = None
    best_val = np.inf

    # Phase 1: Evaluate all candidates and pick the best
    for cand in get_candidates():
        try:
            val = objective(cand)
            if val < best_val:
                best_val = val
                best_x = cand.copy()
        except:
            pass

    if best_x is None:
        # Fallback to seed
        best_x = np.array([4.0 * (i + 1) for i in range(m)])
        best_val = objective(best_x)

    # Phase 2: Local refinement from the best candidate
    if time.time() < deadline - 30:
        try:
            res_nm = minimize(objective, best_x, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res_nm.fun < best_val:
                best_x = res_nm.x
                best_val = res_nm.fun
        except:
            pass

    # Phase 3: Try L-BFGS-B from multiple starting points
    if time.time() < deadline - 20:
        bounds = [(1e-6, None)] * m
        starts = [best_x.copy()]
        # Add a few more diverse starts
        for cand in get_candidates()[:3]:
            starts.append(cand.copy())

        for x0 in starts:
            if time.time() > deadline - 10:
                break
            try:
                res_lbfgs = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': 1000, 'ftol': 1e-12})
                if res_lbfgs.fun < best_val:
                    best_x = res_lbfgs.x
                    best_val = res_lbfgs.fun
            except:
                pass

    # Phase 4: Final polish with Powell if time permits
    if time.time() < deadline - 10:
        try:
            res_powell = minimize(objective, best_x, method='Powell',
                                 options={'maxiter': 300, 'ftol': 1e-12, 'xtol': 1e-12})
            if res_powell.fun < best_val:
                best_x = res_powell.x
                best_val = res_powell.fun
        except:
            pass

    # Ensure positive and sorted
    best_x = np.maximum(best_x, 1e-6)
    best_x = np.sort(best_x)

    return [float(x) for x in best_x]
