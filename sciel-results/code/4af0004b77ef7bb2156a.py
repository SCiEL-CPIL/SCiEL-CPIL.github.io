"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def _halton_point(dim, i):
    """
    Generate a deterministic point in [0, 1)^dim using the Halton sequence.
    Uses prime bases for each dimension.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    point = np.zeros(dim)
    for d in range(dim):
        base = primes[d % len(primes)]
        val = 0.0
        f = 1.0 / base
        n_val = i + 1  # 1-based index for sequence
        while n_val > 0:
            val += f * (n_val % base)
            n_val //= base
            f /= base
        point[d] = val
    return point

class Optimizer:
    """
    Ridge-Tracking Optimizer for BBOB f13 (Sharp Ridge).
    Phase 1: Broad coverage using Halton sequence to locate the ridge basin.
    Phase 2: Estimate ridge direction via PCA of top candidates, then 1-D minimization along the ridge axis.
    Phase 3: L-BFGS-B polish for final convergence.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        candidates = []

        # Phase 1: Global Exploration with Halton Sequence
        n_explore = 50
        n_explore = min(n_explore, budget - 1)
        
        for i in range(n_explore):
            if time.time() > deadline or calls_made >= budget:
                break
            
            p_unit = _halton_point(dim, i)
            p = p_unit * 10.0 - 5.0
            p = np.clip(p, -5.0, 5.0)
            
            try:
                f = func(p)
            except:
                f = np.inf
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = p.copy()
            
            candidates.append((f, p))
        
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except:
                best_f = np.inf
            calls_made += 1
            candidates.append((best_f, best_x))

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Ridge-axis tracking via PCA + 1-D minimization
        # Use top 10 candidates to estimate the ridge direction
        top_k = min(10, len(candidates))
        top_points = np.array([x for _, x in candidates[:top_k]])
        top_fits = np.array([f for f, _ in candidates[:top_k]])
        
        if top_k >= 2 and calls_made < budget and time.time() < deadline:
            # PCA to find principal axis
            centered = top_points - top_points.mean(axis=0)
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Principal axis is eigenvector with largest eigenvalue
            ridge_dir = eigvecs[:, -1]
            ridge_dir = ridge_dir / np.linalg.norm(ridge_dir)
            
            # Project best point onto ridge axis
            base_point = best_x.copy()
            # Find range along ridge axis by projecting all top points
            projections = np.array([np.dot(x - base_point, ridge_dir) for x in top_points])
            proj_min, proj_max = projections.min(), projections.max()
            # Expand range slightly
            proj_range = proj_max - proj_min
            if proj_range < 1e-6:
                proj_range = 1.0
            search_min = proj_min - proj_range * 0.5
            search_max = proj_max + proj_range * 0.5
            
            # 1-D minimization along ridge axis
            def line_func(t):
                x_new = base_point + t * ridge_dir
                x_new = np.clip(x_new, -5.0, 5.0)
                try:
                    return func(x_new)
                except:
                    return np.inf
            
            try:
                res_1d = minimize_scalar(
                    line_func,
                    bounds=(search_min, search_max),
                    method='bounded',
                    options={'xatol': 1e-6}
                )
                if res_1d.fun < best_f:
                    best_f = res_1d.fun
                    best_x = (base_point + res_1d.x * ridge_dir).copy()
                    best_x = np.clip(best_x, -5.0, 5.0)
            except Exception:
                pass
            
            # Update calls_made approximately (minimize_scalar uses ~10-20 evals)
            calls_made += 20

        # Phase 3: L-BFGS-B polish
        if best_x is not None and calls_made < budget and time.time() < deadline:
            def callback(xk):
                nonlocal calls_made
                calls_made += 1
                if calls_made >= budget or time.time() > deadline:
                    raise TimeoutError("Budget or time exceeded")
                return False

            try:
                res = minimize(
                    func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'gtol': 1e-8, 'disp': False},
                    callback=callback
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except TimeoutError:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
