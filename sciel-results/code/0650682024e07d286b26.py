# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """LHS + Nelder-Mead multi-objective search for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 300.0 - 10.0  # Safety margin
        budget = int(self.budget)
        dim = int(self.dim)
        n_obj = int(self.n_obj)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        if np.any(span <= 0):
            span = np.where(span <= 0, 1.0, span)

        # LHS initialization for guaranteed diverse coverage
        n_lhs = min(24, budget // 2)
        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.default_rng(d * 100 + 42).permutation(n_lhs)
            lhs_X[:, d] = (perm + 0.5) / n_lhs
        lhs_X = lo + span * lhs_X

        # Evaluate LHS points
        lhs_F = np.array([problem(x) for x in lhs_X], dtype=float)
        used = n_lhs

        # Find non-dominated points from LHS
        def is_non_dominated(f, F):
            for j in range(len(F)):
                if j != int(np.where(np.all(F == f, axis=1))[0][0]) if np.any(np.all(F == f, axis=1)) else -1:
                    pass
            # Simpler approach
            for j in range(len(F)):
                if np.all(F[j] <= f) and np.any(F[j] < f):
                    return False
            return True

        non_dom_idx = [i for i in range(n_lhs) if is_non_dominated(lhs_F[i], lhs_F)]
        
        # Sort non-dominated by sum of objectives (proxy for quality)
        non_dom_idx.sort(key=lambda i: np.sum(lhs_F[i]))
        
        # Use remaining budget for Nelder-Mead refinement of best non-dominated points
        remaining = budget - used
        if remaining > 0 and non_dom_idx:
            # Allocate evaluations: use ~10-20 per local refinement
            evals_per_refine = max(5, remaining // len(non_dom_idx))
            evals_per_refine = min(evals_per_refine, 20)
            
            refined_X = []
            refined_F = []
            
            for idx in non_dom_idx:
                if used >= budget:
                    break
                if time.time() - t0 > deadline:
                    break
                
                x0 = lhs_X[idx]
                f0 = lhs_F[idx]
                
                # Multi-objective via weighted sum with multiple weight vectors
                # Try a few different weight combinations
                weights_list = [
                    [1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 2.0],
                    [2.0, 2.0, 1.0],
                    [1.0, 2.0, 2.0],
                ]
                
                for w in weights_list:
                    if used >= budget:
                        break
                    if time.time() - t0 > deadline:
                        break
                    
                    w_norm = np.array(w, dtype=float)
                    w_norm = w_norm / np.sum(w_norm)
                    
                    def scalarized(x):
                        fx = problem(x)
                        return np.dot(fx, w_norm)
                    
                    try:
                        res = minimize(scalarized, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': evals_per_refine, 'xatol': 1e-6, 'fatol': 1e-8})
                        if res.x is not None:
                            fx_res = problem(res.x)
                            # Check if this is better than current best in any way
                            refined_X.append(res.x)
                            refined_F.append(fx_res)
                            used += 1
                    except Exception:
                        pass
            
            if refined_X:
                refined_X = np.array(refined_X)
                refined_F = np.array(refined_F)
                # Combine all results
                all_X = np.vstack([lhs_X, refined_X])
                all_F = np.vstack([lhs_F, refined_F])
            else:
                all_F = lhs_F
        else:
            all_F = lhs_F

        return all_F
# EVOLVE-BLOCK-END
