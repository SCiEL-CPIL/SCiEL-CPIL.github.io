"""Multi-objective optimizer on RE21 using Multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        
        n_starts = 10
        alloc_per_start = self.budget // n_starts
        
        # Generate weight vectors
        weights = []
        for _ in range(n_starts):
            w = rng.random(self.n_obj)
            w /= w.sum()
            weights.append(w)
        
        all_points = []
        all_objs = []
        
        def run_nm(x0, w, budget_limit):
            # Local Nelder-Mead
            delta = 0.1 * span
            simplex = [x0.copy()]
            for i in range(self.dim):
                v = x0.copy()
                v[i] += delta[i]
                simplex.append(v)
            
            f_simplex = []
            evals_used = 0
            results = []
            
            # Init simplex
            for v in simplex:
                v_clip = np.clip(v, lo, hi)
                obj = problem(v_clip)
                f_simplex.append(np.dot(w, obj))
                results.append((v_clip, obj))
                evals_used += 1
                if evals_used >= budget_limit:
                    return results
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while evals_used < budget_limit:
                # Sort
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr_vec = problem(xr)
                fr = np.dot(w, fr_vec)
                evals_used += 1
                results.append((xr, fr_vec))
                
                if fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    fe_vec = problem(xe)
                    fe = np.dot(w, fe_vec)
                    evals_used += 1
                    results.append((xe, fe_vec))
                    
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    if fr < f_simplex[-1]:
                        # Outside
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, lo, hi)
                        fc_vec = problem(xc)
                        fc = np.dot(w, fc_vec)
                        evals_used += 1
                        results.append((xc, fc_vec))
                        
                        if fc <= fr:
                            simplex[-1] = xc
                            f_simplex[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if evals_used >= budget_limit:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_simplex[i] = np.dot(w, problem(simplex[i]))
                                evals_used += 1
                                results.append((simplex[i], f_simplex[i]))
                    else:
                        # Inside
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, lo, hi)
                        fc_vec = problem(xc)
                        fc = np.dot(w, fc_vec)
                        evals_used += 1
                        results.append((xc, fc_vec))
                        
                        if fc <= f_simplex[-1]:
                            simplex[-1] = xc
                            f_simplex[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if evals_used >= budget_limit:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_simplex[i] = np.dot(w, problem(simplex[i]))
                                evals_used += 1
                                results.append((simplex[i], f_simplex[i]))
                
                if evals_used >= budget_limit:
                    break
                    
            return results

        remaining = self.budget
        for i in range(n_starts):
            if remaining <= 0:
                break
            
            x0 = lo + 0.5 * span + 0.1 * span * (rng.random(self.dim) - 0.5)
            x0 = np.clip(x0, lo, hi)
            
            w = weights[i]
            alloc = min(remaining, alloc_per_start)
            
            res = run_nm(x0, w, alloc)
            for x, f in res:
                all_points.append(x)
                all_objs.append(f)
            
            remaining -= len(res)
            
        return all_points[0] if all_points else None

# EVOLVE-BLOCK-END
