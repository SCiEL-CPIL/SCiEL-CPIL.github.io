"""Multi-objective optimizer for RE21: Epsilon-constraint decomposition via L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Epsilon-constraint method for tracing the Pareto front of RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin under the 300 s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        X = []
        F = []

        def eval_pt(x):
            nonlocal used
            used += 1
            return np.asarray(problem(x), dtype=float)

        # Phase 1: Initial sampling to estimate objective ranges
        n_init = min(20, budget // 4)
        rng = np.random.default_rng(0)
        
        # Latin Hypercube for initial diversity
        def lhs_sample(n, d):
            points = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                points[:, j] = (perm + rng.random(n)) / n
            return points

        X_init = lo + (hi - lo) * lhs_sample(n_init, dim)
        for x in X_init:
            X.append(x.copy())
            F.append(eval_pt(x))

        if used >= budget:
            return np.array(X)

        # Estimate bounds for objectives
        Fv = np.array(F)
        f1_min, f1_max = Fv[:, 0].min(), Fv[:, 0].max()
        f2_min, f2_max = Fv[:, 1].min(), Fv[:, 1].max()
        
        # Expand ranges slightly to ensure we cover the front
        f1_range = max(f1_max - f1_min, 1e-6)
        f2_range = max(f2_max - f2_min, 1e-6)
        
        # Phase 2: Epsilon-constraint method
        # We optimize f1 subject to f2 <= epsilon.
        # We sweep epsilon from high (f2_max) to low (f2_min).
        
        # Determine number of epsilon constraints based on remaining budget
        # Each L-BFGS-B call can take several evaluations. 
        # Let's aim for roughly 15-25 distinct epsilon levels.
        # We have ~180 evaluations left. If each takes ~5-10 evals, we can do ~20-30 runs.
        # Let's be conservative to stay within time limits too.
        n_eps = 20
        
        # Create epsilon levels
        # Linear spacing in f2
        epsilons = np.linspace(f2_max + 0.01 * f2_range, f2_min - 0.01 * f2_range, n_eps)
        
        # Sort epsilons descending (start with loose constraint on f2, i.e., high f2 allowed, optimize f1)
        # Actually, standard epsilon constraint: min f1 s.t. f2 <= eps.
        # If eps is large, constraint is loose, we find min f1.
        # If eps is small, constraint is tight, we find points with low f2.
        # So we sweep eps from high to low to trace the front from f1-min to f2-min.
        
        for eps in epsilons:
            if used >= budget or time.time() > deadline:
                break
            
            def objective(x, eps=eps):
                f = np.asarray(problem(x), dtype=float)
                # Return f1 + penalty if f2 > eps
                # Using a quadratic penalty for smoothness
                penalty = 0.0
                if f[1] > eps:
                    penalty = 100.0 * (f[1] - eps) ** 2
                return f[0] + penalty
            
            # Choose initial point
            # Pick the point in current archive that best satisfies the constraint
            # or has lowest f1 if constraint is loose
            Fv = np.array(F)
            Xv = np.array(X)
            
            # Filter points that satisfy f2 <= eps
            valid = np.where(Fv[:, 1] <= eps)[0]
            
            if len(valid) > 0:
                # Among valid, pick one with lowest f1
                best_idx = valid[np.argmin(Fv[valid, 0])]
                x0 = Xv[best_idx].copy()
            else:
                # If no point satisfies, pick the one with lowest f2
                best_idx = np.argmin(Fv[:, 1])
                x0 = Xv[best_idx].copy()
                # Maybe add some noise to escape local minima if stuck
                x0 = x0 + 0.05 * (hi - lo) * rng.standard_normal(dim)
                x0 = np.clip(x0, lo, hi)

            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': 30, 'gtol': 1e-4})
                
                if used < budget:
                    xf = np.clip(res.x, lo, hi)
                    ff = eval_pt(xf)
                    
                    # Check if this point is non-dominated and worth adding
                    # For simplicity, we just add it. The final HV calculation will handle it.
                    # But we should avoid duplicates.
                    is_dup = False
                    for f_existing in F:
                        if np.allclose(f_existing, ff, rtol=1e-5, atol=1e-5):
                            is_dup = True
                            break
                    
                    if not is_dup:
                        X.append(xf.copy())
                        F.append(ff)
                        
            except Exception:
                continue

        # Phase 3: Final polish if budget remains
        if time.time() < deadline - 5.0 and used < budget:
            # Try to improve the extremes
            # Min f1
            def obj_f1(x):
                return np.asarray(problem(x), dtype=float)[0]
            
            # Min f2
            def obj_f2(x):
                return np.asarray(problem(x), dtype=float)[1]
            
            # Start from best known
            Fv = np.array(F)
            Xv = np.array(X)
            
            # Min f1
            idx_f1 = np.argmin(Fv[:, 0])
            x0 = Xv[idx_f1].copy()
            try:
                res = minimize(obj_f1, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), options={'maxiter': 20})
                if used < budget and res.success:
                    xf = np.clip(res.x, lo, hi)
                    ff = eval_pt(xf)
                    is_dup = any(np.allclose(f_existing, ff, rtol=1e-5, atol=1e-5) for f_existing in F)
                    if not is_dup:
                        X.append(xf.copy())
                        F.append(ff)
            except:
                pass
            
            # Min f2
            idx_f2 = np.argmin(Fv[:, 1])
            x0 = Xv[idx_f2].copy()
            if used < budget:
                try:
                    res = minimize(obj_f2, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), options={'maxiter': 20})
                    if used < budget and res.success:
                        xf = np.clip(res.x, lo, hi)
                        ff = eval_pt(xf)
                        is_dup = any(np.allclose(f_existing, ff, rtol=1e-5, atol=1e-5) for f_existing in F)
                        if not is_dup:
                            X.append(xf.copy())
                            F.append(ff)
                except:
                    pass

        return np.array(X)
# EVOLVE-BLOCK-END
