# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Weighted Sum decomposition with Nelder-Mead refinement for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2
            for i in range(n):
                f = 1.0
                r = 0.0
                j = i + 1
                while j > 0:
                    f /= base
                    r += f * (j % base)
                    j //= base
                points[i, d] = r
        return points

    def _is_pareto(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(np.array(f, dtype=np.float64))
            return x, np.array(f, dtype=np.float64)

        # --- Phase 1: Initial Space-Filling ---
        # Use a smaller Halton set to save budget for local searches
        n_fill = min(20, budget)
        halton_pts = self._halton(n_fill, dim)
        for i in range(n_fill):
            if used >= budget or time.time() > deadline:
                break
            x = lo + halton_pts[i] * (hi - lo)
            evaluate(x)

        if used >= budget or time.time() > deadline:
            return np.array(all_X) if all_X else None

        F = np.array(all_F)
        X = np.array(all_X)

        # --- Phase 2: Weighted Sum Decomposition ---
        # Generate weight vectors covering the simplex
        # For 3 objectives, we can use a grid or random points on the simplex
        # Let's use 12 weight vectors: 3 vertices, 3 edge midpoints, 6 random
        weights = []
        # Vertices
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        # Edge midpoints
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                w = np.zeros(n_obj)
                w[i] = 0.5
                w[j] = 0.5
                weights.append(w)
        # Random weights
        for _ in range(6):
            w = np.random.rand(n_obj)
            w /= w.sum()
            weights.append(w)

        weights = np.array(weights)
        n_weights = len(weights)
        
        # Calculate weighted sums for initial points to find best starts
        # F shape: (n_fill, n_obj), weights shape: (n_weights, n_obj)
        # We want for each weight w, the point x that minimizes w @ f(x)
        
        evals_left = budget - used
        # Estimate cost per NM run: dim*(dim+1) initial + ~10*dim iterations
        # Let's aim for ~15 evals per run
        cost_per_nm = 15
        n_runs = min(n_weights, max(1, evals_left // cost_per_nm))
        
        # Select top n_weights weights to run, or just run all if budget allows
        # We will run the first n_runs weights
        
        for k in range(n_runs):
            if used >= budget or time.time() > deadline - 5:
                break

            w = weights[k]
            
            # Find best initial point from X based on this weight
            weighted_sums = F @ w
            best_idx = np.argmin(weighted_sums)
            x0 = X[best_idx].copy()

            # Define objective function for this weight
            def obj_weighted(x):
                if used >= budget or time.time() > deadline - 2:
                    return 1e10
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                used += 1
                all_X.append(x_clipped.copy())
                all_F.append(np.array(f, dtype=np.float64))
                return np.dot(w, f)

            # Run Nelder-Mead
            try:
                result = minimize(
                    obj_weighted, x0, method='Nelder-Mead',
                    options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-4,
                             'adaptive': True}
                )
                # Evaluate the result explicitly to ensure it's recorded
                # (minimize might not call it if it converges immediately or uses cache)
                if used < budget and time.time() < deadline - 2:
                    x_best = np.clip(result.x, lo, hi)
                    f_best = problem(x_best)
                    used += 1
                    all_X.append(x_best.copy())
                    all_F.append(np.array(f_best, dtype=np.float64))
            except Exception:
                pass

        # --- Phase 3: Final Polish ---
        if used < budget and time.time() < deadline - 5:
            # Get current best non-dominated points
            F_all = np.array(all_F)
            X_all = np.array(all_X)
            pareto_mask = self._is_pareto(F_all)
            pareto_indices = np.where(pareto_mask)[0]
            
            if len(pareto_indices) > 0:
                # Refine the top 3 non-dominated points with a quick NM on sum
                sums = F_all[pareto_indices].sum(axis=1)
                top_idx = pareto_indices[np.argsort(sums)[:3]]
                
                for tidx in top_idx:
                    if used >= budget or time.time() > deadline - 3:
                        break
                    x0 = X_all[tidx].copy()
                    
                    def obj_sum(x):
                        if used >= budget or time.time() > deadline - 1:
                            return 1e10
                        x_clipped = np.clip(x, lo, hi)
                        f = problem(x_clipped)
                        used += 1
                        all_X.append(x_clipped.copy())
                        all_F.append(np.array(f, dtype=np.float64))
                        return np.sum(f)

                    try:
                        result = minimize(
                            obj_sum, x0, method='Nelder-Mead',
                            options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4,
                                     'adaptive': True}
                        )
                        if used < budget and time.time() < deadline - 1:
                            x_best = np.clip(result.x, lo, hi)
                            f_best = problem(x_best)
                            used += 1
                            all_X.append(x_best.copy())
                            all_F.append(np.array(f_best, dtype=np.float64))
                    except Exception:
                        pass

        return np.array(all_X) if all_X else None

# EVOLVE-BLOCK-END
