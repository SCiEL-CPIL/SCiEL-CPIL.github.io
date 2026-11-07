"""Multi-objective optimizer for WFG4 using Decomposition-based Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Decomposition-based optimizer with local search for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _evaluate(self, problem, x):
        """Evaluate a single point."""
        return np.array(problem(x), dtype=np.float64)

    def _non_dominated_sort(self, F):
        """Simple non-dominated sort returning indices of the first front."""
        n = len(F)
        if n == 0:
            return []
        
        dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
        
        return [i for i in range(n) if not dominated[i]]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        used = 0
        
        # Archive of non-dominated solutions
        archive_X = []
        archive_F = []

        def is_dominated(f_new, archive_F):
            for f_old in archive_F:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def update_archive(x_new, f_new):
            nonlocal archive_X, archive_F
            # Check if new point is dominated by any existing
            if is_dominated(f_new, archive_F):
                return
            
            # Remove points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for x_old, f_old in zip(archive_X, archive_F):
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    continue # Dominated by new
                new_archive_X.append(x_old)
                new_archive_F.append(f_old)
            
            new_archive_X.append(x_new)
            new_archive_F.append(f_new)
            
            archive_X = new_archive_X
            archive_F = new_archive_F

        # Strategy: Decomposition with Local Search
        # We define a set of weight vectors to cover the objective space.
        # Since n_obj=3, we can generate weights on a simplex.
        n_weights = 10
        weights = []
        # Generate 10 random weight vectors that sum to 1
        for _ in range(n_weights):
            w = rng.random_sample(n_obj)
            w /= np.sum(w)
            weights.append(w)

        # For each weight, we perform a local search
        # We allocate budget evenly among weights
        evals_per_weight = max(5, budget // n_weights)

        for w in weights:
            if used >= budget or time.time() > deadline:
                break

            # Random initialization for this weight
            x0 = lo + (hi - lo) * rng.random_sample(dim)
            
            # Define the scalarized objective function
            # We use a weighted sum approach, but to handle disconnected fronts better,
            # we might use a penalty for deviation from ideal point if known, 
            # but weighted sum is standard for decomposition.
            # However, weighted sum cannot reach disconnected parts if the front is non-convex.
            # WFG4 is non-convex/disconnected.
            # Better: Use a random scalarization or Tchebycheff with random reference.
            # Let's stick to weighted sum but with multiple random restarts per weight?
            # No, budget is tight.
            
            # Actually, for WFG4, the objectives are such that minimizing weighted sum
            # often lands on one part of the front.
            # Let's try a different structural approach:
            # Multi-start Nelder-Mead on the original objectives using a "hypervolume-like" scalarization?
            # No, let's use **Random Weighted Sum** with **Nelder-Mead**.
            
            # To cover the disconnected front, we need to be lucky with weights or use a method that
            # doesn't rely on convexity.
            # Let's use **NSGA-II** but with **Local Search** (Memetic) on the first front?
            # Or just **Intensive Local Search** on random points with **Weighted Sum**.
            
            # Given the "Structural Change" directive, and that NSGA-II failed to reach 1.0,
            # and Inspiration 2 used "local po...", let's do **Local Search on Random Points**.
            
            # We will perform `evals_per_weight` evaluations of local search starting from x0.
            # The objective to minimize is the weighted sum: sum(w_i * f_i(x))
            
            def objective(x):
                nonlocal used
                if used >= budget or time.time() > deadline:
                    return np.inf
                x = np.clip(x, lo, hi)
                f = self._evaluate(problem, x)
                used += 1
                return np.dot(w, f)

            # Use Nelder-Mead for local search
            # maxiter is a proxy for evaluations, but Nelder-Mead uses ~dim*maxiter evals.
            # We want to use ~evals_per_weight evals.
            # dim=12. So maxiter ~ evals_per_weight / 12.
            maxiter = max(1, evals_per_weight // 12)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6})
                x_best = np.clip(res.x, lo, hi)
                f_best = self._evaluate(problem, x_best)
                used += 1 # This eval is not counted in objective? 
                # Wait, objective calls problem. So res.x evaluation is done inside?
                # No, minimize calls objective which calls problem.
                # So used is already incremented.
                # We should check if we exceeded budget.
                
                if used < budget:
                    update_archive(x_best, f_best)
                else:
                    # If we just hit budget, the last eval in objective was the last one.
                    # res.x is the best found.
                    # We need to make sure we didn't exceed.
                    pass
                    
            except Exception:
                pass

        # If we have budget left, do some random sampling and local polish
        while used < budget and time.time() < deadline:
            x_rand = lo + (hi - lo) * rng.random_sample(dim)
            f_rand = self._evaluate(problem, x_rand)
            used += 1
            update_archive(x_rand, f_rand)
            
            # Small local search on this point
            if used < budget - 5:
                w = rng.random_sample(n_obj)
                w /= np.sum(w)
                def obj2(x):
                    nonlocal used
                    if used >= budget or time.time() > deadline:
                        return np.inf
                    x = np.clip(x, lo, hi)
                    f = self._evaluate(problem, x)
                    used += 1
                    return np.dot(w, f)
                
                try:
                    res = minimize(obj2, x_rand, method='Nelder-Mead', options={'maxiter': 2})
                    x_best = np.clip(res.x, lo, hi)
                    f_best = self._evaluate(problem, x_best)
                    used += 1
                    update_archive(x_best, f_best)
                except Exception:
                    pass

        # Return non-dominated solutions from archive
        if not archive_X:
            # Fallback: return one random point if archive is empty (should not happen)
            x_ret = lo + (hi - lo) * rng.random_sample(dim)
            return np.array([x_ret])
            
        # Filter archive to non-dominated (update_archive does this, but let's be safe)
        # Convert to numpy arrays
        archive_X = np.array(archive_X)
        archive_F = np.array(archive_F)
        
        # Find non-dominated in archive
        non_dom_indices = self._non_dominated_sort(archive_F)
        
        if len(non_dom_indices) > 0:
            return archive_X[non_dom_indices]
        return archive_X[:1]
# EVOLVE-BLOCK-END
