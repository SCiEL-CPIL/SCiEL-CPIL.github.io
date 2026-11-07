import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 290 # 10s margin
        
        evals_used = 0
        max_evals = self.budget
        
        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        
        # Ideal points (running min)
        z_star = np.full(self.n_obj, np.inf)
        
        def update_archive(x, f):
            nonlocal evals_used, z_star
            # Update ideal points
            for i in range(self.n_obj):
                if f[i] < z_star[i]:
                    z_star[i] = f[i]
            
            # Check dominance
            dominated = False
            for j in range(len(archive_F)):
                if np.all(archive_F[j] <= f) and np.any(archive_F[j] < f):
                    dominated = True
                    break
            
            if dominated:
                return False
            
            # Remove dominated points from archive
            new_archive_X = []
            new_archive_F = []
            for j in range(len(archive_F)):
                if not (np.all(f <= archive_F[j]) and np.any(f < archive_F[j])):
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(archive_F[j])
            
            archive_X = new_archive_X
            archive_F = new_archive_F
            
            archive_X.append(x)
            archive_F.append(f)
            return True

        def evaluate(x):
            nonlocal evals_used
            if evals_used >= max_evals:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            evals_used += 1
            return x_clipped, f

        # 1. Initialization: Random sampling to estimate z_star and get starting points
        n_init = min(20, max_evals // 5)
        rng = np.random.default_rng(0)
        
        init_X = []
        init_F = []
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(self.dim)
            res = evaluate(x)
            if res is None: break
            x, f = res
            init_X.append(x)
            init_F.append(f)
            update_archive(x, f)
            
        if evals_used >= max_evals:
            return archive_X

        # 2. Generate Tchebycheff weights
        # Simple grid on simplex for 3 objectives
        # Weights (a, b, c) such that a+b+c=1, a,b,c >= 0
        weights = []
        step = 0.25 # 5 levels per axis approx
        for a in np.arange(0, 1 + step, step):
            for b in np.arange(0, 1 - a + step, step):
                c = 1 - a - b
                if c >= -1e-9:
                    weights.append([a, b, max(0, c)])
        
        # Normalize weights to ensure they sum to 1 (they should, but floating point)
        weights = np.array(weights)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Select a subset of weights if too many, or just use all if budget allows
        # With step 0.25, we have roughly 15-20 weights.
        # Let's limit to 10 weights to save budget for local search.
        if len(weights) > 10:
            # Randomly select 10
            idx = rng.choice(len(weights), 10, replace=False)
            weights = weights[idx]
            
        # 3. Local Search with Nelder-Mead on Tchebycheff function
        # Tchebycheff: max( lambda_i * (f_i(x) - z_star_i) )
        # Note: z_star changes as we find better points. 
        # This can cause instability in Nelder-Mead. 
        # Alternative: Use weighted sum if front is convex, or fixed z_star from init.
        # Let's use the z_star from init for stability, or update it only between weights.
        
        z_star_fixed = z_star.copy()
        
        for w in weights:
            if time.time() > deadline:
                break
            if evals_used >= max_evals:
                break
                
            # Define objective function for Nelder-Mead
            def tcheb_obj(x):
                res = evaluate(x)
                if res is None:
                    return 1e9 # Penalty
                x, f = res
                # Use current z_star for dynamic adaptation? 
                # It's safer to use the z_star at the start of this local search
                # to avoid the target moving.
                diff = f - z_star_fixed
                # If any objective is worse than ideal, it contributes positively.
                # Tchebycheff distance
                return np.max(w * diff)
            
            # Initialize simplex at a random point or best point
            # Using a random point helps explore different parts
            x0 = lo + (hi - lo) * rng.random(self.dim)
            
            # Nelder-Mead settings
            # maxiter is the number of iterations, not evaluations.
            # For dim=4, ~5 evals/iter.
            # Budget left for this weight: roughly (max_evals - evals_used) / remaining_weights
            # Let's just cap it to ~15 evals per weight.
            # maxiter=3 -> ~15 evals
            options = {
                'maxiter': 5, 
                'xatol': 1e-6,
                'fatol': 1e-6
            }
            
            try:
                res = minimize(tcheb_obj, x0, method='Nelder-Mead', options=options, bounds=bounds)
                if res.success or res.status == 0:
                    # Evaluate the best point found by Nelder-Mead to ensure it's in archive
                    # It might have been evaluated during the process, but let's be safe.
                    # Actually, minimize calls the function, so it's already counted in evals_used.
                    # We just need to update the archive with the result.
                    x_best = np.clip(res.x, lo, hi)
                    f_best = np.array(problem(x_best))
                    # This evaluation might be redundant if it was the last one in minimize.
                    # To save budget, we can check if we already have this point.
                    # But problem() is cheap? No, it's the expensive part.
                    # Let's assume the last eval in minimize was the best.
                    # However, minimize doesn't return the function value of the best point directly in a way that guarantees it was the last call.
                    # Safer: Just call evaluate(x_best) and update archive.
                    # If we exceed budget, evaluate returns None.
                    res_eval = evaluate(x_best)
                    if res_eval:
                        x, f = res_eval
                        update_archive(x, f)
                        
            except Exception:
                pass
            
            # Update z_star_fixed for the next weight?
            # Yes, using the improved z_star helps guide the next search.
            z_star_fixed = z_star.copy()

        # 4. Final Polish
        # If we have budget left, refine the top non-dominated points.
        if evals_used < max_evals and archive_F:
            # Sort archive by hypervolume contribution or just pick a few
            # Let's pick up to 3 points and run a short Nelder-Mead on a weighted sum of their neighbors?
            # Simpler: Just run a few more random local searches around the best points.
            
            # Identify the "best" points (lowest sum of normalized objectives)
            # Normalize objectives by range if possible, or just sum.
            # Let's just pick the first 2 points in the archive (arbitrary but they are non-dominated)
            # Or better: pick points that are "extreme".
            
            # For simplicity and safety, just do a few random local refinements.
            n_polish = min(5, (max_evals - evals_used) // 5)
            for _ in range(n_polish):
                if time.time() > deadline:
                    break
                if evals_used >= max_evals:
                    break
                
                # Pick a random point from archive
                idx = rng.integers(len(archive_X))
                x0 = archive_X[idx]
                
                # Small perturbation
                x0 = x0 + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                x0 = np.clip(x0, lo, hi)
                
                # Minimize sum of objectives (heuristic for local improvement)
                def sum_obj(x):
                    res = evaluate(x)
                    if res is None:
                        return 1e9
                    x, f = res
                    return np.sum(f)
                
                try:
                    res = minimize(sum_obj, x0, method='Nelder-Mead', options={'maxiter': 3})
                    if res.success:
                        x_best = np.clip(res.x, lo, hi)
                        res_eval = evaluate(x_best)
                        if res_eval:
                            x, f = res_eval
                            update_archive(x, f)
                except Exception:
                    pass

        return archive_X
