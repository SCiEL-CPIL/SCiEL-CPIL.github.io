import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        n_obj = self.n_obj
        dim = self.dim
        
        # 1. Initialization: Space-filling Halton sequence for broad coverage
        # This is better than random initialization for hypervolume as it ensures diversity
        def halton(dim, n, skip=1):
            points = np.zeros((n, dim))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for d in range(dim):
                p = primes[d % len(primes)]
                for i in range(n):
                    x = 0.0
                    f = 1.0 / p
                    idx = i + skip
                    while idx > 0:
                        idx, rem = divmod(idx, p)
                        x += f * rem
                        f /= p
                    points[i, d] = x
            return points

        # Use 60% of budget for initial exploration? No, that's too much.
        # Let's use a smaller initial set, say 20-30 points, to establish the front.
        # The current program used 100. Let's try 30.
        n_init = min(30, self.budget // 3)
        H = halton(dim, n_init)
        X = lo + (hi - lo) * H
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # 2. Refinement: Weighted-sum scalarization with local search
        # This structural change (from greedy coverage to scalarization) is the key.
        # We will generate multiple weighted-sum objectives and optimize each using a simple local search.
        # This creates a "structured" coverage of the objective space.
        
        # Generate weight vectors. For 3 objectives, we can use a simplex lattice or random points on the simplex.
        # Let's use random points on the simplex for flexibility.
        n_weights = 15 # Number of sub-problems to solve
        weights = rng.random((n_weights, n_obj))
        weights /= weights.sum(axis=1, keepdims=True)
        
        # For each weight vector, we start from the best point in the current archive
        # that minimizes the weighted sum, and perform a few steps of local search.
        
        # Helper: Find best point for a weight vector
        def best_for_weight(F, w):
            return np.argmin(F @ w)
        
        # Helper: Local search step (simple random perturbation)
        def local_search_step(x, F, w, bounds, rng, step_size=0.1):
            # Generate a few candidates
            best_x = x
            best_f = F
            for _ in range(5): # Try 5 candidates
                noise = step_size * (bounds[1] - bounds[0]) * rng.standard_normal(dim)
                x_new = np.clip(x + noise, bounds[0], bounds[1])
                f_new = problem(x_new)
                used_global = local_search_step.used
                local_search_step.used += 1
                if f_new @ w < best_f @ w:
                    best_x = x_new
                    best_f = f_new
            return best_x, best_f, local_search_step.used

        # We need to track used budget globally.
        # Let's restructure to keep a global `used` counter.
        
        # Actually, the above helper is a bit messy with global state. Let's do it inline.
        
        # Initialize archive with the initial points
        archive_X = [X[i] for i in range(len(X))]
        archive_F = [F[i] for i in range(len(F))]
        
        # Main loop: Iterate through weight vectors, refining the best point for each
        # We will cycle through the weights until budget is exhausted.
        
        # Sort weights by some criterion? Or just iterate.
        # Let's iterate.
        
        # To make it more effective, we can restart local search from different points if stuck.
        # But simple approach: For each weight, take the current best point for that weight,
        # and do a local search.
        
        # Let's maintain a dictionary of best points per weight?
        # Or just keep the global archive and for each weight, find the best in archive,
        # then do local search from there.
        
        # This is essentially a multi-start local search with different scalarizations.
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Pick a weight vector (cycle through them)
            w_idx = used % n_weights
            w = weights[w_idx]
            
            # Find the point in the archive that best satisfies this weight
            if len(archive_F) == 0:
                # Should not happen
                break
            best_idx = np.argmin(np.array(archive_F) @ w)
            x_start = archive_X[best_idx]
            f_start = archive_F[best_idx]
            
            # Perform local search
            # We will do 5 evaluations per weight vector iteration
            n_steps = 5
            x_current = x_start
            f_current = f_start
            
            for _ in range(n_steps):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Random perturbation
                # Step size can decay slightly
                step_size = 0.05 + 0.05 * (1 - used / self.budget)
                noise = step_size * (hi - lo) * rng.standard_normal(dim)
                x_new = np.clip(x_current + noise, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                # Update if better for this specific weight
                if f_new @ w < f_current @ w:
                    x_current = x_new
                    f_current = f_new
                    
                    # Add to archive if non-dominated
                    # Check if dominated by any in archive
                    is_dominated = False
                    for f_old in archive_F:
                        if np.all(f_old <= f_new) and np.any(f_old < f_new):
                            is_dominated = True
                            break
                    
                    if not is_dominated:
                        # Remove dominated points from archive
                        new_archive_X = []
                        new_archive_F = []
                        for i in range(len(archive_X)):
                            if not (np.all(f_new <= archive_F[i]) and np.any(f_new < archive_F[i])):
                                new_archive_X.append(archive_X[i])
                                new_archive_F.append(archive_F[i])
                        
                        new_archive_X.append(x_new)
                        new_archive_F.append(f_new)
                        archive_X = new_archive_X
                        archive_F = new_archive_F
            
            # If we are stuck (no improvement in last 5 steps), we might want to restart from a different point
            # But for now, the cycling of weights should provide diversity.

        return archive_X
