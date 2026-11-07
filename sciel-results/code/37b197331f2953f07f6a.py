"""Rewritten program for RE34 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # 1. Initialization Strategy: Mix of Random and Boundary-focused
        # Budget split: 40% Init, 60% Search
        n_init = int(0.4 * self.budget)
        n_search = self.budget - n_init
        
        # Generate Initial Points
        X_init = []
        
        # 1a. Uniform Random
        n_rand = n_init // 2
        X_init.append(np.random.rand(n_rand, self.dim) * range_ + lo)
        
        # 1b. Boundary Points (Edges of the hypercube)
        # For each point, pick 1 or 2 dimensions to be at boundary
        n_bound = n_init - n_rand
        for _ in range(n_bound):
            x = np.random.rand(self.dim) * range_ + lo
            # Pick k dimensions to snap to boundary
            k = np.random.choice([1, 2], p=[0.5, 0.5])
            idxs = np.random.choice(self.dim, k, replace=False)
            for idx in idxs:
                if np.random.rand() > 0.5:
                    x[idx] = lo[idx]
                else:
                    x[idx] = hi[idx]
            X_init.append(x)
            
        X_init = np.vstack(X_init)
        F_init = np.array([problem(x) for x in X_init])
        
        used = len(X_init)
        
        # Archive of non-dominated solutions
        # We store (x, f) pairs
        archive_X = []
        archive_F = []
        
        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            if len(archive_F) == 0:
                return False
            # f is dominated if there exists g in archive such that g <= f and g != f
            # g <= f: all(g_i <= f_i)
            # g != f: not all(g_i == f_i)
            # Vectorized check
            leq = np.all(archive_F <= f, axis=1)
            neq = np.any(archive_F < f, axis=1)
            return np.any(leq & neq)

        def dominates(f, archive_F):
            """Check if f dominates any point in archive_F. Returns indices of dominated points."""
            if len(archive_F) == 0:
                return []
            leq = np.all(f <= archive_F, axis=1)
            neq = np.any(f < archive_F, axis=1)
            return np.where(leq & neq)[0]

        # Populate initial archive
        for i in range(len(X_init)):
            f = F_init[i]
            x = X_init[i]
            
            # Check if dominated by existing archive
            if is_dominated(f, np.array(archive_F) if archive_F else np.empty((0, self.n_obj))):
                continue
                
            # Remove any points in archive dominated by f
            if archive_F:
                dom_idx = dominates(f, np.array(archive_F))
                if len(dom_idx) > 0:
                    # Filter out dominated points
                    mask = np.ones(len(archive_F), dtype=bool)
                    mask[dom_idx] = False
                    archive_X = [archive_X[i] for i in range(len(archive_X)) if mask[i]]
                    archive_F = [archive_F[i] for i in range(len(archive_F)) if mask[i]]
            
            archive_X.append(x)
            archive_F.append(f)
            
        # 2. Search Phase: Adaptive Mutation
        # Step size decay
        sigma_start = 0.4 * np.max(range_)
        sigma_end = 0.05 * np.min(range_)
        
        current_sigma = sigma_start
        
        for step in range(n_search):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
                
            # Select parent
            if not archive_X:
                # Fallback to random if archive is empty (shouldn't happen)
                x_parent = np.random.rand(self.dim) * range_ + lo
            else:
                # Selection: Pick a random non-dominated point, but bias towards
                # points that are "extreme" (far from reference in objective space)
                # to encourage exploration of the front.
                # However, simple random selection from non-dominated is robust.
                # Let's use a weighted selection based on "distance to reference"
                # in objective space, normalized.
                ref = np.array([1705.0, 12.0, 0.3])
                F_arr = np.array(archive_F)
                # Distances
                dists = np.linalg.norm(F_arr - ref, axis=1)
                # We want to pick points that are NOT yet well-covered?
                # Actually, for HV, we want to fill gaps.
                # A simple heuristic: Pick the point with the largest "potential"
                # But with 200 evals, let's stick to random parent + adaptive mutation.
                # To improve on seed, we use "Boundary Init" + "Adaptive Sigma".
                
                # Random selection from archive
                idx = np.random.randint(len(archive_X))
                x_parent = archive_X[idx]
                
            # Adaptive Step Size
            progress = step / max(1, n_search)
            current_sigma = sigma_start * (1 - progress) + sigma_end * progress
            
            # Mutation
            # Occasional large jump (restart) to escape local optima
            if np.random.rand() < 0.05:
                # Large jump: random position
                x_child = np.random.rand(self.dim) * range_ + lo
            else:
                x_child = x_parent + current_sigma * np.random.randn(self.dim)
                
            # Clip to bounds
            x_child = np.clip(x_child, lo, hi)
            
            # Evaluate
            f_child = problem(x_child)
            used += 1
            
            # Update Archive
            if is_dominated(f_child, np.array(archive_F) if archive_F else np.empty((0, self.n_obj))):
                continue
                
            if archive_F:
                dom_idx = dominates(f_child, np.array(archive_F))
                if len(dom_idx) > 0:
                    mask = np.ones(len(archive_F), dtype=bool)
                    mask[dom_idx] = False
                    archive_X = [archive_X[i] for i in range(len(archive_X)) if mask[i]]
                    archive_F = [archive_F[i] for i in range(len(archive_F)) if mask[i]]
                    
            archive_X.append(x_child)
            archive_F.append(f_child)
            
        # Final Polish: If we have time and budget left (unlikely due to loop structure, 
        # but good practice), do a few tight local searches around best points.
        # The loop above handles the decay, so the last steps are already local.
        
        # Return value is ignored by evaluator, but must be returned.
        # Returning the archive X is a reasonable placeholder.
        if archive_X:
            return np.array(archive_X)
        else:
            return X_init

# EVOLVE-BLOCK-END
