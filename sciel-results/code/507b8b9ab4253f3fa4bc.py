import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, a, b):
        # Returns True if a dominates b
        return np.all(a <= b) and np.any(a < b)

    def _update_archive(self, archive, new_points):
        # new_points: list of (x, f) tuples
        # archive: list of (x, f) tuples
        if not new_points:
            return archive
            
        # Add new points to a temporary list
        candidates = list(archive) + list(new_points)
        
        # Filter dominated points
        non_dominated = []
        for i in range(len(candidates)):
            dominated = False
            for j in range(len(candidates)):
                if i == j:
                    continue
                if self._is_dominated(candidates[j][1], candidates[i][1]):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(candidates[i])
        
        # Prune archive if too large (max size 100)
        max_archive_size = 100
        if len(non_dominated) > max_archive_size:
            # Simple grid-based pruning or random removal of least diverse
            # For simplicity and speed, we can use a grid
            F = np.array([p[1] for p in non_dominated])
            X = np.array([p[0] for p in non_dominated])
            
            # Normalize objectives for grid
            f_min = np.min(F, axis=0)
            f_max = np.max(F, axis=0)
            ranges = f_max - f_min
            ranges[ranges == 0] = 1.0
            normalized_F = (F - f_min) / ranges
            
            # Create grid
            grid_size = 20
            grid = np.zeros((grid_size,) * self.n_obj, dtype=int)
            grid_indices = []
            
            for i in range(len(normalized_F)):
                idx = tuple(int(normalized_F[i][d] * grid_size) for d in range(self.n_obj))
                idx = tuple(min(max(v, 0), grid_size - 1) for v in idx)
                grid_indices.append(idx)
                grid[idx] += 1
            
            # Keep points in less crowded grid cells
            # Sort by grid cell occupancy
            points_with_cell = list(zip(non_dominated, grid_indices))
            points_with_cell.sort(key=lambda x: grid[x[1]])
            
            # Select top max_archive_size
            non_dominated = [p[0] for p in points_with_cell[:max_archive_size]]
            
        return non_dominated

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters
        # Smaller swarm for more exploration steps? Or larger for better coverage?
        # ZDT3 is deceptive. A moderate swarm size is good.
        # Budget 200. 
        # Let's try swarm size 20. 10 generations.
        # Or swarm size 10, 19 generations.
        # MOPSO benefits from more iterations to explore the many local fronts of ZDT3.
        swarm_size = 10
        if swarm_size >= budget:
            swarm_size = max(1, budget // 2)
            
        # Initialize positions
        X = lo + (hi - lo) * np.random.rand(swarm_size, dim)
        
        # Initialize velocities
        # v = (hi - lo) * (random - 0.5)
        V = (hi - lo) * (np.random.rand(swarm_size, dim) - 0.5)
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used_evals = swarm_size
        
        # Personal bests
        P_best_X = X.copy()
        P_best_F = F.copy()
        
        # Global Archive
        archive = [(X[i], F[i]) for i in range(swarm_size)]
        archive = self._update_archive(archive, [])
        
        # Velocity coefficients
        # w: inertia weight
        # c1: cognitive coefficient
        # c2: social coefficient
        w = 0.5
        c1 = 1.5
        c2 = 1.5
        
        # Main loop
        while used_evals < budget:
            # Update velocities and positions
            new_X = []
            new_V = []
            
            for i in range(swarm_size):
                # Select global best from archive
                # Simple strategy: random selection from archive, or tournament
                # To ensure diversity, we can select from the least crowded part
                # For simplicity, random selection from archive
                if len(archive) > 0:
                    # Tournament selection of size 2 from archive
                    idx1 = np.random.randint(len(archive))
                    idx2 = np.random.randint(len(archive))
                    # Calculate crowding distance for these two? 
                    # Or just pick one randomly to encourage exploration?
                    # Let's pick one randomly to avoid getting stuck in one local front.
                    # Actually, for ZDT3, we want to cover the whole front.
                    # A better strategy: Select from archive, but prefer points with larger gap to neighbors.
                    # Given time constraints, random is a safe baseline for "new approach".
                    g_best = archive[np.random.randint(len(archive))]
                else:
                    g_best = (P_best_X[i], P_best_F[i])
                    
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                
                # V[i] = w * V[i] + c1 * r1 * (P_best - X[i]) + c2 * r2 * (G_best - X[i])
                V_new = w * V[i] + c1 * r1 * (P_best_X[i] - X[i]) + c2 * r2 * (g_best[0] - X[i])
                
                # Clip velocity to prevent overshooting bounds too much
                v_max = 0.2 * (hi - lo)
                V_new = np.clip(V_new, -v_max, v_max)
                
                X_new = X[i] + V_new
                X_new = np.clip(X_new, lo, hi)
                
                new_X.append(X_new)
                new_V.append(V_new)
                
            new_X = np.array(new_X)
            new_V = np.array(new_V)
            
            # Evaluate new positions
            new_F = []
            new_archive_additions = []
            
            for i in range(swarm_size):
                if used_evals >= budget:
                    break
                    
                f_new = problem(new_X[i])
                new_F.append(f_new)
                used_evals += 1
                
                # Check if new solution is better than personal best
                if self._is_dominated(f_new, P_best_F[i]):
                    P_best_F[i] = f_new
                    P_best_X[i] = new_X[i]
                    new_archive_additions.append((new_X[i], f_new))
                elif self._is_dominated(P_best_F[i], f_new):
                    # Current is worse, keep personal best
                    pass
                else:
                    # Non-dominated with personal best
                    # Keep both? Usually we keep the one in the archive or add both.
                    # For simplicity, we add the new one to archive candidates.
                    new_archive_additions.append((new_X[i], f_new))
                    # Also add personal best if not already in archive? 
                    # The _update_archive handles dominance.
                    # We should ensure personal best is considered.
                    # Let's just add the new point. The archive update will handle it.
            
            if not new_F:
                break
                
            # Update X and V
            X = new_X[:len(new_F)]
            V = new_V[:len(new_F)]
            
            # Update Archive
            if new_archive_additions:
                archive = self._update_archive(archive, new_archive_additions)
                
        # Return the best solutions found (from archive)
        # The evaluator records all returned objective vectors? 
        # "Every returned objective vector is recorded"
        # The return value of __call__ is ignored.
        # Wait, the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This implies the evaluator might hook into the `problem` calls?
        # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded"
        # Usually, in these tasks, the `problem` function is a wrapper that records the output.
        # So I don't need to return the best solutions from `__call__`.
        # I just need to make sure I don't exceed the budget.
        
        return None
