import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, f1, f2):
        # Returns True if f1 dominates f2
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _update_archive(self, archive, new_point, new_f):
        # Check if new_point is dominated by any point in archive
        for f in archive:
            if np.all(f <= new_f) and np.any(f < new_f):
                return False, archive
        
        # Remove points in archive dominated by new_point
        new_archive = []
        for f in archive:
            if not (np.all(new_f <= f) and np.any(new_f < f)):
                new_archive.append(f)
        
        new_archive.append(new_f)
        return True, new_archive

    def _select_global_best(self, archive):
        if len(archive) == 0:
            return None
        if len(archive) == 1:
            return archive[0]
        
        # Grid-based selection for diversity
        # Determine grid size based on archive size
        n_points = len(archive)
        grid_size = max(1, int(np.ceil(np.sqrt(n_points))))
        
        # Normalize archive to [0, 1] for grid calculation
        # We need min and max of archive
        min_vals = np.min(archive, axis=0)
        max_vals = np.max(archive, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1e-9 # Avoid division by zero
        
        # Assign each point to a grid cell
        cells = {}
        for f in archive:
            norm_f = (f - min_vals) / ranges
            cell_key = tuple([int(norm_f[i] * grid_size) % grid_size for i in range(self.n_obj)])
            if cell_key not in cells:
                cells[cell_key] = []
            cells[cell_key].append(f)
        
        # Select the cell with the fewest points
        min_count = min([len(v) for v in cells.values()])
        candidates = [v for v in cells.values() if len(v) == min_count]
        
        # Pick a random candidate cell, then a random point from it
        chosen_cell = np.random.choice(candidates)
        return np.random.choice(chosen_cell)

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_bounds = hi - lo
        
        # MOPSO Parameters
        swarm_size = 20
        max_iter = (budget - swarm_size) // swarm_size
        
        # Inertia weight decay
        w_max = 0.9
        w_min = 0.4
        
        # Initialize Swarm
        X = lo + (hi - lo) * np.random.rand(swarm_size, dim)
        V = np.zeros((swarm_size, dim))
        
        # Evaluate Initial Swarm
        F = np.array([problem(x) for x in X])
        used_evals = swarm_size
        
        # Personal Bests
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # Archive
        archive = [F[0].copy()]
        for i in range(1, swarm_size):
            _, archive = self._update_archive(archive, X[i], F[i])
            
        # Main Loop
        for t in range(max_iter):
            # Update Inertia Weight
            w = w_max - (w_max - w_min) * t / max(max_iter - 1, 1)
            
            # Select Global Best from Archive
            gbest = self._select_global_best(archive)
            
            new_X = np.zeros_like(X)
            new_F = np.zeros_like(F)
            
            for i in range(swarm_size):
                # Update Velocity
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                V[i] = w * V[i] + 1.5 * r1 * (Pbest_X[i] - X[i]) + 1.5 * r2 * (gbest - X[i])
                
                # Update Position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                f_new = problem(X[i])
                new_F[i] = f_new
                used_evals += 1
                
                # Update Personal Best
                if np.all(f_new <= Pbest_F[i]) and np.any(f_new < Pbest_F[i]):
                    Pbest_X[i] = X[i].copy()
                    Pbest_F[i] = f_new.copy()
                elif np.all(Pbest_F[i] <= f_new) and np.any(Pbest_F[i] < f_new):
                    # Pbest is better, keep it
                    pass
                else:
                    # They are non-dominated? For simplicity in MOPSO, 
                    # we often just keep the one with better hypervolume or random.
                    # Given the small budget, let's just check if new is better in at least one objective 
                    # and not worse in others? No, that's not dominance.
                    # Standard MOPSO: If non-dominated, we can keep either. 
                    # Let's keep the new one to encourage movement, or random.
                    # Actually, if they are non-dominated, we should probably keep the one 
                    # that is more diverse or has better fitness. 
                    # For simplicity, let's keep the new one if it's not strictly worse.
                    # But wait, if they are non-dominated, neither dominates the other.
                    # We will keep the new one to ensure the particle moves.
                    Pbest_X[i] = X[i].copy()
                    Pbest_F[i] = f_new.copy()
                    
                # Update Archive
                _, archive = self._update_archive(archive, X[i], f_new)
                
            # Update X and F for next iteration
            X = new_X
            F = new_F
            
            if used_evals >= budget:
                break
                
        return X
