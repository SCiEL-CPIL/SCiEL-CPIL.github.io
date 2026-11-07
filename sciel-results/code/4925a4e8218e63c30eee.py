import numpy as np


class Optimizer:
    """MOPSO multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo

        pop_size = 20
        archive_size = 20
        
        # Initialize positions
        X = lo + range_vec * rng.random((pop_size, self.dim))
        # Initialize velocities
        V = 0.1 * range_vec * (rng.random((pop_size, self.dim)) - 0.5)
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Personal bests
        P_best_X = X.copy()
        P_best_F = F.copy()
        
        # Global Archive (list of (F, X) or just F and X arrays)
        # We store indices or arrays. Let's store arrays.
        archive_F = []
        archive_X = []
        
        def is_dominated(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def crowding_distance(F_list, X_list):
            # F_list: list of objective vectors
            # Returns list of distances
            n = len(F_list)
            if n <= 2:
                return [float('inf')] * n
            
            cd = [0.0] * n
            m = self.n_obj
            
            # Convert to numpy for easier slicing
            F_arr = np.array(F_list)
            
            for k in range(m):
                # Sort by k-th objective
                sorted_idx = np.argsort(F_arr[:, k])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                
                f_min = F_arr[sorted_idx[0], k]
                f_max = F_arr[sorted_idx[-1], k]
                f_range = f_max - f_min
                if f_range < 1e-10:
                    continue
                
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F_arr[sorted_idx[i+1], k] - F_arr[sorted_idx[i-1], k]) / f_range
            
            return cd

        # Initialize Archive
        for i in range(pop_size):
            # Check if F[i] is dominated by any in archive
            dominated = False
            for j in range(len(archive_F)):
                if is_dominated(F[i], archive_F[j]):
                    dominated = True
                    break
            
            if not dominated:
                # Remove points in archive dominated by F[i]
                new_archive_F = []
                new_archive_X = []
                for j in range(len(archive_F)):
                    if is_dominated(archive_F[j], F[i]):
                        continue
                    new_archive_F.append(archive_F[j])
                    new_archive_X.append(archive_X[j])
                
                archive_F = new_archive_F
                archive_X = new_archive_X
                
                # Add F[i]
                archive_F.append(F[i])
                archive_X.append(X[i])
                
                # If archive is too large, remove worst
                if len(archive_F) > archive_size:
                    cds = crowding_distance(archive_F, archive_X)
                    min_idx = np.argmin(cds)
                    archive_F.pop(min_idx)
                    archive_X.pop(min_idx)

        # Select initial global best
        G_best_F = archive_F[0]
        G_best_X = archive_X[0]
        if len(archive_F) > 1:
            cds = crowding_distance(archive_F, archive_X)
            best_idx = np.argmax(cds)
            G_best_F = archive_F[best_idx]
            G_best_X = archive_X[best_idx]

        w = 0.9
        c1 = 1.49445
        c2 = 1.49445
        
        while used < self.budget:
            for i in range(pop_size):
                if used >= self.budget:
                    break
                
                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)
                
                # Update velocity
                V[i] = w * V[i] + c1 * r1 * (P_best_X[i] - X[i]) + c2 * r2 * (G_best_X - X[i])
                
                # Update position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F[i] = problem(X[i])
                used += 1
                
                # Update Personal Best
                if is_dominated(F[i], P_best_F[i]):
                    # F[i] is dominated by P_best
                    pass
                elif is_dominated(P_best_F[i], F[i]):
                    # F[i] dominates P_best
                    P_best_F[i] = F[i]
                    P_best_X[i] = X[i]
                else:
                    # Non-dominated
                    # Keep P_best? Or replace with random probability?
                    # Standard MOPSO often keeps P_best if non-dominated to maintain stability,
                    # or uses a rule. Let's keep P_best.
                    pass

                # Update Archive
                dominated = False
                for j in range(len(archive_F)):
                    if is_dominated(F[i], archive_F[j]):
                        dominated = True
                        break
                
                if not dominated:
                    new_archive_F = []
                    new_archive_X = []
                    for j in range(len(archive_F)):
                        if is_dominated(archive_F[j], F[i]):
                            continue
                        new_archive_F.append(archive_F[j])
                        new_archive_X.append(archive_X[j])
                    
                    archive_F = new_archive_F
                    archive_X = new_archive_X
                    
                    archive_F.append(F[i])
                    archive_X.append(X[i])
                    
                    if len(archive_F) > archive_size:
                        cds = crowding_distance(archive_F, archive_X)
                        min_idx = np.argmin(cds)
                        archive_F.pop(min_idx)
                        archive_X.pop(min_idx)

            # Update Global Best
            if len(archive_F) > 0:
                cds = crowding_distance(archive_F, archive_X)
                best_idx = np.argmax(cds)
                G_best_F = archive_F[best_idx]
                G_best_X = archive_X[best_idx]

        return X
