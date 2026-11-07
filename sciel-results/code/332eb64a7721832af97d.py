import numpy as np
import time
import random

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        def is_dominated(a, b):
            """Return True if b dominates a (b is better than a)."""
            return np.all(b <= a) and np.any(b < a)
        
        def archive_update(archive, new_point, max_size=100):
            """Update archive with new point, maintaining non-dominance and diversity."""
            # Check if new point is dominated by any in archive
            for a in archive:
                if is_dominated(new_point, a):
                    return archive  # new point is dominated, discard
            
            # Remove points dominated by new_point
            new_archive = []
            for a in archive:
                if not is_dominated(a, new_point):
                    new_archive.append(a)
            
            new_archive.append(new_point)
            
            if len(new_archive) <= max_size:
                return new_archive
            
            # If too large, use grid-based clustering to remove crowded points
            new_archive = np.array(new_archive)
            n = len(new_archive)
            
            # Normalize objectives to [0,1]
            mins = new_archive.min(axis=0)
            maxs = new_archive.max(axis=0)
            ranges = maxs - mins
            ranges[ranges < 1e-10] = 1.0
            normalized = (new_archive - mins) / ranges
            
            # Compute density: number of neighbors within a threshold
            threshold = 0.3
            densities = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.linalg.norm(normalized[i] - normalized[j]) < threshold:
                            densities[i] += 1
            
            # Remove the point with highest density (most crowded)
            worst = np.argmax(densities)
            new_archive = np.delete(new_archive, worst, axis=0)
            
            return list(new_archive)
        
        def select_leader(archive, particle_pos):
            """Select a leader from archive for a particle using niching."""
            if len(archive) == 0:
                return None
            if len(archive) == 1:
                return archive[0]
            
            archive_arr = np.array(archive)
            n = len(archive)
            
            if n <= 3:
                return archive[np.random.randint(0, n)]
            
            # Pick 3 random, choose the one in the least crowded region
            candidates = np.random.choice(n, size=min(3, n), replace=False)
            best_candidate = candidates[0]
            min_density = np.inf
            
            for c in candidates:
                # Count how many archive points are close to this candidate
                close_count = sum(1 for j in range(n) if np.linalg.norm(archive_arr[c] - archive_arr[j]) < 0.2)
                if close_count < min_density:
                    min_density = close_count
                    best_candidate = c
            
            return archive[best_candidate]
        
        # Initialize particles
        pop_size = 30
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        # Initialize personal bests
        pbest_pos = X.copy()
        pbest_fit = F.copy()
        
        # Initialize velocities
        v_max = 0.2 * (hi - lo)
        V = np.random.uniform(-v_max, v_max, size=(pop_size, dim))
        
        # Initialize archive
        archive = []
        for i in range(pop_size):
            archive = archive_update(archive, F[i], max_size=100)
        
        # PSO parameters
        c1 = 1.5  # cognitive
        c2 = 1.5  # social
        w_min = 0.4
        w_max = 0.9
        max_iter = max((self.budget - evaluations) // pop_size, 1)
        
        iteration = 0
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Adaptive inertia weight
            w = w_max - (w_max - w_min) * (iteration / max_iter)
            
            n_off = min(pop_size, self.budget - evaluations)
            
            for i in range(n_off):
                if time.time() > deadline:
                    break
                
                # Select leader from archive
                leader = select_leader(archive, F[i])
                if leader is None:
                    leader = pbest_fit[i]
                
                # Update velocity
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                V[i] = w * V[i] + c1 * r1 * (pbest_pos[i] - X[i]) + c2 * r2 * (leader - X[i])
                
                # Clip velocity
                V[i] = np.clip(V[i], -v_max, v_max)
                
                # Update position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F_new = problem(X[i])
                evaluations += 1
                
                # Update personal best
                if is_dominated(pbest_fit[i], F_new):
                    # pbest dominates new, keep pbest
                    pass
                elif is_dominated(F_new, pbest_fit[i]):
                    # new dominates pbest, update pbest
                    pbest_pos[i] = X[i].copy()
                    pbest_fit[i] = F_new.copy()
                else:
                    # Non-dominated: randomly choose
                    if np.random.rand() < 0.5:
                        pbest_pos[i] = X[i].copy()
                        pbest_fit[i] = F_new.copy()
                
                F[i] = F_new
                
                # Update archive
                archive = archive_update(archive, F_new, max_size=100)
            
            iteration += 1
        
        return np.zeros((1, dim))
