"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining archive-based search with LHS initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Archive of non-dominated points: list of (x, f)
        archive = []
        
        def is_dominated(f_new, archive):
            for _, f_old in archive:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False
            
        def update_archive(x_new, f_new):
            # Check if new point is dominated
            if is_dominated(f_new, archive):
                return archive
            
            # Remove points dominated by the new point
            new_archive = []
            for x_old, f_old in archive:
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    continue
                new_archive.append((x_old, f_old))
                
            new_archive.append((x_new, f_new))
            return new_archive

        def crowding_distance(archive):
            """Calculate crowding distance for points in the archive."""
            n = len(archive)
            if n <= 2:
                return [float('inf')] * n
            
            dists = [0.0] * n
            f_vals = np.array([f for _, f in archive])
            
            for m in range(n_obj):
                # Sort indices by objective m
                sorted_indices = np.argsort(f_vals[:, m])
                
                # Boundary points get infinite distance
                dists[sorted_indices[0]] = float('inf')
                dists[sorted_indices[-1]] = float('inf')
                
                # Normalize objective range
                f_min = f_vals[sorted_indices[0], m]
                f_max = f_vals[sorted_indices[-1], m]
                if f_max == f_min:
                    continue
                
                for i in range(1, n - 1):
                    idx_prev = sorted_indices[i - 1]
                    idx_curr = sorted_indices[i]
                    idx_next = sorted_indices[i + 1]
                    
                    dists[idx_curr] += (f_vals[idx_next, m] - f_vals[idx_prev, m]) / (f_max - f_min)
            
            return dists

        def tournament_selection(archive, dists, k=2):
            """Select a point from the archive using tournament selection based on crowding distance."""
            n = len(archive)
            if n == 1:
                return archive[0]
            
            # Pick k random indices
            indices = np.random.choice(n, k, replace=False)
            
            # Select the one with the highest crowding distance
            best_idx = indices[0]
            best_dist = dists[best_idx]
            for idx in indices[1:]:
                if dists[idx] > best_dist:
                    best_dist = dists[idx]
                    best_idx = idx
            
            return archive[best_idx]

        # 1. Initialization: Latin Hypercube Sampling
        n_init = max(10, int(budget * 0.1))
        
        # Generate LHS points
        perm = np.argsort(np.argsort(np.random.rand(n_init, dim)), axis=0)
        u = (perm + np.random.rand(n_init, dim)) / n_init
        X_init = lo + range_ * u
        
        # Evaluate and populate archive
        used = 0
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = X_init[i]
            f = problem(x)
            used += 1
            # Basic validity check
            if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                continue
            archive = update_archive(x, f)
            
        # 2. Iterative Refinement
        while used < budget:
            if time.time() > deadline:
                break
            
            if not archive:
                # If archive is empty, try random
                x = lo + range_ * np.random.rand(dim)
                f = problem(x)
                used += 1
                if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                    continue
                archive = update_archive(x, f)
                continue
            
            # Calculate crowding distances for parent selection
            dists = crowding_distance(archive)
            
            # Select two parents using tournament selection
            parent1 = tournament_selection(archive, dists)
            parent2 = tournament_selection(archive, dists)
            
            p1 = parent1[0]
            p2 = parent2[0]
            
            # Crossover: BLX-Alpha in normalized space
            p1_norm = (p1 - lo) / range_
            p2_norm = (p2 - lo) / range_
            
            alpha = 0.5
            lower = np.minimum(p1_norm, p2_norm) - alpha * np.abs(p1_norm - p2_norm)
            upper = np.maximum(p1_norm, p2_norm) + alpha * np.abs(p1_norm - p2_norm)
            lower = np.clip(lower, 0, 1)
            upper = np.clip(upper, 0, 1)
            
            x_child_norm = lower + (upper - lower) * np.random.rand(dim)
            x_child = lo + range_ * x_child_norm
            
            # Mutation: Gaussian noise
            progress = used / budget
            # Adaptive step size: start larger, decrease over time
            step_size = 0.15 * (1.0 - 0.5 * progress)
            
            noise = np.random.randn(dim) * step_size
            x_child = x_child + noise * range_
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            used += 1
            
            if np.any(np.isnan(f_child)) or np.any(np.isinf(f_child)):
                continue
                
            archive = update_archive(x_child, f_child)
            
        return archive
# EVOLVE-BLOCK-END
