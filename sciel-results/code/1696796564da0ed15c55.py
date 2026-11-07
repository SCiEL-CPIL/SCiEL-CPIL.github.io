"""Multi-objective optimizer for RE21 using hypervolume-guided search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective optimizer with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, points, ref):
        """Compute exact hypervolume for 2D points against reference point."""
        if len(points) == 0:
            return 0.0
        valid = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
        if len(valid) == 0:
            return 0.0
        
        valid = valid[np.argsort(valid[:, 0])]
        
        hv = 0.0
        prev_y = ref[1]
        for i in range(len(valid)):
            x, y = valid[i]
            if y < prev_y:
                width = ref[0] - x
                height = prev_y - y
                hv += width * height
                prev_y = y
        return hv

    def _pareto_front_2d(self, F):
        """Extract non-dominated front from 2D objective matrix efficiently.
        Returns indices of non-dominated points sorted by obj1 ascending."""
        if len(F) == 0:
            return np.array([], dtype=int)
        order = np.argsort(F[:, 0])
        F_sorted = F[order]
        n = len(F_sorted)
        nd_mask = np.ones(n, dtype=bool)
        best_y = np.inf
        for i in range(n):
            if F_sorted[i, 1] < best_y:
                best_y = F_sorted[i, 1]
            else:
                nd_mask[i] = False
        nd_indices = order[nd_mask]
        return nd_indices

    def _hv_contribution_2d_fast(self, point, front, ref):
        """Compute hypervolume contribution of a single point in 2D efficiently."""
        if point[0] >= ref[0] or point[1] >= ref[1]:
            return 0.0
        if len(front) == 0:
            return (ref[0] - point[0]) * (ref[1] - point[1])
        
        front_sorted = front[np.argsort(front[:, 0])]
        candidates = front_sorted[front_sorted[:, 0] < point[0]]
        if len(candidates) > 0:
            y_left = candidates[-1, 1]
        else:
            y_left = ref[1]
        
        if point[1] < y_left:
            return (ref[0] - point[0]) * (y_left - point[1])
        return 0.0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        ref = np.array([3000.0, 0.06])
        ranges = hi - lo
        
        # Population size: balance between diversity and computational cost
        pop_size = 25
        X = np.zeros((pop_size, dim))
        
        # Initialize with stratified random points and corners
        corners = np.array([
            [lo[j] if (i >> j) & 1 else hi[j] for j in range(dim)]
            for i in range(2 ** dim)
        ])
        
        # Use all 16 corners + 9 random points
        X[:16] = corners
        X[16:] = lo + ranges * np.random.random((pop_size - 16, dim))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for final hypervolume
        all_F = F.copy()
        
        # Adaptive parameters
        mutation_scale = 0.2
        crossover_prob = 0.85
        restart_interval = 20
        last_restart = 0
        best_hv = self._hv_2d(F, ref)
        no_improve = 0
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Extract Pareto front efficiently
            nd_indices = self._pareto_front_2d(F)
            
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])
            
            # Compute HV contributions for front points
            front = F[nd_indices]
            contributions = np.zeros(len(front))
            for i in range(len(front)):
                front_without = np.delete(front, i, axis=0)
                contributions[i] = self._hv_contribution_2d_fast(front[i], front_without, ref)
            
            total_contrib = np.sum(contributions)
            if total_contrib < 1e-10:
                probs = np.ones(len(nd_indices)) / len(nd_indices)
            else:
                probs = contributions / total_contrib
            
            # Select two parents from front using HV contribution
            if len(nd_indices) >= 2:
                p1_idx = nd_indices[np.random.choice(len(nd_indices), p=probs)]
                p2_idx = nd_indices[np.random.choice(len(nd_indices), p=probs)]
                attempts = 0
                while p2_idx == p1_idx and attempts < 10:
                    p2_idx = nd_indices[np.random.choice(len(nd_indices))]
                    attempts += 1
                p1 = X[p1_idx]
                p2 = X[p2_idx]
            else:
                p1 = X[nd_indices[0]]
                p2 = X[np.random.randint(len(X))]
            
            # Crossover: uniform with blending
            if np.random.random() < crossover_prob:
                child = np.zeros(dim)
                for j in range(dim):
                    if np.random.random() < 0.5:
                        child[j] = p1[j]
                    else:
                        child[j] = p2[j]
                    if np.random.random() < 0.4:
                        blend = np.random.random()
                        child[j] = blend * p1[j] + (1 - blend) * p2[j]
            else:
                child = p1.copy()
            
            # Adaptive mutation with exponential distribution
            if np.random.random() < 0.95:
                for j in range(dim):
                    if np.random.random() < 0.5:
                        delta = np.random.exponential(1.0)
                        if np.random.random() < 0.5:
                            child[j] += delta * mutation_scale * ranges[j]
                        else:
                            child[j] -= delta * mutation_scale * ranges[j]
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            all_F = np.vstack([all_F, [fc]])
            
            # Update population: replace worst HV contribution point
            pop_contribs = np.zeros(len(X))
            for i in range(len(X)):
                point = F[i]
                front_without = np.delete(F, i, axis=0)
                # Only compute if point is potentially useful
                if point[0] < ref[0] and point[1] < ref[1]:
                    pop_contribs[i] = self._hv_contribution_2d_fast(point, front_without, ref)
            
            worst_idx = np.argmin(pop_contribs)
            
            # Compute child's contribution
            front_without_worst = np.delete(F, worst_idx, axis=0)
            child_contrib = self._hv_contribution_2d_fast(fc, front_without_worst, ref)
            
            if child_contrib > pop_contribs[worst_idx]:
                X[worst_idx] = child
                F[worst_idx] = fc
            
            # Check for improvement in best HV
            current_hv = self._hv_2d(F, ref)
            if current_hv > best_hv:
                best_hv = current_hv
                no_improve = 0
            else:
                no_improve += 1
            
            # Adaptive mutation scale
            if no_improve > 10:
                mutation_scale *= 0.9
                no_improve = 0
            
            # Periodic restart: inject random points to maintain diversity
            if used - last_restart >= restart_interval:
                n_inject = 3
                for _ in range(n_inject):
                    if used >= budget:
                        break
                    rand_point = lo + ranges * np.random.random(dim)
                    rand_F = problem(rand_point)
                    used += 1
                    all_F = np.vstack([all_F, [rand_F]])
                    
                    # Replace worst point
                    pop_contribs = np.zeros(len(X))
                    for i in range(len(X)):
                        point = F[i]
                        front_without = np.delete(F, i, axis=0)
                        if point[0] < ref[0] and point[1] < ref[1]:
                            pop_contribs[i] = self._hv_contribution_2d_fast(point, front_without, ref)
                    
                    worst_idx = np.argmin(pop_contribs)
                    X[worst_idx] = rand_point
                    F[worst_idx] = rand_F
                
                last_restart = used
        
        return X
# EVOLVE-BLOCK-END
