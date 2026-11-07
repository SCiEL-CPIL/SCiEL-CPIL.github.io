"""Multi-objective optimizer for RE21 using hypervolume-guided search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided adaptive differential evolution for bi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, points, ref):
        """Compute exact hypervolume for 2D points against reference point."""
        if len(points) == 0:
            return 0.0
        # Filter points that are strictly better than reference in all objectives
        valid = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
        if len(valid) == 0:
            return 0.0
        
        # Sort by first objective (ascending)
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
        Returns indices of non-dominated points."""
        if len(F) == 0:
            return np.array([], dtype=int)
        # Sort by first objective ascending
        order = np.argsort(F[:, 0])
        F_sorted = F[order]
        # Walk through, keeping track of best (min) second objective seen so far
        # A point is non-dominated if its second objective is strictly less than 
        # all previously seen second objectives (since obj1 is already sorted ascending)
        n = len(F_sorted)
        nd_mask = np.ones(n, dtype=bool)
        best_y = np.inf
        for i in range(n):
            if F_sorted[i, 1] < best_y:
                best_y = F_sorted[i, 1]
            else:
                nd_mask[i] = False
        # Map back to original indices
        nd_indices = order[nd_mask]
        return nd_indices

    def _hv_contribution_2d_fast(self, point, front, ref):
        """Compute hypervolume contribution of a single point in 2D efficiently.
        front is assumed to be the Pareto front (non-dominated)."""
        if point[0] >= ref[0] or point[1] >= ref[1]:
            return 0.0
        if len(front) == 0:
            return (ref[0] - point[0]) * (ref[1] - point[1])
        
        # Sort front by obj1
        front_sorted = front[np.argsort(front[:, 0])]
        
        # Find the Pareto point with largest x < point[0]
        candidates = front_sorted[front_sorted[:, 0] < point[0]]
        if len(candidates) > 0:
            y_left = candidates[-1, 1]  # largest x, so last in sorted order
        else:
            y_left = ref[1]
        
        # The contribution is the rectangle from (point[0], point[1]) to (ref[0], y_left)
        # but only if point[1] < y_left
        if point[1] < y_left:
            return (ref[0] - point[0]) * (y_left - point[1])
        return 0.0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        ref = np.array([3000.0, 0.06])
        ranges = hi - lo
        
        # Reference-point-aligned deterministic grid initialization
        # This immediately saturates the asymmetric objective space
        pop_size = min(24, budget // 8)
        X = np.zeros((pop_size, dim))
        
        # Create a grid that aligns with the reference point structure
        # Since ref = [3000, 0.06], the first objective dominates HV contribution
        # We want diverse coverage in both objectives
        
        # Strategy: Mix of corners, midpoints, and random points
        corners = np.array([
            [lo[j] if (i >> j) & 1 else hi[j] for j in range(dim)]
            for i in range(2 ** dim)
        ])
        
        # Midpoints of each dimension
        midpoints = []
        for j in range(dim):
            for k in range(j+1, dim):
                mp = (lo + hi) / 2.0
                mp[j] = lo[j]
                mp[k] = hi[k]
                midpoints.append(mp)
        if midpoints:
            midpoints = np.array(midpoints)
        else:
            midpoints = np.array([])
        
        # Build initial population: corners first, then midpoints, then random
        X[:min(pop_size, len(corners))] = corners[:min(pop_size, len(corners))]
        remaining = pop_size - min(pop_size, len(corners))
        
        if remaining > 0 and len(midpoints) > 0:
            n_mid = min(remaining, len(midpoints))
            X[min(pop_size, len(corners)):min(pop_size, len(corners)) + n_mid] = midpoints[:n_mid]
            remaining -= n_mid
        
        if remaining > 0:
            X[pop_size - remaining:] = lo + ranges * np.random.random((remaining, dim))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for final hypervolume
        all_F = F.copy()
        
        # Adaptive differential evolution parameters
        mutation_scale = 0.3  # Initial mutation factor
        crossover_prob = 0.8
        mutation_prob = 0.9
        no_improve_count = 0
        best_hv = self._hv_2d(F, ref)
        best_hv_front = F.copy()
        
        # Adaptive mutation factor that shrinks over time
        F_factor = 0.8  # Differential evolution F parameter
        
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Extract Pareto front efficiently
            nd_indices = self._pareto_front_2d(F)
            
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])
            
            # Compute HV contributions for front points for parent selection
            front = F[nd_indices]
            contributions = np.array([
                self._hv_contribution_2d_fast(front[i], np.delete(front, i, axis=0), ref)
                for i in range(len(front))
            ])
            
            total_contrib = np.sum(contributions)
            if total_contrib < 1e-10:
                probs = np.ones(len(nd_indices)) / len(nd_indices)
            else:
                probs = contributions / total_contrib
            
            # Select parents using HV-guided selection
            if len(nd_indices) >= 3:
                p1_idx = nd_indices[np.random.choice(len(nd_indices), p=probs)]
                p2_idx = nd_indices[np.random.choice(len(nd_indices), p=probs)]
                p3_idx = nd_indices[np.random.choice(len(nd_indices), p=probs)]
                while p2_idx == p1_idx or p3_idx == p1_idx or p3_idx == p2_idx:
                    p2_idx = nd_indices[np.random.choice(len(nd_indices))]
                    p3_idx = nd_indices[np.random.choice(len(nd_indices))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                p3 = X[p3_idx]
            elif len(nd_indices) == 2:
                p1 = X[nd_indices[0]]
                p2 = X[nd_indices[1]]
                p3 = X[np.random.randint(len(X))]
            else:
                p1 = X[nd_indices[0]]
                p2 = X[np.random.randint(len(X))]
                p3 = X[np.random.randint(len(X))]
            
            # Differential evolution mutation: v = p1 + F * (p2 - p3)
            # with adaptive F that shrinks over time
            F_current = F_factor * (0.5 + 0.5 * np.exp(-used / (budget * 0.5)))
            mutant = p1 + F_current * (p2 - p3)
            
            # Add additional mutation for exploration
            if np.random.random() < mutation_prob:
                for j in range(dim):
                    if np.random.random() < 0.5:
                        delta = np.random.exponential(1.0)
                        if np.random.random() < 0.5:
                            mutant[j] += delta * mutation_scale * ranges[j]
                        else:
                            mutant[j] -= delta * mutation_scale * ranges[j]
            
            # Crossover: binomial crossover with donor
            if np.random.random() < crossover_prob:
                child = np.zeros(dim)
                for j in range(dim):
                    if np.random.random() < 0.5:
                        child[j] = p1[j]
                    else:
                        child[j] = mutant[j]
            else:
                child = mutant.copy()
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Add to all points
            all_F = np.vstack([all_F, [fc]])
            
            # Update population: replace worst HV contribution point
            # Compute HV contribution for each point in current population
            pop_contribs = np.array([
                self._hv_contribution_2d_fast(F[i], np.delete(F, i, axis=0), ref)
                for i in range(len(X))
            ])
            
            worst_idx = np.argmin(pop_contribs)
            
            # Decide whether to replace
            child_contrib = self._hv_contribution_2d_fast(fc, np.delete(F, worst_idx, axis=0), ref)
            
            if child_contrib > pop_contribs[worst_idx] * 0.7:
                X[worst_idx] = child
                F[worst_idx] = fc
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Adaptive mutation scale
            if used % 15 == 0:
                mutation_scale *= 0.92
                F_factor *= 0.98
            
            # Periodic restart: inject random points if stuck
            if no_improve_count >= 20 or (used % 30 == 0 and used > pop_size):
                n_inject = max(2, pop_size // 12)
                for _ in range(n_inject):
                    if used >= budget:
                        break
                    rand_point = lo + ranges * np.random.random(dim)
                    rand_F = problem(rand_point)
                    used += 1
                    all_F = np.vstack([all_F, [rand_F]])
                    
                    # Replace worst point
                    pop_contribs = np.array([
                        self._hv_contribution_2d_fast(F[i], np.delete(F, i, axis=0), ref)
                        for i in range(len(X))
                    ])
                    worst_idx = np.argmin(pop_contribs)
                    X[worst_idx] = rand_point
                    F[worst_idx] = rand_F
                
                no_improve_count = 0
                mutation_scale = max(0.1, mutation_scale * 1.2)
                F_factor = min(0.9, F_factor * 1.1)
        
        return X
# EVOLVE-BLOCK-END
