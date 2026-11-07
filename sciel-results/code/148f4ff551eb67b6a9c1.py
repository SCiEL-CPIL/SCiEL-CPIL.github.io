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

    def _hv_contribution_2d(self, point, front, ref):
        """Compute hypervolume contribution of a single point in 2D."""
        if len(front) == 0:
            # Contribution is the area of the rectangle from point to ref
            if point[0] < ref[0] and point[1] < ref[1]:
                return (ref[0] - point[0]) * (ref[1] - point[1])
            return 0.0
        
        # Create a copy of front without the point
        front_list = list(front)
        front_list.remove(point)
        
        hv_with = self._hv_2d(np.array([point] + front_list), ref)
        hv_without = self._hv_2d(np.array(front_list), ref)
        
        return max(0.0, hv_with - hv_without)

    def _non_dominated_mask(self, F):
        """Return boolean mask where True means the point is non-dominated."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            # Check if any other point dominates i
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j <= i in all objectives and j < i in at least one
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        ref = np.array([3000.0, 0.06])
        
        # Initial population: mix of random and boundary points
        pop_size = min(40, budget // 5)
        X = np.zeros((pop_size, dim))
        
        # Some random points
        n_random = pop_size // 2
        X[:n_random] = lo + (hi - lo) * np.random.random((n_random, dim))
        
        # Some points near boundaries (corners of the hypercube)
        corners = []
        for i in range(2 ** dim):
            corner = np.array([(hi[j] if (i >> j) & 1 else lo[j]) for j in range(dim)])
            corners.append(corner)
        corners = np.array(corners)
        
        # Select remaining from corners
        remaining = pop_size - n_random
        if len(corners) >= remaining:
            idx = np.random.choice(len(corners), remaining, replace=False)
            X[n_random:] = corners[idx]
        else:
            # If not enough corners, fill with more random
            extra = remaining - len(corners)
            X[n_random:n_random + len(corners)] = corners
            if extra > 0:
                X[n_random + len(corners):] = lo + (hi - lo) * np.random.random((extra, dim))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best front
        all_points = X.copy()
        all_F = F.copy()
        
        # Adaptive mutation scale
        mutation_scale = 0.3
        restart_interval = 30
        last_restart = 0
        
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            nd_mask = self._non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])
            
            # Hypervolume-based parent selection
            # Compute HV contribution for each non-dominated point
            front = F[nd_indices]
            contributions = []
            for idx in nd_indices:
                point = F[idx]
                # Remove this point from front for contribution calc
                front_without = np.delete(front, list(nd_indices).index(idx), axis=0)
                hv_with = self._hv_2d(np.array([point] + front_without), ref)
                hv_without = self._hv_2d(front_without, ref)
                contributions.append(max(0.0, hv_with - hv_without))
            
            contributions = np.array(contributions)
            total_contrib = np.sum(contributions)
            if total_contrib < 1e-10:
                probs = np.ones(len(nd_indices)) / len(nd_indices)
            else:
                probs = contributions / total_contrib
            
            # Select two parents
            if len(nd_indices) >= 2:
                p1_idx = np.random.choice(nd_indices, p=probs)
                p2_idx = np.random.choice(nd_indices, p=probs)
                # Ensure different parents
                if p1_idx == p2_idx:
                    p2_idx = nd_indices[np.random.choice(len(nd_indices))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
            else:
                p1 = X[nd_indices[0]]
                p2 = X[np.random.randint(len(X))]
            
            # SBX crossover
            if np.random.random() < 0.8:
                child = np.zeros(dim)
                for j in range(dim):
                    if np.random.random() < 0.5:
                        child[j] = p1[j]
                    else:
                        child[j] = p2[j]
            else:
                child = p1.copy()
            
            # Polynomial mutation
            if np.random.random() < 0.9:
                for j in range(dim):
                    if np.random.random() < 0.5:
                        delta = np.random.exponential(1.0)
                        if np.random.random() < 0.5:
                            child[j] += delta * mutation_scale * (hi[j] - lo[j])
                        else:
                            child[j] -= delta * mutation_scale * (hi[j] - lo[j])
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Add to all points
            all_points = np.vstack([all_points, [child]])
            all_F = np.vstack([all_F, [fc]])
            
            # Update population: replace the point with worst HV contribution
            # Compute HV contribution for each point in current population
            pop_front = F
            pop_contribs = []
            for i in range(len(X)):
                point = F[i]
                front_without = np.delete(pop_front, i, axis=0)
                hv_with = self._hv_2d(np.array([point] + front_without), ref)
                hv_without = self._hv_2d(front_without, ref)
                pop_contribs.append(max(0.0, hv_with - hv_without))
            
            pop_contribs = np.array(pop_contribs)
            worst_idx = np.argmin(pop_contribs)
            
            # Decide whether to replace
            child_contrib = 0.0
            front_without_worst = np.delete(pop_front, worst_idx, axis=0)
            hv_with_child = self._hv_2d(np.array([fc] + front_without_worst), ref)
            hv_without_worst = self._hv_2d(front_without_worst, ref)
            child_contrib = max(0.0, hv_with_child - hv_without_worst)
            
            if child_contrib > pop_contribs[worst_idx] * 0.8:
                X[worst_idx] = child
                F[worst_idx] = fc
            
            # Adaptive mutation scale
            if used % 10 == 0:
                mutation_scale *= 0.95
            
            # Periodic restart: inject random points
            if used - last_restart >= restart_interval:
                n_inject = max(2, pop_size // 10)
                for _ in range(n_inject):
                    if used >= budget:
                        break
                    rand_point = lo + (hi - lo) * np.random.random(dim)
                    rand_F = problem(rand_point)
                    used += 1
                    all_points = np.vstack([all_points, [rand_point]])
                    all_F = np.vstack([all_F, [rand_F]])
                    
                    # Replace worst point
                    pop_contribs = []
                    for i in range(len(X)):
                        point = F[i]
                        front_without = np.delete(pop_front, i, axis=0)
                        hv_with = self._hv_2d(np.array([point] + front_without), ref)
                        hv_without = self._hv_2d(front_without, ref)
                        pop_contribs.append(max(0.0, hv_with - hv_without))
                    pop_contribs = np.array(pop_contribs)
                    worst_idx = np.argmin(pop_contribs)
                    X[worst_idx] = rand_point
                    F[worst_idx] = rand_F
                    pop_front = F
                
                last_restart = used
        
        return X
# EVOLVE-BLOCK-END
