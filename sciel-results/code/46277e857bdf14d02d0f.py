# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using decomposition-based search (MOEA/D)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        # --- Generate weight vectors using Das-Dennis method ---
        # For 3 objectives, we need points on a simplex
        # Number of weight vectors: use a grid with resolution
        n_points_per_edge = 8  # gives C(8+3-1, 3-1) = C(10,2) = 45 weight vectors
        weight_vectors = self._generate_weight_vectors(n_obj, n_points_per_edge)
        n_weights = len(weight_vectors)

        # --- Phase 1: Initialize population by minimizing Tchebycheff for each weight ---
        # For each weight vector, do a few local optimization steps
        # Start with random points
        X = lo + (hi - lo) * rng.random((n_weights, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = n_weights

        # Track best solution for each weight (minimize Tchebycheff value)
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)
        range_f = nadir - ideal
        range_f[range_f < 1e-10] = 1.0

        # Tchebycheff value for each solution under each weight
        def tchebycheff(F_row, w, ideal_pt, range_pt):
            return np.max(w * np.abs(F_row - ideal_pt) / range_pt)

        # Store best solution index for each weight
        best_indices = np.arange(n_weights)  # initially each is its own
        best_tch = np.array([tchebycheff(F[i], weight_vectors[i], ideal, range_f) for i in range(n_weights)])

        # --- Phase 2: Iterative improvement ---
        # Each iteration: for each weight, generate a neighbor and try to improve
        remaining = budget - used
        if remaining > 0:
            # Adaptive step size
            n_iters = min(remaining // n_weights + 1, remaining)
            
            for it in range(n_iters):
                if time.time() > deadline - 10:
                    break
                if used >= budget:
                    break
                
                # Update ideal point
                ideal = np.min(F, axis=0)
                nadir = np.max(F, axis=0)
                range_f = nadir - ideal
                range_f[range_f < 1e-10] = 1.0
                
                # For each weight vector, generate a candidate
                # Use a mix of exploitation (small steps) and exploration (larger steps)
                for w_idx in range(n_weights):
                    if used >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    w = weight_vectors[w_idx]
                    center = X[best_indices[w_idx]]
                    
                    # Adaptive step: larger early, smaller late
                    progress = it / max(1, n_iters - 1)
                    base_step = 0.2 * (1.0 - 0.6 * progress) + 0.05
                    
                    # Generate child with Gaussian perturbation
                    child = center + base_step * (hi - lo) * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    
                    # Update ideal point
                    ideal = np.minimum(ideal, fc)
                    nadir = np.maximum(nadir, fc)
                    range_f = nadir - ideal
                    range_f[range_f < 1e-10] = 1.0
                    
                    # Check if child improves Tchebycheff value for this weight
                    tch_child = tchebycheff(fc, w, ideal, range_f)
                    
                    # Also check against neighbors (weights close to w)
                    # For simplicity, just check if it's better for this weight
                    if tch_child < best_tch[w_idx]:
                        X[best_indices[w_idx]] = child
                        F[best_indices[w_idx]] = fc
                        best_tch[w_idx] = tch_child
                        
                        # Update the stored solution
                        X[w_idx] = child
                        F[w_idx] = fc
                        best_indices[w_idx] = w_idx

        # --- Final: collect all non-dominated solutions from the population ---
        # Get unique non-dominated points
        all_F = F.copy()
        all_X = X.copy()
        
        # Find non-dominated indices
        n = len(all_F)
        nd_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd_mask[i]:
                continue
            for j in range(n):
                if i == j or not nd_mask[j]:
                    continue
                if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                    nd_mask[i] = False
                    break
        
        return all_X[nd_mask]

    def _generate_weight_vectors(self, n_obj, n_points):
        """Generate weight vectors on a simplex using Das-Dennis method."""
        from itertools import combinations
        
        weights = []
        
        if n_obj == 1:
            return np.array([[1.0]])
        
        elif n_obj == 2:
            for i in range(n_points + 1):
                w = np.array([i / n_points, (n_points - i) / n_points])
                weights.append(w)
        
        elif n_obj == 3:
            for i in range(n_points + 1):
                for j in range(n_points - i + 1):
                    k = n_points - i - j
                    w = np.array([i / n_points, j / n_points, k / n_points])
                    weights.append(w)
        
        else:
            # General case: use simplex lattice
            def simplex_lattice(n_obj, n_div):
                """Generate points on simplex with n_div divisions."""
                if n_obj == 1:
                    return [np.array([1.0])]
                
                def _gen(n_obj, n_div, remaining):
                    if n_obj == 1:
                        return [np.array([remaining / n_div])]
                    
                    results = []
                    for i in range(remaining + 1):
                        sub = _gen(n_obj - 1, n_div, remaining - i)
                        for s in sub:
                            results.append(np.append(np.array([i / n_div]), s))
                    return results
                
                return _gen(n_obj, n_div, n_div)
            
            weights = simplex_lattice(n_obj, n_points)
        
        return np.array(weights, dtype=np.float64)

# EVOLVE-BLOCK-END
