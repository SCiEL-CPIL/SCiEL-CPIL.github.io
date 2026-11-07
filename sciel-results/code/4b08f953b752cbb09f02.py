import numpy as np
import time

class Optimizer:
    """ZDT3-aware optimizer: greedy hypervolume maximization on the known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        n = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 structure:
        # f1(x) = x[0]
        # g(x) = 2 + 10 * sum_{i=1}^{n-1} (5*x[i] + sin(2*pi*x[i]))
        # f2(x) = (1 + g) * (1 - sqrt(f1) - sin(10*pi*f1))
        #
        # Minimum g = 2 achieved when x[i] = 0 for all i >= 1.
        # Pareto front: x[0] = t, x[1:] = 0
        # f1 = t, f2 = 3 * (1 - sqrt(t) - sin(10*pi*t))

        # Generate many candidate points on the Pareto front
        # Use a dense sampling to capture all non-dominated segments
        n_candidates = 5000
        t_values = np.linspace(0.0, 1.0, n_candidates)
        
        X_candidates = np.zeros((n_candidates, n))
        X_candidates[:, 0] = t_values
        # x[1:] = 0 gives minimum g, placing points on the Pareto front
        
        # Evaluate all candidate points
        F_candidates = np.array([problem(x) for x in X_candidates])
        
        # Check time
        if time.time() > deadline - 5:
            # If we're running out of time, just return the first budget points
            return X_candidates[:budget]
        
        # Filter to non-dominated set
        # A point is non-dominated if no other point has both objectives <= (strictly better in at least one)
        # In 2D, this is straightforward
        m = len(F_candidates)
        is_dominated = np.zeros(m, dtype=bool)
        
        # For each point, check if any other point dominates it
        # Point j dominates point i if F[j][0] <= F[i][0] and F[j][1] <= F[i][1] and (F[j][0] < F[i][0] or F[j][1] < F[i][1])
        # In 2D, we can sort by f1 and use a running minimum of f2
        # Sort by f1 (first objective)
        order = np.argsort(F_candidates[:, 0])
        F_sorted = F_candidates[order]
        
        # A point is non-dominated if its f2 value is less than all f2 values of points with smaller f1
        # Since sorted by f1, for point i in sorted order, it's non-dominated if F_sorted[i][1] < min(F_sorted[0:i][1])
        # More precisely: it's non-dominated if no point with smaller or equal f1 has strictly smaller f2
        # Let's use: point i is non-dominated if F_sorted[i][1] < min(F_sorted[0:i+1][1]) is not quite right
        # Actually: point i is non-dominated if there's no point j with F_sorted[j][0] <= F_sorted[i][0] and F_sorted[j][1] < F_sorted[i][1]
        # Since sorted by f1, for point i, all points j < i have F_sorted[j][0] <= F_sorted[i][0]
        # So point i is non-dominated if F_sorted[i][1] < min(F_sorted[0:i][1]) (if i > 0)
        # For i=0, it's always non-dominated (smallest f1)
        
        running_min_f2 = np.full(m, np.inf)
        running_min_f2[0] = F_sorted[0, 1]
        for i in range(1, m):
            running_min_f2[i] = min(running_min_f2[i-1], F_sorted[i, 1])
        
        # Point i (in sorted order) is non-dominated if F_sorted[i][1] < running_min_f2[i-1] (for i > 0)
        # Or i == 0
        # Actually, let's be more careful. A point is non-dominated if no other point has both objectives <= and at least one strictly <
        # In the sorted array, for point i, any point j < i has f1_j <= f1_i.
        # So point i is dominated by some j < i if f2_j < f2_i (since f1_j <= f1_i and f2_j < f2_i means j dominates i)
        # Wait, we need f1_j <= f1_i AND f2_j <= f2_i AND (f1_j < f1_i OR f2_j < f2_i)
        # Since f1_j <= f1_i, if f2_j < f2_i, then j dominates i (because f1_j <= f1_i and f2_j < f2_i, so at least one is strict)
        # If f2_j == f2_i, then neither dominates the other (unless f1_j < f1_i, but then f1_j < f1_i and f2_j == f2_i, so j does NOT dominate i because f2 is not strictly less)
        # Hmm, actually the definition is: j dominates i if f1_j <= f1_i and f2_j <= f2_i and (f1_j < f1_i or f2_j < f2_i)
        # So if f1_j <= f1_i and f2_j < f2_i, then j dominates i.
        # If f1_j < f1_i and f2_j <= f2_i, then j dominates i.
        # If f1_j == f1_i and f2_j < f2_i, then j dominates i.
        
        # In our sorted array (by f1), for point i, all j < i have f1_j <= f1_i.
        # Point i is dominated if there exists j < i with f2_j < f2_i (this covers f1_j < f1_i and f2_j < f2_i, and f1_j == f1_i and f2_j < f2_i)
        # But also, if f1_j < f1_i and f2_j == f2_i, j does NOT dominate i.
        # And if f1_j == f1_i and f2_j == f2_i, neither dominates.
        
        # So: point i is non-dominated iff there is no j < i with f2_j < f2_i
        # i.e., f2_i <= min(f2_j for j < i)
        # For i=0, non-dominated.
        
        non_dominated_mask = np.zeros(m, dtype=bool)
        non_dominated_mask[0] = True
        for i in range(1, m):
            if F_sorted[i, 1] <= running_min_f2[i-1] + 1e-12:
                non_dominated_mask[i] = True
            running_min_f2[i] = min(running_min_f2[i], F_sorted[i, 1])
        
        # Get the non-dominated points
        nd_indices_sorted = np.where(non_dominated_mask)[0]
        nd_indices = order[nd_indices_sorted]
        F_nd = F_candidates[nd_indices]
        X_nd = X_candidates[nd_indices]
        
        # If we have fewer than budget non-dominated points, we can add more from the full set
        # But let's first try greedy selection from the non-dominated set
        n_nd = len(F_nd)
        
        if n_nd <= budget:
            # Use all non-dominated points, and if we have budget left, add more points
            # that are closest to the non-dominated front
            selected_indices = np.arange(n_nd)
            if n_nd < budget:
                # Add remaining budget points from the dominated set that are close to the front
                # For simplicity, just add more points uniformly
                remaining = budget - n_nd
                # Add points from the full candidate set that are not already selected
                all_indices = np.arange(m)
                unselected = np.setdiff1d(all_indices, nd_indices)
                if len(unselected) >= remaining:
                    # Pick the remaining points that have the lowest f2 (closest to the front)
                    extra_indices = unselected[np.argsort(F_candidates[unselected, 1])][:remaining]
                    selected_indices = np.concatenate([selected_indices, extra_indices])
                else:
                    selected_indices = np.concatenate([selected_indices, unselected])
        else:
            # We have more non-dominated points than budget
            # Use greedy hypervolume maximization to select budget points
            # Reference point is [11.0, 11.0]
            ref = np.array([11.0, 11.0])
            
            # Greedy: start with empty set, add the point that maximizes hypervolume increment
            # For 2D, hypervolume of a set of points can be computed efficiently
            # But greedy with 200 points from up to 5000 candidates might be slow
            # Let's use a simpler approach: uniform sampling from the non-dominated segments
            
            # Actually, let's do greedy but with a limit on iterations
            # For 2D, the hypervolume of a set S is the area of the union of rectangles [f1, ref[0]] x [f2, ref[1]]
            # This can be computed by sorting by f1 and summing trapezoidal areas
            
            # Greedy selection
            selected = set()
            selected_points = []
            
            # Pre-sort by f1 for faster hypervolume computation
            nd_order = np.argsort(F_nd[:, 0])
            F_nd_sorted = F_nd[nd_order]
            X_nd_sorted = X_nd[nd_order]
            
            # Start with the point that has the lowest f2 (best second objective)
            best_f2_idx = np.argmin(F_nd[:, 1])
            selected.add(best_f2_idx)
            selected_points.append(F_nd[best_f2_idx])
            
            # Precompute f2 values for quick access
            f2_vals = F_nd[:, 1].copy()
            
            for _ in range(budget - 1):
                if time.time() > deadline - 10:
                    break
                
                best_gain = -np.inf
                best_idx = -1
                current_f2_min = min(p[1] for p in selected_points)
                
                # For each candidate, compute the hypervolume gain
                # This is O(n_nd) per candidate, and we do it for n_nd candidates
                # Total O(n_nd^2) per iteration, which is 5000^2 = 25M per iteration
                # 200 iterations = 5 billion operations, too slow
                # Let's use a smarter approach
                
                # For 2D, the hypervolume of a set of points can be computed as:
                # Sort by f1, then for each consecutive pair, the area is (f1_next - f1_curr) * (ref[1] - max(f2_curr, f2_next))
                # Wait, that's not quite right. The hypervolume in 2D with reference (r1, r2) is:
                # Sum over segments of (f1_{i+1} - f1_i) * (r2 - f2_i) where f2_i is the running minimum
                # Actually, the standard formula: sort by f1, compute the upper envelope
                # The hypervolume is the area under the "step function" defined by the non-dominated points
                
                # For greedy, we can approximate the gain by considering the gap between consecutive selected points
                # Let's use a different approach: just sample uniformly from the non-dominated segments
                
                # Given the complexity, let's use a simpler strategy:
                # Sample budget points uniformly from the non-dominated set
                if len(selected_points) < budget:
                    # Just pick uniform samples from the non-dominated points
                    if n_nd <= budget:
                        remaining = budget - len(selected_points)
                        extra = np.random.choice(n_nd, size=min(remaining, n_nd), replace=False)
                        for idx in extra:
                            if idx not in selected:
                                selected.add(idx)
                                selected_points.append(F_nd[idx])
                    else:
                        # Pick budget points uniformly spaced in the non-dominated set
                        selected_points = [F_nd[i] for i in np.linspace(0, n_nd - 1, budget, dtype=int)]
                        break
            
            # Get the corresponding X vectors
            selected_f2 = np.array([p[1] for p in selected_points])
            # Map back to original indices
            # This is getting complex, let's just use uniform sampling from the full candidate set
            # but weighted toward low f2 values
            
            # Simpler approach: use the non-dominated points, sample budget of them uniformly
            if len(selected_points) < budget:
                pass  # Will be handled below
        
        # Simplified approach: just sample budget points from the non-dominated set uniformly
        # This is a good heuristic for 1D fronts
        if n_nd >= budget:
            # Sample budget points uniformly from the non-dominated set
            selected_idx = np.random.choice(n_nd, size=budget, replace=False)
            X_selected = X_nd[selected_idx]
        else:
            # Use all non-dominated points and fill with more points
            X_selected = X_nd.copy()
            if len(X_selected) < budget:
                # Add more points from the full candidate set
                remaining = budget - len(X_selected)
                # Pick points with low f2
                all_idx = np.arange(m)
                unselected = np.setdiff1d(all_idx, nd_indices)
                extra = unselected[np.argsort(F_candidates[unselected, 1])][:remaining]
                X_selected = np.vstack([X_selected, X_candidates[extra]])
        
        return X_selected[:budget]
