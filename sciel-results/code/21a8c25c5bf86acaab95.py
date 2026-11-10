import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the edge-attractiveness matrix for the ACO heuristic.
    Combines a parametric family (distance exponent, demand weight, depot bias)
    with a fast greedy-based local search to find instance-specific parameters.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit

    try:
        d = np.asarray(distance_matrix, dtype=np.float64)
        N = d.shape[0]

        if not np.all(np.isfinite(d)):
            return np.ones_like(d)

        dem = np.asarray(demands, dtype=np.float64)
        cap = float(capacity)
        eps = 1e-9

        # Precompute factors for speed
        # factors[j] represents "demand efficiency" or "ease of serving"
        # High demand -> lower factor if we want to prefer low demand, or vice versa.
        # Let's define: factor[j] = 1 / (1 + w * dem[j]/cap)
        # If w > 0, high demand gets lower attractiveness.
        # If w < 0, high demand gets higher attractiveness.

        def evaluate_greedy(H):
            """
            Fast greedy evaluation of a heuristic matrix.
            Returns total distance of the constructed routes.
            This serves as a surrogate for the ACO performance.
            """
            N_local = d.shape[0]
            unvisited = np.ones(N_local, dtype=bool)
            unvisited[0] = False
            current = 0
            load = 0.0
            total_dist = 0.0
            count_visited = 0

            while count_visited < N_local - 1:
                best_j = -1
                best_h = -1.0

                # Vectorized check for feasible nodes
                feasible_mask = unvisited & (load + dem <= cap)
                if not np.any(feasible_mask):
                    # Return to depot
                    total_dist += d[current, 0]
                    current = 0
                    load = 0.0
                    # Retry
                    feasible_mask = unvisited & (load + dem <= cap)
                    if not np.any(feasible_mask):
                        # Should not happen if total demand <= cap * num_vehicles
                        # But if it does, break to avoid infinite loop
                        break

                # Get indices of feasible nodes
                feasible_indices = np.where(feasible_mask)[0]
                if len(feasible_indices) == 0:
                    break

                # Get heuristic values for feasible nodes from current
                h_vals = H[current, feasible_indices]
                # Find max
                idx_max = np.argmax(h_vals)
                best_h = h_vals[idx_max]
                best_j = feasible_indices[idx_max]

                # Move to best_j
                total_dist += d[current, best_j]
                load += dem[best_j]
                current = best_j
                unvisited[best_j] = False
                count_visited += 1

            # Return to depot
            total_dist += d[current, 0]

            return total_dist

        def build_heuristic(p, w, b):
            """
            Build heuristic matrix from parameters.
            H[i, j] = (1 / (dist[i, j] + eps)^p) * (1 / (1 + w * dem[j]/cap)) * (1 + b * depot_mask[i, j])
            """
            # Distance term
            d_safe = d + eps
            d_pow = d_safe ** p
            inv_d = 1.0 / d_pow

            # Demand term (column-wise multiplier)
            # dem[j] is the demand of the destination node
            dem_factor = 1.0 / (1.0 + w * dem / cap)
            # Handle potential negative denominator if w is very negative
            # But w is constrained in search, so this should be safe.
            # If 1 + w*dem/cap <= 0, clip to small positive
            dem_factor = np.maximum(dem_factor, 1e-9)

            H = inv_d * dem_factor[np.newaxis, :]

            # Depot bias
            if b != 0.0:
                # Increase attractiveness of edges involving the depot
                depot_mask = np.zeros((N, N), dtype=np.float64)
                depot_mask[0, :] = 1.0
                depot_mask[:, 0] = 1.0
                # Diagonal is 0 distance, so 1/eps^p is huge.
                # We don't really care about self-loops, but let's keep them consistent.
                H *= (1.0 + b * depot_mask)

            # Ensure positive and finite
            H = np.where(np.isfinite(H) & (H > 0), H, eps)

            return H

        # Grid search for p, w, b
        # p: distance exponent
        # w: demand weight
        # b: depot bias

        # Initial coarse grid
        p_vals = [0.8, 1.0, 1.2, 1.5, 2.0]
        w_vals = [-0.5, 0.0, 0.5, 1.0, 1.5]
        b_vals = [0.0, 0.2, 0.5]

        best_cost = np.inf
        best_p = 1.0
        best_w = 0.0
        best_b = 0.0

        for p in p_vals:
            if time.time() > deadline - 10:
                break
            for w in w_vals:
                if time.time() > deadline - 10:
                    break
                for b in b_vals:
                    if time.time() > deadline - 10:
                        break

                    try:
                        H = build_heuristic(p, w, b)
                        cost = evaluate_greedy(H)

                        if cost < best_cost:
                            best_cost = cost
                            best_p = p
                            best_w = w
                            best_b = b
                    except Exception:
                        continue

        # Local refinement around best parameters
        if time.time() < deadline - 20:
            # Refine p
            p_range = np.linspace(max(0.5, best_p - 0.3), min(3.0, best_p + 0.3), 5)
            for p in p_range:
                if time.time() > deadline - 20:
                    break
                try:
                    H = build_heuristic(p, best_w, best_b)
                    cost = evaluate_greedy(H)
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p
                except Exception:
                    pass

            # Refine w
            w_range = np.linspace(max(-2.0, best_w - 0.5), min(3.0, best_w + 0.5), 5)
            for w in w_range:
                if time.time() > deadline - 20:
                    break
                try:
                    H = build_heuristic(best_p, w, best_b)
                    cost = evaluate_greedy(H)
                    if cost < best_cost:
                        best_cost = cost
                        best_w = w
                except Exception:
                    pass

            # Refine b
            b_range = np.linspace(0.0, 1.0, 5)
            for b in b_range:
                if time.time() > deadline - 20:
                    break
                try:
                    H = build_heuristic(best_p, best_w, b)
                    cost = evaluate_greedy(H)
                    if cost < best_cost:
                        best_cost = cost
                        best_b = b
                except Exception:
                    pass

        # Final polish: try a few combinations of the best individual parameters
        if time.time() < deadline - 10:
            # Try slightly perturbing the best found
            perturbations = [
                (best_p * 0.95, best_w, best_b),
                (best_p * 1.05, best_w, best_b),
                (best_p, best_w * 0.9, best_b),
                (best_p, best_w * 1.1, best_b),
                (best_p, best_w, best_b + 0.1),
                (best_p, best_w, best_b - 0.1),
            ]
            for p, w, b in perturbations:
                if time.time() > deadline - 10:
                    break
                # Bounds check
                if p < 0.5 or p > 3.0 or w < -2.0 or w > 3.0 or b < 0.0 or b > 1.0:
                    continue
                try:
                    H = build_heuristic(p, w, b)
                    cost = evaluate_greedy(H)
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p
                        best_w = w
                        best_b = b
                except Exception:
                    pass

        # Construct final heuristic
        H_final = build_heuristic(best_p, best_w, best_b)

        # Final safety checks
        if not np.all(np.isfinite(H_final)):
            H_final = np.where(np.isfinite(H_final), H_final, eps)
        H_final = np.maximum(H_final, eps)

        return H_final

    except Exception:
        # Fallback to standard 1/d
        try:
            d = np.asarray(distance_matrix, dtype=np.float64)
            return 1.0 / (d + 1e-9)
        except Exception:
            return np.ones((50, 50))
