import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the edge-attractiveness matrix for the ACO heuristic.
    
    Uses a remaining-capacity-potential field derived from a greedy insertion tour:
    H[i,j] = dist^(-p) * (1/(1+a*dem[j]/cap)) * (1+b*depot_mask[i,j]) * (1+c*cluster_factor[j]) * cap_potential[j]
    
    where cap_potential[j] estimates the residual vehicle capacity when visiting j,
    strongly penalizing edges that risk exceeding the vehicle limit.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit

    try:
        d = np.asarray(distance_matrix, dtype=np.float64)
        N = d.shape[0]

        if not np.all(np.isfinite(d)):
            return np.ones_like(d)

        dem = np.asarray(demands, dtype=np.float64)
        cap = float(capacity)
        eps = 1e-9

        # Precompute cluster factor: lower avg distance to others = denser area
        coords = np.asarray(coordinates, dtype=np.float64)
        dist_to_others = d.copy()
        np.fill_diagonal(dist_to_others, 0)
        avg_dist = dist_to_others.mean(axis=1)
        avg_dist_max = avg_dist.max() + eps
        # cluster_factor: 1 for densest, 0 for sparsest (higher = more attractive)
        cluster_factor = 1.0 - avg_dist / avg_dist_max

        # Compute greedy insertion tour to estimate residual capacity at each node
        # This gives us a "position" for each customer in the tour, and we can estimate
        # the remaining capacity when that customer is visited
        def compute_greedy_tour_capacity_potential():
            """
            Run a simple greedy insertion to create a tour, then estimate the
            remaining vehicle capacity at each customer's visit position.
            Returns an array of length N where cap_potential[j] is in [0, 1].
            """
            # Simple greedy: start at depot, repeatedly insert the nearest feasible customer
            # that minimizes additional distance
            unvisited = set(range(1, N))
            tour = [0]  # Start at depot
            current = 0
            load = 0.0
            visit_order = []
            residual_cap_at_visit = {}
            
            while unvisited:
                # Find best insertion point: try all unvisited customers, pick the one
                # that minimizes d[current, j] + d[j, 0] - d[current, 0] (savings)
                # but also respects capacity
                best_j = -1
                best_savings = -float('inf')
                
                for j in unvisited:
                    if load + dem[j] > cap:
                        continue
                    # Savings-based: prefer customers that are close to current and depot
                    savings = d[current, j] + d[j, 0] - d[current, 0]
                    # We want to MINIMIZE additional distance, so maximize -savings
                    # But also prefer lower demand to leave more capacity
                    score = -savings - 0.1 * dem[j] / cap
                    if score > best_savings:
                        best_savings = score
                        best_j = j
                
                if best_j == -1:
                    # No feasible customer, return to depot and restart
                    tour.append(0)
                    current = 0
                    load = 0.0
                    continue
                
                # Visit best_j
                residual_cap_at_visit[best_j] = cap - load - dem[best_j]
                visit_order.append(best_j)
                tour.append(best_j)
                load += dem[best_j]
                current = best_j
                unvisited.remove(best_j)
            
            # Return to depot
            tour.append(0)
            
            # Convert residual capacity to a potential in [0, 1]
            # Higher residual = more attractive (more room for other customers)
            cap_potential = np.ones(N, dtype=np.float64)
            cap_potential[0] = 1.0  # Depot has full capacity
            
            for j in visit_order:
                if j in residual_cap_at_visit:
                    cap_potential[j] = residual_cap_at_visit[j] / cap
                else:
                    cap_potential[j] = 0.5  # Default if not visited
            
            # Customers not in visit_order (shouldn't happen) get a moderate value
            for j in range(1, N):
                if j not in residual_cap_at_visit:
                    cap_potential[j] = 0.3
            
            return cap_potential

        # Compute the capacity potential field
        cap_potential = compute_greedy_tour_capacity_potential()
        # Normalize to [0.1, 1.0] to avoid zero values
        cap_potential = 0.1 + 0.9 * cap_potential

        # Baseline
        base = 1.0 / (d + eps)

        def build_heuristic(p, a, b, c):
            """
            H[i,j] = dist^(-p) * (1/(1+a*dem[j]/cap)) * (1+b*depot_mask[i,j]) * (1+c*cluster_factor[j]) * cap_potential[j]
            """
            H = (1.0 / (d + eps) ** p).astype(np.float64)

            # Demand inverse: lower demand -> higher attractiveness
            dem_factor = 1.0 / (1.0 + a * dem / cap)
            H = H * dem_factor[np.newaxis, :]

            # Depot boost
            if b > 0:
                depot_row = (np.arange(N) == 0).astype(np.float64)
                depot_col = (np.arange(N) == 0).astype(np.float64)
                depot_mask = (depot_row[:, np.newaxis] | depot_col[np.newaxis, :]).astype(np.float64)
                H *= (1.0 + b * depot_mask)

            # Cluster factor: reward dense areas
            if c > 0:
                H = H * (1.0 + c * cluster_factor)[np.newaxis, :]

            # Capacity potential: penalize edges to nodes with low residual capacity
            H = H * cap_potential[np.newaxis, :]

            H = np.maximum(H, eps)
            return H

        def simulate_aco(heuristic, seed=42, n_ants=5, n_iters=10):
            """
            Approximate the ACO: pheromone decay 0.9, reinforce with 1/best_cost.
            Returns best total distance found by any ant in any iteration.
            """
            rng = np.random.RandomState(seed)
            best_overall = float('inf')
            pher = np.ones((N, N), dtype=np.float64)

            for iteration in range(n_iters):
                iter_best = float('inf')

                for _ant in range(n_ants):
                    visited = np.zeros(N, dtype=bool)
                    visited[0] = True
                    current = 0
                    load = 0.0
                    total_dist = 0.0

                    while True:
                        # Find feasible customers
                        feasible = []
                        for j in range(1, N):
                            if not visited[j] and load + dem[j] <= cap:
                                feasible.append(j)

                        if not feasible:
                            # Return to depot
                            total_dist += d[current, 0]
                            current = 0
                            load = 0.0
                            if visited[1:].all():
                                break
                            continue

                        fe = np.array(feasible, dtype=int)
                        weights = pher[current, fe] * heuristic[current, fe]
                        weights = np.maximum(weights, 1e-9)
                        wsum = weights.sum()
                        if wsum <= 0:
                            probs = np.ones(len(fe)) / len(fe)
                        else:
                            probs = weights / wsum
                        next_j = fe[rng.choice(len(fe), p=probs)]

                        total_dist += d[current, next_j]
                        visited[next_j] = True
                        load += dem[next_j]
                        current = next_j

                    iter_best = min(iter_best, total_dist)
                    best_overall = min(best_overall, total_dist)

                # Update pheromone
                pher *= 0.9
                if iter_best > 0:
                    pher += 1.0 / iter_best

            return best_overall

        # Grid search over (p, a, b, c)
        # p: distance power, a: demand penalty, b: depot boost, c: cluster boost
        p_vals = [0.7, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0]
        a_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        b_vals = [0.0, 0.1, 0.3, 0.5]
        c_vals = [0.0, 0.3, 0.6, 1.0, 1.5]

        best_cost = float('inf')
        best_params = (1.0, 0.0, 0.0, 0.0)
        best_heuristic = base.copy()

        eval_count = 0
        max_evals = 100  # Keep manageable within time budget

        # Coarse grid: skip some combinations to stay within budget
        for p in p_vals:
            if time.time() > deadline - 30.0:
                break
            for a in a_vals:
                if time.time() > deadline - 30.0:
                    break
                for b in b_vals:
                    if time.time() > deadline - 30.0:
                        break
                    for c in c_vals:
                        if time.time() > deadline - 30.0:
                            break
                        if eval_count >= max_evals:
                            break
                        try:
                            H = build_heuristic(p, a, b, c)
                            c_val = simulate_aco(H, seed=42, n_ants=5, n_iters=10)
                            eval_count += 1
                            if np.isfinite(c_val) and c_val < best_cost:
                                best_cost = c_val
                                best_params = (p, a, b, c)
                                best_heuristic = H.copy()
                        except Exception:
                            continue
                    if time.time() > deadline - 30.0:
                        break
                if time.time() > deadline - 30.0:
                    break
            if time.time() > deadline - 30.0:
                break

        # Local refinement around best_params
        if time.time() < deadline - 40.0:
            p0, a0, b0, c0 = best_params
            search_space = [
                (p0 * 0.9, a0, b0, c0), (p0 * 1.1, a0, b0, c0),
                (p0, max(0, a0 - 0.3), b0, c0), (p0, a0 + 0.3, b0, c0),
                (p0, a0, max(0, b0 - 0.05), c0), (p0, a0, b0 + 0.05, c0),
                (p0, a0, b0, max(0, c0 - 0.2)), (p0, a0, b0, c0 + 0.2),
                (p0 * 0.95, a0 + 0.2, b0, c0 + 0.1),
                (p0 * 1.05, max(0, a0 - 0.2), b0 + 0.05, c0),
                (p0, a0 + 0.5, b0 + 0.1, c0 + 0.3),
                (p0 * 1.1, a0 + 0.3, max(0, b0 - 0.1), c0 + 0.2),
            ]
            for (p, a, b, c) in search_space:
                if time.time() > deadline - 40.0:
                    break
                if p < 0.3 or p > 4.0 or a < -0.1 or a > 5.0 or b < -0.05 or b > 1.0 or c < -0.1 or c > 3.0:
                    continue
                try:
                    H = build_heuristic(p, a, b, c)
                    c_val = simulate_aco(H, seed=42, n_ants=5, n_iters=10)
                    if np.isfinite(c_val) and c_val < best_cost - 1e-6:
                        best_cost = c_val
                        best_params = (p, a, b, c)
                        best_heuristic = H.copy()
                except Exception:
                    continue

        # If time remains, do a second refinement with slightly different seeds
        if time.time() < deadline - 20.0:
            p0, a0, b0, c0 = best_params
            search_space2 = [
                (p0 * 0.85, a0, b0, c0), (p0 * 1.15, a0, b0, c0),
                (p0, max(0, a0 - 0.5), b0, c0), (p0, a0 + 0.5, b0, c0),
                (p0, a0, max(0, b0 - 0.1), c0), (p0, a0, b0 + 0.1, c0),
                (p0, a0, b0, max(0, c0 - 0.3)), (p0, a0, b0, c0 + 0.3),
            ]
            for (p, a, b, c) in search_space2:
                if time.time() > deadline - 20.0:
                    break
                if p < 0.3 or p > 4.0 or a < -0.1 or a > 5.0 or b < -0.05 or b > 1.0 or c < -0.1 or c > 3.0:
                    continue
                try:
                    H = build_heuristic(p, a, b, c)
                    # Use different seed for diversity
                    c_val = simulate_aco(H, seed=123, n_ants=5, n_iters=10)
                    if np.isfinite(c_val) and c_val < best_cost - 1e-6:
                        best_cost = c_val
                        best_params = (p, a, b, c)
                        best_heuristic = H.copy()
                except Exception:
                    continue

        # Final safety: ensure all positive and finite
        if not np.all(np.isfinite(best_heuristic)):
            best_heuristic = base.copy()
        best_heuristic = np.maximum(best_heuristic, eps)

        return best_heuristic

    except Exception:
        try:
            d = np.asarray(distance_matrix, dtype=np.float64)
            H = 1.0 / (d + 1e-9)
            H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
            return H
        except:
            return np.ones((50, 50))
