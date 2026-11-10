import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the edge-attractiveness matrix for the ACO heuristic.
    Combines parametric family search with mini-ACO evaluation,
    plus a greedy reference tour boost for structural guidance.
    """
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 300s limit

    try:
        d = np.asarray(distance_matrix, dtype=np.float64)
        N = d.shape[0]

        if not np.all(np.isfinite(d)):
            return np.ones_like(d)

        dem = np.asarray(demands, dtype=np.float64)
        cap = float(capacity)
        eps = 1e-9

        # Baseline heuristic
        base = 1.0 / (d + eps)
        result = base.copy()

        # Fast ACO simulator for scoring candidate heuristics
        def simulate_aco(heuristic, seed=0, n_ants=5, n_iters=8):
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

        def build_heuristic(p, a, b, boost_w=0.0):
            """
            Build heuristic with depot-pressure field:
            H[i,j] = dist^(-p) * demand_eff[j] * cap_potential[i] * depot_pressure[i,j] * (1 + boost)
            
            - demand_eff[j]: high-demand nodes are more attractive targets (cluster them)
            - cap_potential[i]: departing from nodes with more remaining capacity is better
            - depot_pressure: edges from depot to high-demand nodes get extra boost
            """
            H = (1.0 / (d + eps) ** p).astype(np.float64)

            # Demand efficiency: higher demand -> more attractive to visit (cluster high-demand)
            # Use (1 + a * dem_j/cap) so high-demand nodes get boosted
            dem_factor = 1.0 + a * dem / cap
            H = H * dem_factor[np.newaxis, :]

            # Capacity potential per node: approximate expected remaining capacity when at node i
            # Nodes closer to depot (lower index in greedy order) have more remaining capacity
            # Use a proxy: 1/cap as baseline, boosted for depot
            # For static heuristic, use demand-based proxy:
            # If node i has high demand, it likely gets visited early, so less remaining cap after
            # We want to boost edges FROM nodes that typically have more remaining capacity
            # Proxy: depot has full capacity, others scale by (1 - dem_i/cap)
            cap_pot = np.ones(N, dtype=np.float64)
            cap_pot[0] = 1.0  # depot always full
            for i in range(1, N):
                cap_pot[i] = 1.0 - 0.5 * (dem[i] / cap)  # higher demand -> less remaining cap
            cap_pot = np.maximum(cap_pot, 0.1)
            H = H * cap_pot[:, np.newaxis]

            # Depot pressure: edges from depot to high-demand nodes get extra boost
            # This encourages visiting high-demand customers first (when truck is full)
            if b > 0:
                depot_row = (np.arange(N) == 0).astype(np.float64)
                depot_col = (np.arange(N) == 0).astype(np.float64)
                depot_mask = (depot_row[:, np.newaxis] | depot_col[np.newaxis, :]).astype(np.float64)
                # Extra pressure: from depot, boost proportional to demand of target
                depot_pressure = 1.0 + b * (depot_row[:, np.newaxis] * dem[np.newaxis, :] / cap)
                # From customers to depot, mild boost
                depot_pressure += 0.5 * b * (depot_col[np.newaxis, :] * 1.0)
                H *= depot_pressure

            # Greedy reference tour boost
            if boost_w > 0:
                greedy_edges = _get_greedy_edges(d, dem, cap)
                for (i, j) in greedy_edges:
                    if i < N and j < N:
                        H[i, j] *= (1.0 + boost_w)

            H = np.maximum(H, eps)
            return H

        def _get_greedy_edges(dist, demands, capacity):
            """
            Build a greedy nearest-neighbor tour respecting capacity.
            Returns list of (i, j) edges used.
            """
            n = dist.shape[0]
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            load = 0.0
            edges = []

            while True:
                feasible = []
                for j in range(1, n):
                    if not visited[j] and load + demands[j] <= capacity:
                        feasible.append(j)

                if not feasible:
                    # Return to depot
                    edges.append((current, 0))
                    current = 0
                    load = 0.0
                    if visited[1:].all():
                        break
                    continue

                # Pick nearest feasible
                min_dist = float('inf')
                best_j = -1
                for j in feasible:
                    if dist[current, j] < min_dist:
                        min_dist = dist[current, j]
                        best_j = j

                edges.append((current, best_j))
                visited[best_j] = True
                load += demands[best_j]
                current = best_j

            return edges

        # Coarse grid search over (p, a, b, boost_w)
        # a now boosts high-demand nodes (positive = more attractive)
        p_vals = [0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
        a_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        b_vals = [0.0, 0.1, 0.2, 0.4, 0.6]
        boost_vals = [0.0, 0.5, 1.0, 2.0]

        best_cost = float('inf')
        best_params = (1.0, 0.0, 0.0, 0.0)
        best_heuristic = base.copy()

        # Evaluate a subset to stay within time budget
        # Total combos: 6*5*3*3 = 270, but we'll sample
        eval_count = 0
        max_evals = 60  # Keep it manageable

        for p in p_vals:
            if time.time() > deadline - 30.0:
                break
            for a in a_vals:
                if time.time() > deadline - 30.0:
                    break
                for b in b_vals:
                    if time.time() > deadline - 30.0:
                        break
                    for bw in boost_vals:
                        if time.time() > deadline - 30.0:
                            break
                        if eval_count >= max_evals:
                            break
                        try:
                            H = build_heuristic(p, a, b, bw)
                            c = simulate_aco(H, seed=42, n_ants=5, n_iters=8)
                            eval_count += 1
                            if np.isfinite(c) and c < best_cost:
                                best_cost = c
                                best_params = (p, a, b, bw)
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
            p0, a0, b0, bw0 = best_params
            search_space = [
                (p0 * 0.9, a0, b0, bw0), (p0 * 1.1, a0, b0, bw0),
                (p0, max(0, a0 - 0.3), b0, bw0), (p0, a0 + 0.3, b0, bw0),
                (p0, a0, max(0, b0 - 0.05), bw0), (p0, a0, b0 + 0.05, bw0),
                (p0, a0, b0, max(0, bw0 - 0.3)), (p0, a0, b0, bw0 + 0.3),
                (p0 * 0.95, a0 + 0.2, b0, bw0 + 0.2),
                (p0 * 1.05, max(0, a0 - 0.2), b0 + 0.05, bw0),
                (p0, a0 + 0.5, b0 + 0.1, bw0),
                (p0 * 1.1, a0 + 0.3, max(0, b0 - 0.1), bw0 + 0.2),
            ]
            for (p, a, b, bw) in search_space:
                if time.time() > deadline - 40.0:
                    break
                if p < 0.3 or p > 4.0 or a < -0.1 or a > 5.0 or b < -0.05 or b > 1.0 or bw < -0.1 or bw > 3.0:
                    continue
                try:
                    H = build_heuristic(p, a, b, bw)
                    c = simulate_aco(H, seed=42, n_ants=5, n_iters=8)
                    if np.isfinite(c) and c < best_cost - 1e-6:
                        best_cost = c
                        best_params = (p, a, b, bw)
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
