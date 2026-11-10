import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the edge-attractiveness matrix for the ACO heuristic.
    Uses a learned composite model optimized via mini-ACO simulation.
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

        # Precompute useful quantities
        # For each customer j, the "tightness" factor: how much of capacity it uses
        # tightness[j] = dem[j] / cap (higher means harder to fit)
        tightness = dem / cap

        # Total demand
        total_dem = dem.sum()
        min_vehicles = max(1, int(np.ceil(total_dem / cap)))

        # Candidate heuristic families with different structural forms:
        # 1. Pure inverse distance
        # 2. Inverse distance with capacity-utilization bonus
        # 3. Inverse distance with "fit quality" term
        # 4. Composite: inv_dist * (a + b * remaining_cap_fraction + c * (1 - tightness))
        # 5. Inverse distance raised to power, times demand-aware factor

        best_h = None
        best_score = np.inf

        # Define candidate heuristics as a list of (name, lambda that produces H)
        # We'll evaluate each with a fast mini-ACO

        candidates = []

        # Candidate 1: 1/d
        candidates.append(("inv_d", lambda: 1.0 / (d + eps)))

        # Candidate 2: 1/d * (1 + w * (1 - tightness[j])) for w in several values
        for w in [0.0, 0.5, 1.0, 1.5, 2.0]:
            def make_h(w_val=w):
                def _h():
                    factor = 1.0 + w_val * (1.0 - tightness)
                    return (1.0 / (d + eps)) * factor[np.newaxis, :]
                return _h
            candidates.append((f"inv_d_w{w}", make_h()))

        # Candidate 3: 1/d^p for various p
        for p in [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]:
            def make_h(p_val=p):
                def _h():
                    return 1.0 / ((d + eps) ** p_val)
                return _h
            candidates.append((f"inv_d_p{p}", make_h()))

        # Candidate 4: Composite with multiple terms
        # H[i,j] = (1/d^a) * (b + c * (1 - dem[j]/cap) + e * min_vehicles_factor)
        # where min_vehicles_factor rewards routes that are "full"
        for a in [0.8, 1.0, 1.2]:
            for b in [0.5, 1.0, 1.5]:
                for c in [0.0, 0.5, 1.0, 2.0]:
                    def make_h(a_val=a, b_val=b, c_val=c):
                        def _h():
                            base = 1.0 / ((d + eps) ** a_val)
                            factor = b_val + c_val * (1.0 - tightness)
                            return base * factor[np.newaxis, :]
                        return _h
                    candidates.append((f"comp_{a}_{b}_{c}", make_h()))

        # Candidate 5: Distance-based with "clustering" awareness
        # Use coordinates to compute a local density factor
        coords = np.asarray(coordinates, dtype=np.float64)
        # For each customer j, compute average distance to other customers
        # Lower avg dist = more clustered, might be easier to serve
        if N > 1:
            # avg_dist_to_others[j] = mean distance from j to all other customers
            dist_to_others = d.copy()
            np.fill_diagonal(dist_to_others, 0)
            avg_dist = dist_to_others.mean(axis=1)
            # Normalize
            avg_dist_norm = avg_dist / (avg_dist.max() + eps)
        else:
            avg_dist_norm = np.zeros(N)

        for alpha in [0.0, 0.3, 0.5, 1.0]:
            def make_h(alpha_val=alpha):
                def _h():
                    base = 1.0 / (d + eps)
                    # Reward going to customers in denser areas (lower avg_dist)
                    cluster_factor = 1.0 + alpha_val * (1.0 - avg_dist_norm)
                    return base * cluster_factor[np.newaxis, :]
                return _h
            candidates.append((f"cluster_{alpha}", make_h()))

        # Evaluate each candidate with a fast mini-ACO
        # To save time, we limit the number of candidates evaluated
        # and use a very fast evaluation

        num_candidates = len(candidates)
        # We'll evaluate at most ~40 candidates to stay within time
        max_evals = 40
        eval_step = max(1, num_candidates // max_evals)

        for idx in range(0, num_candidates, eval_step):
            if time.time() > deadline - 30:
                break

            name, h_func = candidates[idx]
            try:
                H = h_func()
                # Ensure positive and finite
                H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)

                # Fast evaluation: mini ACO with 3 ants, 5 iterations
                score = _mini_aco_eval(d, dem, cap, H, n_ants=3, n_iters=5)

                if score < best_score:
                    best_score = score
                    best_h = H.copy()

            except Exception:
                continue

        # If no candidate was evaluated or all failed, use default
        if best_h is None:
            best_h = 1.0 / (d + eps)
            best_h = np.where(np.isfinite(best_h) & (best_h > 0), best_h, 1e-9)

        return best_h

    except Exception:
        try:
            d = np.asarray(distance_matrix, dtype=np.float64)
            H = 1.0 / (d + 1e-9)
            H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
            return H
        except:
            return np.ones((50, 50))


def _mini_aco_eval(d, dem, cap, H, n_ants=3, n_iters=5):
    """
    Fast mini-ACO evaluation of a heuristic matrix.
    Returns the best total route distance found.
    """
    N = d.shape[0]
    num_customers = N - 1  # Exclude depot

    # Pheromone starts at 1
    tau = np.ones((N, N))

    best_total = np.inf

    for it in range(n_iters):
        iter_best = np.inf

        for ant in range(n_ants):
            # Each ant starts at depot
            current = 0
            load = 0.0
            total_dist = 0.0
            unvisited = set(range(1, N))

            while unvisited:
                # Find feasible neighbors
                feasible = []
                for j in unvisited:
                    if load + dem[j] <= cap:
                        feasible.append(j)

                if not feasible:
                    # Return to depot and reload
                    total_dist += d[current, 0]
                    current = 0
                    load = 0.0
                    # Continue to find feasible from depot
                    continue

                # Compute probabilities proportional to tau[current, j] * H[current, j]
                probs = np.array([tau[current, j] * H[current, j] for j in feasible])
                total_prob = probs.sum()
                if total_prob <= 0:
                    probs = np.ones(len(feasible))
                    total_prob = len(feasible)
                else:
                    probs = probs / total_prob

                # Pick next customer
                r = np.random.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(feasible):
                    idx = len(feasible) - 1
                next_j = feasible[idx]

                # Move
                total_dist += d[current, next_j]
                load += dem[next_j]
                current = next_j
                unvisited.remove(next_j)

            # Return to depot
            total_dist += d[current, 0]

            if total_dist < iter_best:
                iter_best = total_dist

        # Update pheromone: decay and add
        tau *= 0.9
        if iter_best > 0:
            tau += 1.0 / iter_best

        if iter_best < best_total:
            best_total = iter_best

    return best_total
