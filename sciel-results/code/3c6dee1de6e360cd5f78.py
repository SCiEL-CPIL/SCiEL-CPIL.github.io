"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit

    n = adjacency.shape[0]
    adjacency = adjacency.astype(bool)

    # Precompute neighbor lists for faster simulation
    neighbor_lists = [np.where(adjacency[i])[0] for i in range(n)]

    # Precompute neighbor mask arrays for vectorized simulation
    neighbor_masks = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if len(neighbor_lists[i]) > 0:
            neighbor_masks[i, neighbor_lists[i]] = True

    def simulate_greedy(order):
        """Simulate greedy decoder given an explicit vertex order (descending priority)."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_lists[v]) > 0:
                    selected[neighbor_lists[v]] = True
        return count

    def simulate_greedy_fast(order):
        """Fast simulation using integer marking to avoid repeated allocation."""
        mark = np.zeros(n, dtype=np.int32)
        stamp = 1
        count = 0
        for v in order:
            if mark[v] != stamp:
                count += 1
                mark[v] = stamp
                nbrs = neighbor_lists[v]
                if len(nbrs) > 0:
                    mark[nbrs] = stamp
        return count

    degrees = adjacency.sum(axis=1).astype(float)
    rng = np.random.RandomState(42)

    # Compute iterative relaxation scores (captures global structure)
    def compute_relaxation_scores(max_iters=50, lr=0.5):
        """Iterative relaxation: score_i = 1/(1+sum_j in N(i) score_j)"""
        scores = np.ones(n)
        for _ in range(max_iters):
            if time.time() > deadline - 30:
                break
            # sum of neighbor scores
            neighbor_score_sums = adjacency.astype(float) @ scores
            new_scores = 1.0 / (1.0 + neighbor_score_sums)
            scores = (1 - lr) * scores + lr * new_scores
        return scores

    relaxation_scores = compute_relaxation_scores(max_iters=80, lr=0.4)

    # Mean degree of neighbors (structural feature)
    sum_neighbor_degrees = adjacency.astype(float) @ degrees
    avg_neighbor_degree = np.zeros(n)
    nonzero_mask = degrees > 0
    avg_neighbor_degree[nonzero_mask] = sum_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]

    # Smooth soft-min surrogate: score_i = log-sum-exp over neighbors of -degree_j / temperature
    # This gives a smooth approximation to "prefer vertices whose neighbors have low degree"
    def compute_softmin_scores(temperature=2.0, max_iters=30):
        """Iterative soft-min: score_i captures smooth penalty for high-degree neighbors."""
        scores = -degrees / temperature
        adj_float = adjacency.astype(float)
        for _ in range(max_iters):
            if time.time() > deadline - 30:
                break
            # Soft-max of neighbor scores (log-sum-exp trick)
            shifted = adj_float @ scores
            # Where no neighbors, shifted is 0; use a small negative to avoid log(0)
            shifted = np.where(adjacency.any(axis=1), shifted, -100.0)
            # log-sum-exp: log(sum(exp(shifted)))
            max_shifted = np.max(shifted, axis=0)
            max_shifted = np.where(np.isfinite(max_shifted), max_shifted, 0.0)
            log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted), axis=0))
            # New score: negative of soft-max (want low neighbor scores)
            new_scores = -log_sum_exp
            # Normalize to comparable scale
            new_scores = new_scores - np.mean(new_scores)
            scores = 0.7 * scores + 0.3 * new_scores
        return scores

    softmin_scores = compute_softmin_scores(temperature=2.0, max_iters=40)

    # Generate several initial orderings to start local search from
    initial_orders = []
    # 1. Ascending degree (low degree first)
    initial_orders.append(np.argsort(degrees))
    # 2. Relaxation scores (higher score = more "independent" = pick first)
    initial_orders.append(np.argsort(-relaxation_scores))
    # 3. Combined: low degree + high relaxation score
    combined = degrees / (degrees.max() + 1e-9) - relaxation_scores / (relaxation_scores.max() + 1e-9)
    initial_orders.append(np.argsort(combined))
    # 4. Low avg neighbor degree (prefer vertices whose neighbors have low degree)
    initial_orders.append(np.argsort(avg_neighbor_degree))
    # 5. Soft-min surrogate (smooth structural balance)
    initial_orders.append(np.argsort(-softmin_scores))
    # 6. Combined soft-min + relaxation
    sm_norm = softmin_scores / (np.abs(softmin_scores).max() + 1e-9)
    rel_norm = relaxation_scores / (relaxation_scores.max() + 1e-9)
    combined2 = sm_norm + rel_norm
    initial_orders.append(np.argsort(-combined2))
    # 7. Random
    initial_orders.append(rng.permutation(n))
    # 8. Degree with small random noise for tie-breaking
    deg_noise = degrees + rng.uniform(0, 1, n) * 0.01
    initial_orders.append(np.argsort(deg_noise))
    # 9. Another random
    initial_orders.append(rng.permutation(n))
    # 10. Random perturbation of relaxation
    relax_noise = relaxation_scores + rng.uniform(0, 1, n) * 0.05
    initial_orders.append(np.argsort(-relax_noise))
    # 11. Random perturbation of soft-min
    sm_noise = softmin_scores + rng.uniform(0, 1, n) * 0.05
    initial_orders.append(np.argsort(-sm_noise))

    best_order = initial_orders[0].copy()
    best_size = simulate_greedy_fast(best_order)

    for idx, init_order in enumerate(initial_orders):
        if time.time() > deadline - 12:
            break
        curr_order = init_order.copy()
        curr_size = simulate_greedy_fast(curr_order)

        # Phase 1: Adjacent swap local search (multiple passes until no improvement)
        improved = True
        max_passes = 30
        passes = 0
        while improved and time.time() < deadline - 8 and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n - 1):
                if time.time() > deadline - 8:
                    break
                v1 = curr_order[i]
                v2 = curr_order[i + 1]
                curr_order[i], curr_order[i + 1] = v2, v1
                new_size = simulate_greedy_fast(curr_order)
                if new_size > curr_size:
                    curr_size = new_size
                    improved = True
                else:
                    curr_order[i], curr_order[i + 1] = v1, v2

        # Phase 2: Random swap local search to escape local optima
        no_improve = 0
        max_no_improve = 300
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            i, j = rng.randint(0, n, 2)
            if i == j:
                continue
            v1 = curr_order[i]
            v2 = curr_order[j]
            curr_order[i], curr_order[j] = v2, v1
            new_size = simulate_greedy_fast(curr_order)
            if new_size > curr_size:
                curr_size = new_size
                no_improve = 0
            else:
                curr_order[i], curr_order[j] = v1, v2
                no_improve += 1

        if curr_size > best_size:
            best_size = curr_size
            best_order = curr_order.copy()

    # Final polish: more thorough adjacent swap passes on the best order
    improved = True
    passes = 0
    max_polish_passes = 20
    while improved and time.time() < deadline - 2 and passes < max_polish_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            if time.time() > deadline - 2:
                break
            v1 = best_order[i]
            v2 = best_order[i + 1]
            best_order[i], best_order[i + 1] = v2, v1
            new_size = simulate_greedy_fast(best_order)
            if new_size > best_size:
                best_size = new_size
                improved = True
            else:
                best_order[i], best_order[i + 1] = v1, v2

    # Convert order to scores: earlier in order = higher score
    scores = np.zeros(n)
    for pos, v in enumerate(best_order):
        scores[v] = -pos

    return scores


# EVOLVE-BLOCK-END
