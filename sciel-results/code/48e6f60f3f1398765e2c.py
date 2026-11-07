"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, seed):
    """Simulate the ACO colony and return the best prize collected by any ant."""
    np.random.seed(seed)

    n = len(prizes)
    if n <= 1:
        return 0.0

    n_ants = 20
    n_iter = 30
    evaporation = 0.9
    deposition = 1.0 / 10.0

    # Pheromone matrix initialized to 1
    pheromone = np.ones((n, n), dtype=np.float64)

    # Ensure heuristic is positive
    h = np.maximum(heuristic, 1e-9)

    best_total = 0.0

    # Precompute d_j0 for speed
    d_j0 = distance_matrix[:, 0]

    for it in range(n_iter):
        iter_best = 0.0

        for ant in range(n_ants):
            visited = [0]
            current = 0
            length = 0.0
            total_prize = prizes[0]  # prize at node 0 is 0
            visited_set = {0}

            while True:
                # Find eligible next nodes
                eligible = []
                for j in range(1, n):
                    if j not in visited_set:
                        d_ij = distance_matrix[current, j]
                        if length + d_ij + d_j0[j] <= max_len:
                            eligible.append(j)

                if not eligible:
                    break

                # Compute probabilities proportional to pheromone * heuristic
                probs = np.array([pheromone[current, j] * h[current, j] for j in eligible], dtype=np.float64)
                total_p = probs.sum()

                if total_p <= 0 or not np.isfinite(total_p):
                    break

                # Normalize and pick
                probs = probs / total_p
                probs = np.maximum(probs, 0)
                probs_sum = probs.sum()
                if probs_sum <= 0:
                    break
                probs = probs / probs_sum

                # Sample
                r = np.random.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(eligible):
                    idx = len(eligible) - 1

                next_j = eligible[idx]

                # Update state
                length += distance_matrix[current, next_j]
                total_prize += prizes[next_j]
                visited.append(next_j)
                visited_set.add(next_j)
                current = next_j

            if total_prize > iter_best:
                iter_best = total_prize

        if iter_best > best_total:
            best_total = iter_best

        # Update pheromone
        pheromone = pheromone * evaporation
        pheromone += deposition * iter_best

    return best_total


def _compute_future_prize(prizes, distance_matrix, max_len):
    """
    future_prize[j] = total prize of nodes k (k != j) that are still reachable
    from j within the length budget left after going 0 -> j.
    A structural lookahead signal: nodes that keep many high-prize options open.
    """
    n = len(prizes)
    d = distance_matrix
    d_to_0 = d[:, 0]

    future_prize = np.zeros(n)
    for j in range(1, n):
        remaining = max_len - d_to_0[j]
        if remaining <= 0:
            future_prize[j] = 0.0
            continue
        s = 0.0
        for k in range(1, n):
            if k == j:
                continue
            if d[j, k] + d_to_0[k] <= remaining:
                s += prizes[k]
        future_prize[j] = s

    return future_prize


def _compute_optionality_cost(prizes, distance_matrix, max_len):
    """
    Dominance-frontier / optionality-preservation term.

    For each node j, count how many nodes k that are reachable from node 0
    (d[0,k] + d[k,0] <= max_len) would become infeasible to visit if the ant
    commits to j next (i.e. the remaining length after 0->j can no longer reach
    k and return to 0). A larger count means visiting j closes off more future
    options, so this is a *cost* to be subtracted from node attractiveness.
    """
    n = len(prizes)
    d = distance_matrix
    d_to_0 = d[:, 0]

    # Nodes reachable directly from 0 within budget
    reach_from_0 = [k for k in range(1, n) if d[0, k] + d_to_0[k] <= max_len]

    optionality_cost = np.zeros(n)
    for j in range(1, n):
        remaining = max_len - d_to_0[j]
        if remaining <= 0:
            optionality_cost[j] = 0.0
            continue
        cost = 0.0
        for k in reach_from_0:
            if k == j:
                continue
            # After committing to j, can we still reach k and return to 0?
            if d[j, k] + d_to_0[k] <= remaining:
                continue
            # k becomes infeasible -> optionality lost
            cost += prizes[k]
        optionality_cost[j] = cost

    return optionality_cost


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for ACO orienteering.

    Node affinity combines three structural signals:
      - prize^beta * exp(-d_j0 / scale)            (high prize, near home)
      - (1 + gamma * future_prize)                 (keeps high-prize options open)
      - omega * optionality_cost                   (penalizes moves that close options)
    Edge (i,j) attractiveness = node_affinity[j] / d[i,j]^alpha.

    A parameter sweep with two-seed validation picks the best configuration.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)

    n = len(prizes)
    if n == 0:
        return np.array([])

    if n == 1:
        return np.ones((1, 1))

    eps = 1e-9
    d_j0 = distance_matrix[:, 0]

    # Start time for budget management
    start_time = time.time()
    time_limit = 290.0  # Safety margin before 300s kill

    # Precompute structural components (computed once, reused across candidates)
    future_prize = _compute_future_prize(prizes, distance_matrix, max_len)
    optionality_cost = _compute_optionality_cost(prizes, distance_matrix, max_len)

    prize_pow_base = np.maximum(prizes, 1e-9)
    d_plus = distance_matrix + eps

    best_score = -1.0
    best_h = None

    # Candidate parameter space
    alphas = [0.5, 1.0, 1.5, 2.0]
    scales = [0.3, 0.5, 0.7, 1.0, 1.5]
    betas = [1.0, 1.5, 2.0]
    gammas = [0.0, 0.5, 1.0]
    omegas = [0.0, 0.2, 0.5]

    # Build candidates, but cap the count to keep within budget.
    # We prioritize a curated set first, then fill in.
    candidates = []
    for alpha in alphas:
        for scale in scales:
            for beta in betas:
                for gamma in gammas:
                    for omega in omegas:
                        candidates.append((alpha, scale, beta, gamma, omega))

    # Shuffle to spread risk across the space
    np.random.seed(42)
    indices = np.random.permutation(len(candidates))

    # Estimate per-candidate cost to decide how many to try.
    # Measure one simulation to get a rough per-sim time.
    t0 = time.time()
    _dummy_h = np.maximum(prize_pow_base[None, :] / d_plus, 1e-9)
    _simulate_aco(prizes, distance_matrix, max_len, _dummy_h, seed=42)
    per_sim = max(time.time() - t0, 1e-6)

    # Budget for candidate testing: leave headroom for the second-seed validation.
    remaining_budget = time_limit - (time.time() - start_time)
    # Each candidate costs ~1 sim, plus ~0.5 sim for second-seed on good ones.
    max_candidates = int(remaining_budget / (per_sim * 1.5))
    max_candidates = max(max_candidates, 4)

    for idx in indices[:max_candidates]:
        if time.time() - start_time > time_limit:
            break

        alpha, scale, beta, gamma, omega = candidates[idx]

        # Compute heuristic
        prize_pow = prize_pow_base ** beta
        d_decay = np.exp(-d_j0 / scale)

        # Node affinity with structural signals
        node_affinity = prize_pow * d_decay * (1.0 + gamma * future_prize)
        node_affinity = node_affinity - omega * optionality_cost

        # h[i][j] = node_affinity[j] / d[i][j]^alpha
        h = node_affinity[None, :] / (d_plus ** alpha)
        h = np.maximum(h, 1e-9)

        # Test with one seed first
        score = _simulate_aco(prizes, distance_matrix, max_len, h, seed=42)
        if score > best_score:
            best_score = score
            best_h = h.copy()

        # If this looks good, test with second seed
        if score > best_score * 0.95 and time.time() - start_time < time_limit * 0.8:
            score2 = _simulate_aco(prizes, distance_matrix, max_len, h, seed=123)
            avg = (score + score2) / 2.0
            if avg > best_score:
                best_score = avg
                best_h = h.copy()

    # If no good candidate found, use a reasonable default
    if best_h is None:
        node_affinity = prizes * np.exp(-d_j0 / 0.5) * (1.0 + future_prize)
        best_h = node_affinity[None, :] / d_plus
        best_h = np.maximum(best_h, 1e-9)

    return best_h
# EVOLVE-BLOCK-END
