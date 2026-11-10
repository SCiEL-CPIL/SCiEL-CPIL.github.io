"""
Ant colony orienteering heuristic: model-and-optimize with budget-aware prize density.
Uses maximum collectible prize from node j within remaining budget as a key component,
combined with distance decay terms and a soft feasibility penalty.
"""

import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # 20s safety margin before 330s kill
_params_cache = None


def _simulate_aco(prizes, dist, max_len, heuristic, n, rng):
    """Simulate ACO and return the highest total prize collected by any ant in any iteration."""
    num_ants = 20
    num_iters = 30
    pheromone_decay = 0.9
    pheromone_gain_divisor = 10.0

    pheromone = np.ones((n, n))
    dist_to_home = dist[:, 0].copy()

    global_best = 0.0

    for it in range(num_iters):
        iter_best = 0.0

        for ant in range(num_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path_len = 0.0
            total_prize = 0.0

            while True:
                remaining = max_len - path_len
                feasible = (~visited) & ((dist[current, :] + dist_to_home) <= remaining)
                feasible[0] = False

                if not np.any(feasible):
                    break

                candidates = np.where(feasible)[0]
                weights = pheromone[current, candidates] * heuristic[current, candidates]

                total_w = weights.sum()
                if total_w <= 0:
                    break

                probs = weights / total_w
                idx = rng.choice(len(candidates), p=probs)
                next_node = candidates[idx]

                path_len += dist[current, next_node]
                total_prize += prizes[next_node]
                visited[next_node] = True
                current = next_node

            if total_prize > iter_best:
                iter_best = total_prize

        pheromone *= pheromone_decay
        pheromone += iter_best / pheromone_gain_divisor

        if iter_best > global_best:
            global_best = iter_best

    return global_best


def _compute_max_prize_from_node(prizes, dist, max_len, node, exclude_node=0):
    """
    Compute the maximum total prize collectible starting from `node`,
    visiting any subset of unvisited nodes, and returning to node 0.
    This is a simplified greedy approximation: visit nodes in order of
    prize/distance ratio until budget is exhausted.
    """
    n = len(prizes)
    remaining_budget = max_len - dist[node, 0]  # budget after returning home
    if remaining_budget <= 0:
        return 0.0

    visited = np.zeros(n, dtype=bool)
    visited[node] = True
    if exclude_node >= 0:
        visited[exclude_node] = True

    total_prize = 0.0
    current = node
    path_len = 0.0

    while True:
        # Find best next node: maximize prize[j] / (dist[current,j] + dist[j,0])
        # subject to dist[current,j] + dist[j,0] <= remaining budget
        best_node = -1
        best_ratio = -1.0
        best_cost = 0.0

        for j in range(n):
            if visited[j]:
                continue
            cost = dist[current, j] + dist[j, 0]
            if cost <= max_len - path_len:
                ratio = prizes[j] / (cost + 1e-9)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_node = j
                    best_cost = cost

        if best_node < 0:
            break

        path_len += dist[current, best_node]
        total_prize += prizes[best_node]
        visited[best_node] = True
        current = best_node

    return total_prize


def _compute_prize_density_table(prizes, dist, max_len, n):
    """
    For each node j, compute the maximum prize collectible if we visit j
    (assuming we can still return home). This gives a "prize density" for
    each node that accounts for the remaining budget.
    """
    density = np.zeros(n)
    for j in range(1, n):
        # Budget available after visiting j and returning home
        cost_j = dist[0, j] + dist[j, 0]
        if cost_j > max_len:
            density[j] = 0.0
            continue
        remaining = max_len - cost_j
        # Greedy: from j, visit other nodes
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        visited[j] = True
        total_prize = prizes[j]
        current = j
        path_len = cost_j

        while True:
            best_node = -1
            best_ratio = -1.0
            for k in range(1, n):
                if visited[k]:
                    continue
                cost = dist[current, k] + dist[k, 0]
                if cost <= max_len - path_len:
                    ratio = prizes[k] / (cost + 1e-9)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_node = k

            if best_node < 0:
                break

            path_len += dist[current, best_node]
            total_prize += prizes[best_node]
            visited[best_node] = True
            current = best_node

        density[j] = total_prize

    density[0] = 0.0
    return density


def _make_heuristic(prizes, dist, max_len, params):
    """
    Generate heuristic matrix with budget-aware prize density.
    params: (a, b, c, d, e, f)
    h[i,j] = (prize[j]^a * density[j]^e) / (d_ij^b * (d_j0+eps)^c) * soft_budget^d
    where soft_budget uses logaddexp for smooth feasibility penalty.
    f controls the sharpness of the soft penalty.
    """
    n = len(prizes)
    a, b, c, d, e, f = params

    prize = prizes[None, :]  # (1, n)
    d_ij = dist + 1e-9
    d_j0 = dist[:, 0] + 1e-9
    d_j0_row = d_j0[None, :]  # (1, n)

    # Compute prize density table
    density = _compute_prize_density_table(prizes, dist, max_len, n)
    density_row = density[None, :]  # (1, n)

    # Budget: max_len - d_ij - d_j0
    budget = max_len - dist - d_j0_row  # (n, n)

    # Soft feasibility penalty using logaddexp
    # When budget is negative, we want to penalize heavily
    # Use: exp(-f * max(0, -budget)) = exp(-f * softplus(-budget))
    # softplus(x) = log(1 + exp(x))
    # For large positive x, softplus(x) ≈ x
    # So penalty = exp(-f * softplus(-budget))
    # We want this to be 1 when budget >= 0, and decay when budget < 0
    # Actually, let's use: penalty = 1 / (1 + exp(f * max(0, -budget)))
    # But for numerical stability, use:
    # penalty = exp(-f * logaddexp(0, -budget))
    # When budget >= 0: logaddexp(0, -budget) ≈ 0, penalty ≈ 1
    # When budget < 0: logaddexp(0, -budget) ≈ -budget, penalty ≈ exp(f*budget) which decays

    # Compute soft penalty
    neg_budget = -budget
    soft_penalty = np.exp(-f * np.logaddexp(0.0, neg_budget))

    # Core heuristic
    h = (prize ** a) * (density_row ** e) / (d_ij ** b * d_j0_row ** c)

    # Apply soft budget penalty
    h = h * soft_penalty

    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _optimize_parameters():
    """Optimize heuristic parameters on representative instances."""
    global _params_cache
    if _params_cache is not None:
        return _params_cache

    start_time = time.time()
    deadline = _DEADLINE - 10.0  # 10s margin

    # Generate representative instances
    n_instances = 6
    instances = []
    for seed in range(n_instances):
        rng = np.random.RandomState(seed * 7 + 3)
        n = 50
        points = rng.uniform(0, 1, size=(n, 2))
        prizes = rng.uniform(0, 1, size=n)
        prizes[0] = 0.0
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        instances.append((prizes, dist, 3.0))

    best_params = (1.0, 1.0, 0.5, 0.0, 0.5, 3.0)
    best_score = -1.0

    # Parameter grid: (a, b, c, d, e, f)
    # a: prize exponent
    # b: inverse distance exponent
    # c: inverse distance-to-home exponent
    # d: (unused, kept for compatibility)
    # e: density exponent
    # f: soft penalty sharpness
    param_grid = []

    for a in [0.5, 1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for c in [0.0, 0.5, 1.0, 1.5]:
                for e in [0.0, 0.5, 1.0, 1.5]:
                    for f in [1.0, 2.0, 3.0, 5.0]:
                        param_grid.append((a, b, 0.0, 0.0, e, f))

    # Sort by a heuristic priority: favor moderate parameters first
    param_grid.sort(key=lambda p: (abs(p[0]-1.0) + abs(p[1]-2.0) + abs(p[4]-1.0) + abs(p[5]-3.0)))

    eval_rng = np.random.RandomState(42)

    for params in param_grid:
        if time.time() > deadline:
            break

        total_score = 0.0
        for prizes, dist, max_len in instances:
            n = len(prizes)
            heuristic = _make_heuristic(prizes, dist, max_len, params)
            score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
            total_score += score

        mean_score = total_score / len(instances)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    _params_cache = best_params
    return _params_cache


def heuristics(prizes, distance_matrix, max_len):
    """Compute the heuristic matrix for ant colony optimization."""
    if time.time() > _DEADLINE - 5.0:
        # Emergency: return a simple heuristic
        n = len(prizes)
        prize = prizes[None, :]
        d_ij = distance_matrix + 1e-9
        h = prize / d_ij
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return h

    params = _optimize_parameters()
    heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)

    # Final safety check
    if not np.all(np.isfinite(heuristic)):
        h = prizes[None, :] / (distance_matrix + 1e-9)
        return np.where(np.isfinite(h) & (h > 0), h, 1e-9)

    return heuristic
