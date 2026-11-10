import numpy as np
import time
from scipy.optimize import minimize
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # Safety margin

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
                # Feasibility: dist[current, j] + dist[j, 0] <= remaining
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

def _compute_future_prize_table(prizes, dist, max_len):
    """
    Precompute for each node j a structure to quickly query
    the sum of prizes of nodes k reachable from j with remaining budget B.
    Reachability condition: dist[j, k] + dist[k, 0] <= B
    """
    n = len(prizes)
    # future_prize_data[j] = (sorted_costs, prefix_prizes)
    future_prize_data = []
    
    for j in range(n):
        if j == 0:
            # Node 0 is start/end, usually not visited as 'next' in the same way
            # But we can still compute it, though it won't be used as a destination from 0
            # because visited[0] is True.
            future_prize_data.append((np.array([]), np.array([])))
            continue
            
        costs = np.zeros(n)
        for k in range(n):
            if k == j or k == 0:
                costs[k] = np.inf
            else:
                costs[k] = dist[j, k] + dist[k, 0]
        
        # Sort by cost
        sorted_indices = np.argsort(costs)
        sorted_costs = costs[sorted_indices]
        sorted_prizes = prizes[sorted_indices]
        
        # Prefix sums
        prefix_prizes = np.cumsum(sorted_prizes)
        
        future_prize_data.append((sorted_costs, prefix_prizes))
        
    return future_prize_data

def _get_future_prize(future_prize_data, j, budget):
    """
    Get the sum of prizes of nodes k such that dist[j, k] + dist[k, 0] <= budget.
    """
    sorted_costs, prefix_prizes = future_prize_data[j]
    if len(sorted_costs) == 0:
        return 0.0
    # Find the rightmost index where cost <= budget
    # np.searchsorted returns the index where budget would be inserted to maintain order
    # side='right' means if budget exists, it returns the index after it.
    # We want the last index where cost <= budget.
    idx = np.searchsorted(sorted_costs, budget, side='right') - 1
    if idx < 0:
        return 0.0
    return prefix_prizes[idx]

def _make_heuristic(prizes, dist, max_len, params):
    """
    Generate heuristic matrix.
    params: (alpha, beta, gamma, delta)
    H[i, j] = (Prize[j] + alpha * FuturePrize(j, B_ij))^delta / (d_ij^beta * d_j0^gamma)
    """
    n = len(prizes)
    alpha, beta, gamma, delta = params
    
    # Precompute future prize data
    fp_data = _compute_future_prize_table(prizes, dist, max_len)
    
    d_ij = dist + 1e-9
    d_j0 = dist[:, 0] + 1e-9
    d_j0_row = d_j0[None, :] # (1, n)
    
    H = np.zeros((n, n))
    
    # Vectorized calculation is tricky because FuturePrize depends on B_ij which varies per (i,j)
    # But N=50, so N^2 = 2500 queries. This is fast enough.
    
    for i in range(n):
        for j in range(n):
            if i == j or j == 0:
                H[i, j] = 1e-9
                continue
                
            B_ij = max_len - dist[i, j] - dist[j, 0]
            if B_ij < 0:
                H[i, j] = 1e-9
                continue
                
            fp_val = _get_future_prize(fp_data, j, B_ij)
            
            # Core value
            val = prizes[j] + alpha * fp_val
            
            # Ensure val is positive for power
            if val <= 0:
                val = 1e-9
                
            denom = (d_ij[i, j] ** beta) * (d_j0_row[0, j] ** gamma)
            
            H[i, j] = (val ** delta) / denom
            
    # Replace non-finite or non-positive with small value
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def _optimize_parameters():
    """Optimize heuristic parameters on representative instances."""
    
    # Generate representative instances
    # The task mentions "3 fixed instances". We don't know the seeds.
    # We will use a set of random seeds to find a robust heuristic.
    n_instances = 5
    instances = []
    for seed in range(n_instances):
        rng = np.random.RandomState(seed * 11 + 5)
        n = 50
        points = rng.uniform(0, 1, size=(n, 2))
        prizes = rng.uniform(0, 1, size=n)
        prizes[0] = 0.0
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        instances.append((prizes, dist, 3.0))

    eval_rng = np.random.RandomState(42)
    
    def objective(params):
        if time.time() > _DEADLINE - 10.0:
            return 1e9 # Timeout penalty
            
        alpha, beta, gamma, delta = params
        
        # Constraints: params should be positive to avoid issues with powers
        if any(p < 0.01 for p in params):
            return 1e9
            
        total_score = 0.0
        for prizes, dist, max_len in instances:
            n = len(prizes)
            heuristic = _make_heuristic(prizes, dist, max_len, params)
            score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
            total_score += score

        mean_score = total_score / len(instances)
        return -mean_score # Minimize negative score

    # Initial guess
    x0 = np.array([1.0, 1.0, 1.0, 1.0])
    
    # Bounds
    bounds = [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0), (0.1, 3.0)]
    
    try:
        # Use Nelder-Mead for robustness and speed on low dimensions
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200, 'maxfev': 200})
        
        if res.success and time.time() < _DEADLINE - 5.0:
            return res.x
        else:
            return x0
    except Exception as e:
        print(f"Optimization error: {e}", file=sys.stderr)
        return x0

def heuristics(prizes, distance_matrix, max_len):
    """Compute the heuristic matrix for ant colony optimization."""
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple heuristic
        n = len(prizes)
        d_ij = distance_matrix + 1e-9
        d_j0 = distance_matrix[:, 0] + 1e-9
        h = prizes[None, :] / (d_ij * d_j0[None, :])
        return np.where(np.isfinite(h) & (h > 0), h, 1e-9)

    params = _optimize_parameters()
    heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)
    return heuristic
