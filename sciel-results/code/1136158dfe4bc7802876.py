"""Ant colony orienteering heuristic: regret-based flexibility."""
import numpy as np
import time

_DEADLINE = time.time() + 310.0  # 20s safety margin before 330s kill
_params_cache = {}


def _compute_regret_matrix(prizes, dist, max_len):
    """
    Compute a heuristic matrix based on prize-to-cost ratio and flexibility.
    Flexibility is the number of nodes reachable from j within the budget remaining
    after visiting j and returning home.
    """
    n = len(prizes)
    eps = 1e-9
    
    # Precompute distance to home
    d_to_home = dist[:, 0].copy()
    
    # Base heuristic: Prize / Distance
    # We use dist[i, j] for the cost of moving i->j
    # We want H[i, j] to be high if prizes[j] is high and dist[i, j] is low.
    
    # Vectorized base calculation
    # dist is (n, n), prizes is (n,)
    # We want (n, n) matrix where H[i, j] ~ prizes[j] / dist[i, j]
    # To avoid division by zero, add eps
    safe_dist = dist + eps
    base_heur = prizes[None, :] / safe_dist  # (n, n)
    
    # Compute Flexibility for each node j
    # Flexibility[j] = count of k such that dist[j, k] + dist[k, 0] <= remaining_budget_j
    # remaining_budget_j = max_len - (dist[0, j] + dist[j, 0])
    
    flexibility = np.zeros(n)
    for j in range(n):
        min_cost = dist[0, j] + dist[j, 0]
        remaining = max_len - min_cost
        
        if remaining < 0:
            # Node j cannot even be visited and returned from
            flexibility[j] = 0.0
            continue
            
        # Find all k such that dist[j, k] + dist[k, 0] <= remaining
        # dist[j, :] is (n,), d_to_home is (n,)
        costs = dist[j, :] + d_to_home
        feasible_count = np.sum(costs <= remaining + 1e-9)
        flexibility[j] = feasible_count
        
    # Construct final heuristic
    # We multiply base_heur by a factor that rewards flexibility.
    # If flexibility is 0, it's a dead end. We want to penalize it heavily.
    # If flexibility is high, we want to reward it.
    # Let's use: Factor = 1.0 + flexibility
    # Or maybe exponential? Factor = np.exp(flexibility * 0.1)?
    # Let's stick to linear for stability: Factor = 1.0 + flexibility
    
    flex_factor = 1.0 + flexibility[None, :]  # (1, n)
    
    heur = base_heur * flex_factor
    
    # Ensure positive and finite
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    return heur


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for Ant Colony Optimization on the
    Orienteering Problem using a regret-based flexibility model.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)

    n = len(prizes)

    # Check cache
    cache_key = (n, tuple(prizes.tobytes()), tuple(distance_matrix.tobytes()), max_len)
    if cache_key in _params_cache:
        return _params_cache[cache_key]
    
    # Compute the heuristic
    eta = _compute_regret_matrix(prizes, distance_matrix, max_len)
    
    # Cache the result
    _params_cache[cache_key] = eta
    
    return eta
