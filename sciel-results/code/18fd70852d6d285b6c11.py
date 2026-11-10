"""Ant colony orienteering heuristic: model-and-optimize with local density."""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 310.0
_params_cache = {}

def _simulate_aco(prizes, dist, max_len, heuristic, n, rng):
    """
    Simulate the ACO process with the given heuristic and return the best prize.
    This function is used for evaluation during parameter optimization.
    """
    num_ants = 20
    num_iters = 30
    pheromone_decay = 0.9
    pheromone_gain_divisor = 10.0
    
    # Pheromone matrix
    pheromone = np.ones((n, n))
    
    # Precompute distance to home for quick feasibility checks
    dist_to_home = dist[:, 0].copy()
    
    global_best = 0.0
    
    for it in range(num_iters):
        # Check deadline
        if time.time() > _DEADLINE - 5:
            break
            
        iter_best = 0.0
        
        for ant in range(num_ants):
            cur_node = 0
            visited = [0]
            current_len = 0.0
            total_prize = 0.0
            
            # Ant moves
            while True:
                # Find feasible next nodes
                # Condition: current_len + dist[cur_node, j] + dist[j, 0] <= max_len
                # And j not in visited
                
                # Vectorized feasibility check
                # dist[cur_node, :] is the distance from current node to all others
                # dist_to_home is distance from all others to home
                # total_cost = current_len + dist[cur_node, :] + dist_to_home
                # feasible_mask = total_cost <= max_len
                # Also need to exclude visited nodes
                
                costs = current_len + dist[cur_node, :] + dist_to_home
                feasible_mask = costs <= max_len
                
                # Exclude visited
                for v in visited:
                    feasible_mask[v] = False
                
                # If no feasible nodes, stop
                if not np.any(feasible_mask):
                    break
                
                # Get indices of feasible nodes
                feasible_indices = np.where(feasible_mask)[0]
                
                # Calculate probabilities
                # prob[i] = pheromone[cur_node, i] * heuristic[cur_node, i]
                # Only for feasible nodes
                
                probs = np.zeros(n)
                for j in feasible_indices:
                    probs[j] = pheromone[cur_node, j] * heuristic[cur_node, j]
                
                # Handle case where all probs are 0 (shouldn't happen if heuristic > 0)
                if np.sum(probs) == 0:
                    break
                    
                # Normalize
                probs_sum = np.sum(probs)
                probs = probs / probs_sum
                
                # Choose next node
                next_node = rng.choice(n, p=probs)
                
                # Move
                current_len += dist[cur_node, next_node]
                total_prize += prizes[next_node]
                visited.append(next_node)
                cur_node = next_node
            
            if total_prize > iter_best:
                iter_best = total_prize
        
        if iter_best > global_best:
            global_best = iter_best
        
        # Update pheromone
        pheromone *= pheromone_decay
        pheromone += iter_best / pheromone_gain_divisor
        
    return global_best

def _make_heuristic(prizes, dist, max_len, params, n):
    """
    Construct the heuristic matrix based on parameters.
    params: [alpha, beta, gamma]
    alpha: exponent for prize/distance
    beta: weight for local density
    gamma: exponent for distance in density calculation
    """
    alpha, beta, gamma = params
    eps = 1e-9
    
    # Base heuristic: (Prize_j / Dist_ij)^alpha
    # dist is (n, n), prizes is (n,)
    # We want H[i, j] based on prizes[j] and dist[i, j]
    
    safe_dist = dist + eps
    base = (prizes[None, :] / safe_dist) ** alpha
    
    # Local Density: For each node j, sum of prizes[k] / dist[j, k]^gamma for k != j
    # We only consider k if the tour 0->...->j->k->0 is potentially feasible?
    # Approximation: Just use global proximity.
    # Density_j = sum_{k != j} (Prize_k / (Dist_jk + eps)^gamma)
    
    # Vectorized density calculation
    # dist is (n, n). We want row-wise sum excluding diagonal.
    # inv_dist = 1 / (dist + eps)^gamma
    inv_dist = 1.0 / (safe_dist ** gamma)
    # Set diagonal to 0 to exclude self
    np.fill_diagonal(inv_dist, 0.0)
    # Multiply by prizes of target nodes (columns)
    # density_matrix[j, k] = inv_dist[j, k] * prizes[k]
    weighted_inv_dist = inv_dist * prizes[None, :]
    density = np.sum(weighted_inv_dist, axis=1) # Shape (n,)
    
    # Heuristic H[i, j] = base[i, j] * (1 + beta * density[j])
    # density is per node j, so we expand it to (1, n)
    density_factor = 1.0 + beta * density[None, :]
    
    heur = base * density_factor
    
    # Ensure positivity and finiteness
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    return heur

def _optimize_parameters(prizes, dist, max_len, n):
    """
    Find good parameters for the heuristic using local search.
    """
    # Initial guess
    x0 = np.array([1.0, 0.5, 1.0])
    
    def objective(params):
        if time.time() > _DEADLINE - 10:
            return 0.0
        try:
            heur = _make_heuristic(prizes, dist, max_len, params, n)
            rng = np.random.default_rng(42) # Fixed seed for consistency during optimization
            score = _simulate_aco(prizes, dist, max_len, heur, n, rng)
            return -score
        except:
            return 1e9

    # Use Nelder-Mead for a few iterations
    # Limit iterations to keep it fast
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 0.1, 'fatol': 0.1})
    
    # Also try a few random restarts if time permits
    best_params = res.x
    best_score = -res.fun
    
    # Simple random search to see if we can beat Nelder-Mead quickly
    rng = np.random.default_rng(123)
    for _ in range(10):
        if time.time() > _DEADLINE - 20:
            break
        x_try = np.array([rng.uniform(0.5, 2.0), rng.uniform(0.0, 2.0), rng.uniform(0.5, 2.0)])
        score = -objective(x_try)
        if score > best_score:
            best_score = score
            best_params = x_try
            
    return best_params

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for Ant Colony Optimization on the
    Orienteering Problem using a model-and-optimize approach with local density.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)

    n = len(prizes)

    # Check cache
    cache_key = (n, tuple(prizes.tobytes()), tuple(distance_matrix.tobytes()), max_len)
    if cache_key in _params_cache:
        return _params_cache[cache_key]
    
    # Check deadline
    if time.time() > _DEADLINE - 30:
        # Fallback to a simple heuristic if we are out of time
        eps = 1e-9
        safe_dist = distance_matrix + eps
        heur = prizes[None, :] / safe_dist
        heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
        _params_cache[cache_key] = heur
        return heur

    # Optimize parameters
    params = _optimize_parameters(prizes, distance_matrix, max_len, n)
    
    # Generate final heuristic
    heur = _make_heuristic(prizes, distance_matrix, max_len, params, n)
    
    # Cache
    _params_cache[cache_key] = heur
    
    return heur
