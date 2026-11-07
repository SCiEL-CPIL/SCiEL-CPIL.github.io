"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# ACO parameters (fixed by contract)
N_ANTS = 20
N_ITERS = 30
RHO = 0.9  # evaporation
Q = 10.0   # pheromone deposit divisor


def _run_aco(prizes, dist, max_len, heuristic):
    """Run the ACO and return the best total prize collected by any ant."""
    n = len(prizes)
    pheromone = np.ones((n, n), dtype=float)
    
    best_total = 0.0
    
    for _ in range(N_ITERS):
        iter_best = 0.0
        
        for _ in range(N_ANTS):
            visited = [0]
            current = 0
            total_len = 0.0
            total_prize = prizes[0]  # prize at node 0 is 0
            unvisited = set(range(1, n))
            
            while True:
                # Find candidate nodes
                candidates = []
                for j in unvisited:
                    d_ij = dist[current, j]
                    d_j0 = dist[j, 0]
                    if total_len + d_ij + d_j0 <= max_len:
                        candidates.append(j)
                
                if not candidates:
                    break
                
                # Compute probabilities proportional to pheromone * heuristic
                probs = np.zeros(len(candidates))
                for idx, j in enumerate(candidates):
                    probs[idx] = pheromone[current, j] * heuristic[current, j]
                
                # Ensure positive
                probs = np.maximum(probs, 1e-9)
                probs = probs / probs.sum()
                
                # Sample next node
                r = np.random.random()
                cumsum = np.cumsum(probs)
                chosen_idx = np.searchsorted(cumsum, r)
                if chosen_idx >= len(candidates):
                    chosen_idx = len(candidates) - 1
                
                j = candidates[chosen_idx]
                total_len += dist[current, j]
                total_prize += prizes[j]
                visited.append(j)
                unvisited.remove(j)
                current = j
            
        # After all ants, update pheromone
        pheromone *= RHO
        pheromone += iter_best / Q
        
        if iter_best > best_total:
            best_total = iter_best
    
    return best_total


def _evaluate_heuristic(prizes, dist, max_len, heuristic):
    """Evaluate a heuristic by running ACO and returning the best prize."""
    np.random.seed(42)  # Fixed seed for consistent evaluation
    return _run_aco(prizes, dist, max_len, heuristic)


def _optimize_heuristic(prizes, dist, max_len, time_limit):
    """
    Optimize the heuristic parameters using a simple hill-climbing approach.
    We parameterize the heuristic as:
        h[i, j] = prize[j] / (d[i,j] + alpha * d[j,0] + eps)
    and optimize alpha.
    """
    n = len(prizes)
    dist_to_start = dist[:, 0]
    
    # Initial alpha from the seed (effectively 0)
    best_alpha = 0.0
    best_score = -1.0
    
    # Try a range of alphas
    alphas = np.linspace(0.0, 1.5, 16)
    
    for alpha in alphas:
        heuristic = prizes[None, :] / (dist + alpha * dist_to_start[None, :] + 1e-9)
        heuristic = np.where(heuristic > 0, heuristic, 1e-9)
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        
        score = _evaluate_heuristic(prizes, dist, max_len, heuristic)
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
        
        # Check time
        if time.time() > time_limit:
            break
    
    # Refine around the best alpha
    if best_alpha > 0:
        local_alphas = np.linspace(max(0, best_alpha - 0.2), min(1.5, best_alpha + 0.2), 8)
    else:
        local_alphas = np.linspace(0, 0.4, 8)
    
    for alpha in local_alphas:
        heuristic = prizes[None, :] / (dist + alpha * dist_to_start[None, :] + 1e-9)
        heuristic = np.where(heuristic > 0, heuristic, 1e-9)
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        
        score = _evaluate_heuristic(prizes, dist, max_len, heuristic)
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
        
        # Check time
        if time.time() > time_limit:
            break
    
    # Final heuristic with the best alpha
    heuristic = prizes[None, :] / (dist + best_alpha * dist_to_start[None, :] + 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Uses meta-optimization to find the best heuristic parameter for the instance.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Ensure prize at node 0 is 0
    prizes = prizes.copy()
    prizes[0] = 0.0
    
    # Deadline guard: leave 5 seconds margin before the 330s limit
    start_time = time.time()
    time_limit = start_time + 320.0
    
    # If we have plenty of time, do meta-optimization
    if time.time() < time_limit - 10.0:
        heuristic = _optimize_heuristic(prizes, distance_matrix, max_len, time_limit)
    else:
        # Fallback to a reasonable heuristic
        dist_to_start = distance_matrix[:, 0]
        heuristic = prizes[None, :] / (distance_matrix + 0.5 * dist_to_start[None, :] + 1e-9)
        heuristic = np.where(heuristic > 0, heuristic, 1e-9)
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Ensure correct shape
    if heuristic.shape != (n, n):
        heuristic = np.ones((n, n), dtype=float) * 1e-9
    
    # Ensure all values are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
