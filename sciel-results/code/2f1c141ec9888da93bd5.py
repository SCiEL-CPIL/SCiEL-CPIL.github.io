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


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for ACO orienteering.
    
    We use a greedy-first heuristic that strongly favors nodes with high prize
    and low distance-to-home, with a distance penalty tuned to encourage
    short tours. We also try a few parameter variants and pick the best.
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
    
    # Try several heuristic forms and pick the best via simulation
    # Form 1: prize * exp(-d_j0 / scale) / (d + eps)^alpha
    # Form 2: (prize / (d + eps)) * exp(-d_j0 / scale)
    # Form 3: prize^beta / (d + eps)^alpha * exp(-d_j0 / scale)
    
    best_score = -1.0
    best_h = None
    
    # Quick candidate list
    candidates = []
    
    # Alpha values for distance penalty
    alphas = [0.5, 1.0, 1.5, 2.0]
    # Scale values for exp decay
    scales = [0.3, 0.5, 0.7, 1.0, 1.5]
    # Beta values for prize exponent
    betas = [1.0, 1.5, 2.0]
    
    # Generate candidate parameter sets
    for alpha in alphas:
        for scale in scales:
            for beta in betas:
                candidates.append((alpha, scale, beta))
    
    # Shuffle candidates to spread risk
    np.random.seed(42)
    indices = np.random.permutation(len(candidates))
    
    seeds_to_test = [42, 123]
    
    for idx in indices:
        if time.time() - start_time > time_limit:
            break
        
        alpha, scale, beta = candidates[idx]
        
        # Compute heuristic
        prize_pow = np.maximum(prizes, 1e-9) ** beta
        d_plus = distance_matrix + eps
        node_affinity = prize_pow * np.exp(-d_j0 / scale)
        
        # h[i][j] = node_affinity[j] / d[i][j]^alpha
        h = node_affinity[None, :] / (d_plus ** alpha)
        h = np.maximum(h, 1e-9)
        
        # Test with one or two seeds
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
        d_plus = distance_matrix + eps
        node_affinity = prizes * np.exp(-d_j0 / 0.5)
        best_h = node_affinity[None, :] / d_plus
        best_h = np.maximum(best_h, 1e-9)
    
    return best_h
# EVOLVE-BLOCK-END
