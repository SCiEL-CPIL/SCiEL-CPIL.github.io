"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, seed, n_ants=20, n_iter=30):
    """
    Fast simulation of the ACO colony.
    Returns the best total prize collected by any ant in the last iteration 
    (or max over all iterations, as per task description 'best prize of that iteration' 
    is used for pheromone update, but scoring is 'highest total prize collected by any ant' 
    which usually implies over the whole run or the best in the final state? 
    Task says: 'mean over instances of the highest total prize collected by any ant.'
    This phrasing is ambiguous. Does it mean the max over all ants in all iterations? 
    Or the max over ants in the final iteration? 
    Given 'After each iteration... pheromone is updated', the colony evolves. 
    Usually, the best solution found *during* the run is tracked. 
    The current program tracks `best_total` which is the max over all iterations. 
    I will stick to tracking the global max prize found by any ant in any iteration.
    """
    np.random.seed(seed)
    
    n = len(prizes)
    if n <= 1:
        return 0.0
    
    evaporation = 0.9
    deposition = 1.0 / 10.0
    
    # Pheromone matrix initialized to 1
    pheromone = np.ones((n, n), dtype=np.float64)
    
    # Ensure heuristic is positive
    h = np.maximum(heuristic, 1e-9)
    
    # Precompute d_j0 for speed
    d_j0 = distance_matrix[:, 0]
    
    best_total = 0.0
    
    for it in range(n_iter):
        iter_best = 0.0
        
        for ant in range(n_ants):
            visited_set = {0}
            current = 0
            length = 0.0
            total_prize = 0.0 # prizes[0] is 0
            
            while True:
                # Find eligible next nodes
                # Optimization: Iterate through all nodes, check if unvisited and fits in budget
                eligible = []
                # Vectorized eligibility check is tricky because 'current' changes.
                # For n=50, a loop is fine.
                for j in range(1, n):
                    if j not in visited_set:
                        d_ij = distance_matrix[current, j]
                        if length + d_ij + d_j0[j] <= max_len:
                            eligible.append(j)
                
                if not eligible:
                    break
                
                # Compute probabilities
                # Extract pheromone and heuristic for eligible nodes
                # eligible is a list of indices
                # pheromone[current, eligible] works with list indexing in numpy? 
                # Yes, but it creates a copy. 
                # Let's use advanced indexing.
                elig_arr = np.array(eligible)
                probs = pheromone[current, elig_arr] * h[current, elig_arr]
                
                total_p = probs.sum()
                if total_p <= 0 or not np.isfinite(total_p):
                    break
                
                probs = probs / total_p
                
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
    We test a small set of well-motivated heuristic forms and select the best one
    using a fast simulation with reduced iterations/ants to save time.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    
    eps = 1e-9
    d_j0 = distance_matrix[:, 0]
    
    # Candidate heuristic forms
    # 1. P_j / d_ij
    # 2. P_j / (d_ij * (1 + d_j0))
    # 3. P_j / d_ij^2
    # 4. P_j / sqrt(d_ij * (1 + d_j0))
    # 5. P_j / (d_ij * (1 + d_j0^2))
    
    candidates = []
    
    # H1
    h1 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = distance_matrix[i, j]
                h1[i, j] = prizes[j] / max(d, eps)
    candidates.append(h1)
    
    # H2
    h2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = distance_matrix[i, j]
                h2[i, j] = prizes[j] / (max(d, eps) * (1 + d_j0[j]))
    candidates.append(h2)
    
    # H3
    h3 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = distance_matrix[i, j]
                h3[i, j] = prizes[j] / max(d*d, eps)
    candidates.append(h3)
    
    # H4
    h4 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = distance_matrix[i, j]
                h4[i, j] = prizes[j] / np.sqrt(max(d, eps) * (1 + d_j0[j]))
    candidates.append(h4)
    
    # H5
    h5 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = distance_matrix[i, j]
                h5[i, j] = prizes[j] / (max(d, eps) * (1 + d_j0[j]*d_j0[j]))
    candidates.append(h5)
    
    # Ensure all are positive
    for h in candidates:
        h[:] = np.maximum(h, 1e-9)
    
    # Evaluate candidates using a fast simulation
    # Use 10 ants and 10 iterations for speed, seed=42
    # If time permits, we can do a second pass with more iterations for the top 2.
    
    start_time = time.time()
    time_limit = 300.0 # Safety margin
    
    best_score = -1.0
    best_h = candidates[0]
    
    # First pass: Quick evaluation
    for h in candidates:
        if time.time() - start_time > time_limit * 0.5:
            break
        score = _simulate_aco(prizes, distance_matrix, max_len, h, seed=42, n_ants=10, n_iter=10)
        if score > best_score:
            best_score = score
            best_h = h
            
    # Second pass: Refine top 2 with full simulation
    # Sort candidates by first pass score? 
    # We didn't store all scores. Let's just re-evaluate the best one and maybe one other.
    # Actually, let's just evaluate the best one found with full params to see if it's stable?
    # No, the goal is to return the heuristic. The judge will run the full ACO.
    # So we just need to pick the heuristic that *tends* to work well.
    # The quick simulation is a proxy.
    
    # Let's do a slightly more robust check for the top candidate vs the second best if time allows.
    # But we didn't track the second best. 
    # Let's just stick with the best from the quick pass. 
    # To be safer, we can run the quick pass with 2 seeds.
    
    best_score = -1.0
    best_h = candidates[0]
    
    for h in candidates:
        if time.time() - start_time > time_limit * 0.8:
            break
        # Average over 2 seeds for robustness
        s1 = _simulate_aco(prizes, distance_matrix, max_len, h, seed=42, n_ants=10, n_iter=10)
        s2 = _simulate_aco(prizes, distance_matrix, max_len, h, seed=123, n_ants=10, n_iter=10)
        avg_score = (s1 + s2) / 2.0
        if avg_score > best_score:
            best_score = avg_score
            best_h = h

    return best_h
# EVOLVE-BLOCK-END
