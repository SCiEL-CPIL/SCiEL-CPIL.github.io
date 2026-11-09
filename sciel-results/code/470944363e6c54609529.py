import numpy as np
import time

def _run_aco(prizes, dist, max_len, heuristic_matrix, seed):
    """
    Simulates the ACO process to evaluate a heuristic matrix.
    Returns the best total prize found by any ant in any iteration.
    """
    n = len(prizes)
    np.random.seed(seed)
    pheromone = np.ones((n, n))
    best_global = 0.0
    
    # Precompute d_j0 for faster access
    d_j0 = dist[:, 0]
    
    # Pre-sanitize heuristic: non-positive or non-finite become 1e-9
    # We do this outside the loop for speed
    h_safe = np.where(np.isfinite(heuristic_matrix) & (heuristic_matrix > 0), heuristic_matrix, 1e-9)
    
    for _iter in range(30):
        best_iter = 0.0
        
        # 20 ants
        for _ant in range(20):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize_sum = prizes[0]  # 0
            
            while True:
                curr = current
                curr_d = dist[curr]
                curr_p = pheromone[curr]
                curr_h = h_safe[curr]
                
                # Vectorized candidate check
                unvisited_mask = ~visited
                unvisited_mask[0] = False  # node 0 already visited
                
                if not unvisited_mask.any():
                    break
                
                # Constraint: length + d[curr,j] + d_j0[j] <= max_len
                remaining_budget = max_len - length
                feasible = unvisited_mask & (curr_d + d_j0 <= remaining_budget)
                
                if not feasible.any():
                    break
                
                # Compute probabilities: pheromone * heuristic for feasible nodes
                prob_vec = curr_p * curr_h
                prob_vec = np.where(feasible, prob_vec, 0.0)
                
                total_p = prob_vec.sum()
                if total_p <= 0:
                    break
                
                # Normalize and sample
                prob_norm = prob_vec / total_p
                feasible_indices = np.where(feasible)[0]
                feasible_probs = prob_norm[feasible_indices]
                fp_sum = feasible_probs.sum()
                if fp_sum <= 0:
                    break
                feasible_probs = feasible_probs / fp_sum
                
                next_node = np.random.choice(feasible_indices, p=feasible_probs)
                
                # Move
                length += dist[curr, next_node]
                visited[next_node] = True
                prize_sum += prizes[next_node]
                current = next_node
            
        best_iter = max(best_iter, prize_sum)
        
        # Pheromone update
        pheromone *= 0.9
        pheromone += best_iter / 10.0
        
        if best_iter > best_global:
            best_global = best_iter
            
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    start_time = time.time()
    timeout = 280 # Safety margin
    
    p = prizes
    d = distance_matrix
    ml = max_len
    d_j0 = d[:, 0]
    
    # Parametric Grid Search
    # Family 1: Product p^gamma / (d_ij^alpha * d_j0^beta)
    # Family 2: Sum p^gamma / (d_ij + d_j0)^alpha
    
    candidates = []
    
    # Alphas and Gammas
    alphas = [0.5, 1.0, 1.5, 2.0]
    gammas = [0.5, 1.0, 1.5]
    betas = [0.0, 0.5, 1.0] # 0.0 means no d_j0 penalty in product
    
    for gamma in gammas:
        for alpha in alphas:
            # Family 1: Product
            for beta in betas:
                def make_prod(g, a, b):
                    def gen(p, d, d_j0, ml):
                        num = np.power(p[None, :] + 1e-9, g)
                        den = np.power(d + 1e-9, a)
                        if b > 0:
                            den = den * np.power(d_j0[None, :] + 1e-9, b)
                        return num / (den + 1e-9)
                    return gen
                candidates.append(make_prod(gamma, alpha, beta))
            
            # Family 2: Sum
            def make_sum(g, a):
                def gen(p, d, d_j0, ml):
                    num = np.power(p[None, :] + 1e-9, g)
                    den = np.power(d + d_j0[None, :] + 1e-9, a)
                    return num / (den + 1e-9)
                return gen
            candidates.append(make_sum(gamma, alpha))
            
    # Add a few "classic" ones that might be missed or edge cases
    # e.g. p / d
    def gen_classic_1(p, d, d_j0, ml):
        return (p[None, :] + 1e-9) / (d + 1e-9)
    candidates.append(gen_classic_1)
    
    # p / (d + d_j0)
    def gen_classic_2(p, d, d_j0, ml):
        return (p[None, :] + 1e-9) / (d + d_j0[None, :] + 1e-9)
    candidates.append(gen_classic_2)

    best_score = -np.inf
    best_matrix = None
    
    # Use 10 seeds for robustness
    test_seeds = [42, 123, 999, 7, 314, 555, 2024, 1, 100, 200]
    
    for gen in candidates:
        if time.time() - start_time > timeout:
            break
            
        try:
            h_mat = gen(prizes, distance_matrix, d_j0, max_len)
            # Sanitize
            h_mat = np.where(np.isfinite(h_mat) & (h_mat > 0), h_mat, 1e-9)
            
            scores = []
            for seed in test_seeds:
                if time.time() - start_time > timeout:
                    break
                score = _run_aco(prizes, distance_matrix, max_len, h_mat, seed)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_matrix = h_mat
        except Exception:
            continue
            
    if best_matrix is None:
        # Fallback
        best_matrix = gen_classic_1(prizes, distance_matrix, d_j0, max_len)
        best_matrix = np.where(np.isfinite(best_matrix) & (best_matrix > 0), best_matrix, 1e-9)
        
    return best_matrix
