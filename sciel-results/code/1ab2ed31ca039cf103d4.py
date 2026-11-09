import numpy as np
import time

def _run_aco(prizes, dist, max_len, heuristic_matrix, seed):
    """
    Simulates the ACO process to evaluate a heuristic matrix.
    Returns the best total prize found by any ant in the entire run.
    """
    n = len(prizes)
    np.random.seed(seed)
    pheromone = np.ones((n, n))
    best_global = 0.0
    
    for _ in range(30):
        best_iter = 0.0
        # 20 ants
        for _ant in range(20):
            current = 0
            visited = [False] * n
            visited[0] = True
            length = 0.0
            prize_sum = 0.0
            prize_sum += prizes[0] # prize 0 is 0
            
            while True:
                candidates = []
                probs = []
                for j in range(1, n): # node 0 is start, usually not revisited or not needed as dest
                    if visited[j]:
                        continue
                    if length + dist[current, j] + dist[j, 0] <= max_len:
                        candidates.append(j)
                        # Prob proportional to pheromone * heuristic
                        # heuristic entries not positive become 1e-9
                        h = heuristic_matrix[current, j]
                        if h <= 0 or not np.isfinite(h):
                            h = 1e-9
                        p = pheromone[current, j] * h
                        probs.append(p)
                
                if not candidates:
                    break
                
                # Normalize probs
                total_p = sum(probs)
                if total_p == 0:
                    break
                probs = [p / total_p for p in probs]
                
                # Select next node
                r = np.random.rand()
                cumsum = 0.0
                chosen_idx = 0
                for idx, p in enumerate(probs):
                    cumsum += p
                    if r <= cumsum:
                        chosen_idx = idx
                        break
                # Fallback if numerical issues
                next_node = candidates[chosen_idx]
                
                # Move
                length += dist[current, next_node]
                visited[next_node] = True
                prize_sum += prizes[next_node]
                current = next_node
            
        best_iter = max(best_iter, prize_sum)
        
        # Pheromone update
        pheromone = pheromone * 0.9
        pheromone += best_iter / 10.0
        
        if best_iter > best_global:
            best_global = best_iter
            
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    start_time = time.time()
    timeout = 290 # Safety margin for 300s limit
    
    # Strategy: Dynamic, budget-aware node-centric heuristic
    # Instead of a static p_j / d_ij, we estimate the "effective cost" of visiting j
    # from i as d[i,j] + d[j,0] (cost to go there and return home).
    # However, the constraint is global: length_so_far + d[i,j] + d[j,0] <= max_len.
    # A better heuristic guides the ant to high prize nodes that are "cheap" to add 
    # given the remaining budget.
    
    # Since the heuristic is static (computed once), we can't know length_so_far.
    # But we can compute a heuristic that favors nodes with high prize relative to 
    # their "return cost" d[j,0] and "entry cost" d[i,j].
    
    # The inspiration suggests a "greedy ratio" p_j / (dist[i,j] + dist[j,0]).
    # Let's enhance this by considering the distribution of prizes and distances.
    
    # 1. Compute base attractiveness: p_j / (d[i,j] + d[j,0] + epsilon)
    # 2. Scale by a factor that accounts for the overall density of prizes?
    
    # Let's try a few variants of the budget-aware ratio and select the best 
    # using a quick ACO simulation on a subset of seeds.
    
    def gen_budget_ratio(p, d, ml, alpha=1.0, beta=1.0):
        # H_ij = p_j^alpha / (d_ij + d_0j)^beta
        # d_0j is the cost to return to 0 from j
        d_0j = d[0, :]
        # Ensure no division by zero
        denom = (d + d_0j[None, :])
        denom = np.where(denom > 0, denom, 1e-9)
        num = np.power(p[None, :], alpha)
        num = np.where(num > 0, num, 1e-9)
        return num / np.power(denom, beta)
    
    def gen_distance_only(p, d, ml, alpha=1.0):
        # H_ij = p_j / d_ij
        d_safe = np.where(d > 0, d, 1e-9)
        num = np.power(p[None, :], alpha)
        num = np.where(num > 0, num, 1e-9)
        return num / d_safe
    
    def gen_return_cost_heavy(p, d, ml):
        # Heavily penalize nodes far from start, as they consume budget quickly
        d_0j = d[0, :]
        # H_ij = p_j / (d_ij * (d_0j + 1))
        denom = d * (d_0j[None, :] + 1.0)
        denom = np.where(denom > 0, denom, 1e-9)
        num = p[None, :]
        num = np.where(num > 0, num, 1e-9)
        return num / denom

    candidates = [
        ("budget_ratio_1_1", lambda p, d, ml: gen_budget_ratio(p, d, ml, 1.0, 1.0)),
        ("budget_ratio_1_2", lambda p, d, ml: gen_budget_ratio(p, d, ml, 1.0, 2.0)),
        ("budget_ratio_2_1", lambda p, d, ml: gen_budget_ratio(p, d, ml, 2.0, 1.0)),
        ("dist_only", lambda p, d, ml: gen_distance_only(p, d, ml, 1.0)),
        ("return_heavy", lambda p, d, ml: gen_return_cost_heavy(p, d, ml)),
    ]
    
    best_score = -np.inf
    best_matrix = None
    
    # Use a few seeds to estimate robustness
    test_seeds = [42, 123, 999]
    
    for name, gen in candidates:
        if time.time() - start_time > timeout:
            break
            
        try:
            h_mat = gen(prizes, distance_matrix, max_len)
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
        # Fallback to simple ratio
        d_0j = distance_matrix[0, :]
        denom = distance_matrix + d_0j[None, :]
        denom = np.where(denom > 0, denom, 1e-9)
        num = prizes[None, :]
        num = np.where(num > 0, num, 1e-9)
        best_matrix = num / denom
        best_matrix = np.where(np.isfinite(best_matrix) & (best_matrix > 0), best_matrix, 1e-9)
        
    return best_matrix
