import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290.0 # 10s margin
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # If time is already up (unlikely, but safe)
    if time.time() > deadline:
         return np.ones((n, n)) * 1e-9

    # Define greedy evaluator
    def get_prize(eta):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        curr = 0
        total_len = 0.0
        total_prize = 0.0
        
        # We need to find the best next node
        for _ in range(n):
            best_val = -1.0
            best_j = -1
            for j in range(1, n):
                if visited[j]:
                    continue
                # Constraint check
                new_len = total_len + distance_matrix[curr, j] + distance_matrix[j, 0]
                if new_len <= max_len:
                    val = eta[curr, j]
                    if val > best_val:
                        best_val = val
                        best_j = j
            
            if best_j == -1:
                break
                
            total_len += distance_matrix[curr, best_j]
            total_prize += prizes[best_j]
            visited[best_j] = True
            curr = best_j
            
        return total_prize

    # Helper to compute heuristic for given params
    def compute_heuristic(a, b, g, use_slack=False):
        eps = 1e-9
        dist_to_origin = distance_matrix[:, 0]
        prize_term = (prizes + eps) ** a
        dist_ij_term = (distance_matrix + eps) ** b
        dist_j0_term = (dist_to_origin + eps) ** g
        eta = prize_term[None, :] / (dist_ij_term * dist_j0_term[None, :])
        
        if use_slack:
            # Add slack-awareness: penalize edges that leave little remaining budget
            # slack[i,j] = max_len - (d[i,j] + d[j,0]) - current_len, but we don't have current_len in heuristic
            # Instead, use a global slack factor: edges with larger d[i,j]+d[j,0] get slightly penalized
            # This is a proxy for "marginal gain" - prefer edges that leave more room
            total_cost = distance_matrix + dist_to_origin[None, :]
            # Normalize cost to [0,1]
            max_cost = np.max(total_cost)
            if max_cost > 0:
                slack_factor = 1.0 - 0.3 * (total_cost / max_cost)
                eta = eta * slack_factor
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    # Parameter Grid - expanded
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    betas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    gammas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    best_prize = -1.0
    best_params = (3.0, 1.5, 1.0)
    best_use_slack = False
    
    eps = 1e-9
    dist_to_origin = distance_matrix[:, 0]
    
    # Coarse grid search
    for a in alphas:
        for b in betas:
            for g in gammas:
                for use_slack in [False, True]:
                    eta = compute_heuristic(a, b, g, use_slack)
                    p = get_prize(eta)
                    if p > best_prize:
                        best_prize = p
                        best_params = (a, b, g)
                        best_use_slack = use_slack
                    
                    if time.time() > deadline - 30:
                        break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break
    
    # Local refinement around best params
    a_best, b_best, g_best = best_params
    refinement_steps = [0.1, 0.05, 0.02]
    
    for step in refinement_steps:
        a_neighbors = [a_best - step, a_best, a_best + step]
        b_neighbors = [b_best - step, b_best, b_best + step]
        g_neighbors = [g_best - step, g_best, g_best + step]
        
        for a in a_neighbors:
            if a <= 0:
                continue
            for b in b_neighbors:
                if b <= 0:
                    continue
                for g in g_neighbors:
                    if g <= 0:
                        continue
                    for use_slack in [best_use_slack, not best_use_slack]:
                        eta = compute_heuristic(a, b, g, use_slack)
                        p = get_prize(eta)
                        if p > best_prize:
                            best_prize = p
                            best_params = (a, b, g)
                            best_use_slack = use_slack
                            a_best, b_best, g_best = a, b, g
                    
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break
    
    a, b, g = best_params
    heuristic = compute_heuristic(a, b, g, best_use_slack)
    
    return heuristic
