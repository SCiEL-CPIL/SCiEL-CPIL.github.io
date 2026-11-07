import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 295.0 # 5s margin

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)

    if time.time() > deadline:
        return np.ones((n, n)) * 1e-9

    def get_prize(eta):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        curr = 0
        total_len = 0.0
        total_prize = 0.0
        
        for _ in range(n):
            best_val = -1.0
            best_j = -1
            for j in range(1, n):
                if visited[j]:
                    continue
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

    eps = 1e-9
    dist_to_origin = distance_matrix[:, 0]
    
    def evaluate_params(a, b, g):
        prize_term = (prizes + eps) ** a
        dist_ij_term = (distance_matrix + eps) ** b
        dist_j0_term = (dist_to_origin + eps) ** g
        eta = prize_term[None, :] / (dist_ij_term * dist_j0_term[None, :])
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return get_prize(eta)

    best_prize = -1.0
    best_params = (2.0, 0.5, 0.5)
    
    # Phase 1: Coarse Grid
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    betas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    gammas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    for a in alphas:
        prize_term = (prizes + eps) ** a
        for b in betas:
            dist_ij_term = (distance_matrix + eps) ** b
            for g in gammas:
                dist_j0_term = (dist_to_origin + eps) ** g
                eta = prize_term[None, :] / (dist_ij_term * dist_j0_term[None, :])
                eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
                
                p = get_prize(eta)
                if p > best_prize:
                    best_prize = p
                    best_params = (a, b, g)
                    
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break

    # Phase 2: Local Refinement
    if time.time() < deadline - 5:
        a0, b0, g0 = best_params
        steps = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        for step in steps:
            improved = False
            offsets = [-step, -step/2, 0, step/2, step]
            for da in offsets:
                for db in offsets:
                    for dg in offsets:
                        if da == 0 and db == 0 and dg == 0:
                            continue
                        a = a0 + da
                        b = b0 + db
                        g = g0 + dg
                        
                        if a < 0.1 or b < 0.0 or g < 0.0:
                            continue
                        if a > 15 or b > 5 or g > 5:
                            continue
                        
                        if time.time() > deadline - 2:
                            break
                            
                        p = evaluate_params(a, b, g)
                        if p > best_prize:
                            best_prize = p
                            best_params = (a, b, g)
                            a0, b0, g0 = a, b, g
                            improved = True
                    
                    if time.time() > deadline - 2:
                        break
                if time.time() > deadline - 2:
                    break
            
            if not improved:
                if step == steps[-1]:
                    break

    a, b, g = best_params
    prize_term = (prizes + eps) ** a
    dist_ij_term = (distance_matrix + eps) ** b
    dist_j0_term = (dist_to_origin + eps) ** g
    heuristic = prize_term[None, :] / (dist_ij_term * dist_j0_term[None, :])
    
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
