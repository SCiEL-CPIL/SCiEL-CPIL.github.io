"""Regret-based Heuristic ACO Orienteering."""
import numpy as np
import time
import random

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 320.0  # Safety margin before 330s kill switch
    
    N = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    d_j0 = distance_matrix[:, 0].copy()
    
    def compute_regret_heuristic(params):
        """
        Compute regret-based heuristic.
        params: [alpha, beta, gamma]
        Regret[i,j] = P_j * (1 - exp(-alpha * P_j / (D_ij^beta + D_j0^gamma + eps)))
        """
        alpha, beta, gamma = params
        eps = 1e-9
        
        P = np.maximum(prizes, eps)
        D = np.maximum(distance_matrix, eps)
        Dj0 = np.maximum(d_j0, eps)
        
        # cost_ij = D_ij^beta + Dj0^gamma
        # For each i, j: cost = D[i,j]^beta + Dj0[j]^gamma
        cost = np.power(D, beta) + np.power(Dj0[None, :], gamma)
        
        # regret[i,j] = P_j * (1 - exp(-alpha * P_j / cost_ij))
        # Note: P_j is shape (N,), cost is (N,N)
        # P_j[None, :] is (1, N)
        ratio = alpha * P[None, :] / cost
        regret = P[None, :] * (1.0 - np.exp(-ratio))
        
        # Set node 0 column to 0 (can't go back to start in a meaningful way for heuristic)
        regret[:, 0] = 0.0
        # Set node 0 row to 0 (start node)
        regret[0, :] = 0.0
        
        # Replace non-positive with 1e-9
        regret = np.where(regret > 0, regret, 1e-9)
        
        return regret

    def evaluate_heuristic(params, seed):
        """
        Run ACO with a specific heuristic and return best prize.
        """
        np.random.seed(seed)
        
        eta = compute_regret_heuristic(params)
        
        # ACO Simulation
        tau = np.ones((N, N))
        best_prize_global = 0.0
        
        for _ in range(30):
            best_prize_iter = 0.0
            
            for _ in range(20):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                
                while True:
                    costs = distance_matrix[current, :] + d_j0
                    remaining = max_len - path_len
                    mask = (~visited) & (costs <= remaining)
                    
                    if not np.any(mask):
                        break
                    
                    eligible_idx = np.where(mask)[0]
                    
                    weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                    total_w = weights.sum()
                    
                    if total_w <= 0:
                        break
                    
                    probs = weights / total_w
                    r = np.random.rand()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible_idx):
                        idx = len(eligible_idx) - 1
                    
                    next_node = eligible_idx[idx]
                    path_len += distance_matrix[current, next_node]
                    prize_total += prizes[next_node]
                    visited[next_node] = True
                    current = next_node
                
                if prize_total > best_prize_iter:
                    best_prize_iter = prize_total
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global

    def multi_seed_eval(params, seeds):
        """Evaluate params across multiple seeds to reduce variance."""
        scores = []
        for s in seeds:
            score = evaluate_heuristic(params, s)
            scores.append(score)
        return np.mean(scores)

    # Parameter search
    # alpha: scale of regret (higher = more selective)
    # beta: distance exponent
    # gamma: return distance exponent
    
    initial_x0 = [1.0, 1.0, 1.0]
    
    # Phase 1: Grid Search
    grid_alpha = [0.5, 1.0, 2.0, 3.0, 5.0]
    grid_beta = [0.5, 1.0, 1.5, 2.0]
    grid_gamma = [0.0, 0.5, 1.0, 1.5]
    
    candidates = []
    for a in grid_alpha:
        for b in grid_beta:
            for g in grid_gamma:
                candidates.append([a, b, g])
    
    best_score_grid = -1
    best_params_grid = initial_x0
    
    for p in candidates:
        if time.time() - start_time > time_limit - 60:
            break
        score = evaluate_heuristic(p, 0)
        if score > best_score_grid:
            best_score_grid = score
            best_params_grid = p
    
    # Phase 2: Local Optimization
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy:
        def objective(params):
            if time.time() - start_time > time_limit - 10:
                return 0.0
            return -multi_seed_eval(params, [0, 1])
        
        x0 = best_params_grid
        bounds = [(0.1, 10.0), (0.1, 5.0), (0.0, 5.0)]
        
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={'maxiter': 50, 'xatol': 0.05, 'fatol': 0.01}
        )
        
        if time.time() - start_time < time_limit - 20:
            best_params_local = res.x
            best_score_local = -res.fun
        else:
            best_params_local = x0
            best_score_local = -objective(x0)
    else:
        best_params_local = best_params_grid
        best_score_local = best_score_grid
        
        for _ in range(50):
            if time.time() - start_time > time_limit - 20:
                break
            new_p = [x + random.uniform(-0.2, 0.2) for x in best_params_local]
            new_p[0] = np.clip(new_p[0], 0.1, 10.0)
            new_p[1] = np.clip(new_p[1], 0.1, 5.0)
            new_p[2] = np.clip(new_p[2], 0.0, 5.0)
            
            s = evaluate_heuristic(new_p, 0)
            if s > best_score_local:
                best_params_local = new_p
                best_score_local = s

    # Phase 3: Final Selection
    final_candidates = [
        (best_params_grid, "grid"),
        (best_params_local, "local"),
        (initial_x0, "init")
    ]
    
    best_final_params = best_params_local
    best_final_score = -np.inf
    
    for p, name in final_candidates:
        if time.time() - start_time > time_limit - 5:
            break
        score = multi_seed_eval(p, [0, 1, 2])
        if score > best_final_score:
            best_final_score = score
            best_final_params = p
    
    # Construct final heuristic
    eta = compute_regret_heuristic(best_final_params)
    
    # Safety check
    if not np.all(np.isfinite(eta)) or eta.shape != distance_matrix.shape:
        # Fallback
        eps = 1e-9
        eta = (prizes[None, :] + eps) / (distance_matrix + eps)
        eta = np.where(eta > 0, eta, 1e-9)

    return eta

# EVOLVE-BLOCK-END
