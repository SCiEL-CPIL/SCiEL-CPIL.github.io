import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    n = len(prizes)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    rng = np.random.default_rng(seed)
    
    for it in range(num_iters):
        best_iter_prize = 0
        
        for _ in range(num_ants):
            current_node = 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            path_len = 0.0
            total_prize = prizes[0]
            
            while True:
                remaining_budget = max_len - path_len
                valid_indices = np.where(
                    (~visited_mask) & 
                    (distance_matrix[current_node, :] + distance_matrix[:, 0] <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                h_vals = H[current_node, valid_indices]
                total_h = np.sum(h_vals)
                if total_h == 0:
                    break
                    
                probs = h_vals / total_h
                next_node = valid_indices[rng.choice(len(valid_indices), p=probs)]
                
                path_len += distance_matrix[current_node, next_node]
                total_prize += prizes[next_node]
                visited_mask[next_node] = True
                current_node = next_node
            
        best_iter_prize = max(best_iter_prize, total_prize)
        best_prize = max(best_prize, best_iter_prize)

    return best_prize

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290
    
    n = len(prizes)
    eps = 1e-9
    
    def get_heuristic(params):
        alpha, beta, gamma = params
        if alpha < 0: alpha = 0
        if beta < 1e-4: beta = 1e-4
        if gamma < 0.1: gamma = 0.1
        
        d = distance_matrix + beta
        p = prizes[None, :] + alpha
        d = np.maximum(d, eps)
        p = np.maximum(p, eps)
        
        h = p / (d ** gamma)
        h = np.where(h > 0, h, eps)
        return h

    def objective(params):
        if time.time() > deadline:
            return 1e9
        h = get_heuristic(params)
        score = _simulate_aco(prizes, distance_matrix, max_len, h)
        return -score

    best_score = -1
    best_params = None
    best_h = None
    
    alpha_grid = [0.0, 0.1, 0.2, 0.5]
    beta_grid = [0.01, 0.05, 0.1, 0.2, 0.5]
    gamma_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    candidates = []
    
    for alpha in alpha_grid:
        for beta in beta_grid:
            for gamma in gamma_grid:
                if time.time() > deadline:
                    break
                params = (alpha, beta, gamma)
                h = get_heuristic(params)
                score = _simulate_aco(prizes, distance_matrix, max_len, h)
                candidates.append((score, params))
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_h = h
        
        if time.time() > deadline:
            break
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    num_refine = min(3, len(candidates))
    
    for i in range(num_refine):
        if time.time() > deadline:
            break
            
        init_params = candidates[i][1]
        
        try:
            # Using a simple local search instead of scipy to avoid dependency issues if any, 
            # but scipy is available. Let's stick to the previous logic but ensure it's robust.
            # Actually, the prompt says SciPy is available.
            from scipy.optimize import minimize
            res = minimize(
                objective, 
                init_params, 
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3}
            )
            
            if res.success or res.fun < 0:
                final_score = -res.fun
                if final_score > best_score:
                    best_score = final_score
                    best_params = res.x
                    best_h = get_heuristic(best_params)
        except Exception:
            pass
    
    if time.time() < deadline - 30:
        rng = np.random.default_rng(123)
        for _ in range(10):
            if time.time() > deadline:
                break
            alpha = rng.uniform(0, 0.5)
            beta = rng.uniform(0.01, 0.5)
            gamma = rng.uniform(0.5, 3.0)
            params = (alpha, beta, gamma)
            h = get_heuristic(params)
            score = _simulate_aco(prizes, distance_matrix, max_len, h)
            if score > best_score:
                best_score = score
                best_params = params
                best_h = h
                
                try:
                    from scipy.optimize import minimize
                    res = minimize(
                        objective, 
                        params, 
                        method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-3, 'fatol': 1e-3}
                    )
                    if -res.fun > best_score:
                        best_score = -res.fun
                        best_params = res.x
                        best_h = get_heuristic(best_params)
                except Exception:
                    pass

    if best_h is None:
        best_h = get_heuristic((0.0, 0.01, 1.0))
        
    return best_h
