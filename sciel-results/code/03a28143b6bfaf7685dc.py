import numpy as np
import time
from scipy.optimize import minimize

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    """
    Simulates the ACO process to evaluate a heuristic.
    Deterministic because seed is fixed.
    """
    n = len(prizes)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    rng = np.random.default_rng(seed)
    
    d_to_start = distance_matrix[:, 0]
    
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
                d_from_current = distance_matrix[current_node, :]
                valid_indices = np.where(
                    (~visited_mask) & 
                    (d_from_current + d_to_start <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                h_vals = H[current_node, valid_indices]
                
                total_h = np.sum(h_vals)
                if total_h <= 0:
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
    deadline = start_time + 290  # 10s safety margin
    
    n = len(prizes)
    eps = 1e-9
    
    # Pre-compute distances to start for speed in objective if needed, 
    # but _simulate_aco does it.
    
    # Strategy: Parametric Heuristic Optimization
    # H[i, j] = ((prizes[j] + a) ** alpha) / ((distance_matrix[i, j] + b) ** beta)
    
    # We search for a, b, alpha, beta
    # Constraints: a > -min(prizes) to keep prize term positive (prizes >= 0, so a > -1 is safe, let's say a > -0.5)
    # b > 0 to avoid division by zero
    # alpha, beta > 0
    
    def build_heuristic(params):
        a, b, alpha, beta = params
        # Ensure positivity of base terms
        prize_term = prizes + a
        if np.any(prize_term <= 0):
            # Fallback to safe values if params are bad
            return np.full((n, n), 1e-9)
        
        dist_term = distance_matrix + b
        if np.any(dist_term <= 0):
            return np.full((n, n), 1e-9)
            
        H = np.zeros((n, n))
        # Vectorized computation
        # H[i, j] = (prizes[j] + a)^alpha / (dist[i, j] + b)^beta
        # Note: prizes is 1D (n,), dist is 2D (n, n)
        # We need prizes[j] for each column j.
        p_col = np.tile(prizes + a, (n, 1)) # n x n
        d_mat = distance_matrix + b          # n x n
        
        # Avoid 0^0 or negative bases
        p_safe = np.maximum(p_col, 1e-9)
        d_safe = np.maximum(d_mat, 1e-9)
        
        H = (p_safe ** alpha) / (d_safe ** beta)
        
        # Zero out diagonal? Not strictly necessary as ants don't stay, but good practice.
        np.fill_diagonal(H, 1e-9)
        
        # Ensure finite
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        
        return H

    def objective(params):
        if time.time() > deadline:
            return 0.0
        H = build_heuristic(params)
        score = _simulate_aco(prizes, distance_matrix, max_len, H)
        return -score

    # Multi-start Nelder-Mead
    # Initial guesses for (a, b, alpha, beta)
    # Standard heuristic is a=0, b=0.1, alpha=1, beta=1
    # Let's try a variety of starts
    
    starts = [
        [0.0, 0.1, 1.0, 1.0],   # Standard prize/distance
        [0.1, 0.05, 1.5, 1.0],  # Higher prize sensitivity
        [0.0, 0.2, 1.0, 1.5],   # Higher distance penalty
        [0.5, 0.1, 0.5, 0.5],   # Softer exponents
        [0.0, 0.01, 2.0, 2.0],  # Sharper contrast
        [0.2, 0.1, 1.0, 0.8],   # Slight distance discount
        [0.0, 0.5, 1.0, 1.0],   # Heavy distance smoothing
        [0.1, 0.1, 1.2, 1.2]
    ]
    
    best_score = -1
    best_h = None
    best_params = None
    
    for start in starts:
        if time.time() > deadline - 30:
            break
            
        # Bounds for Nelder-Mead (it doesn't support bounds natively, so we rely on the function handling bad params)
        # But we can clip params inside build_heuristic if needed.
        # For now, build_heuristic handles invalids by returning 1e-9 matrix (score 0).
        
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 100, 'fatol': 1e-6, 'disp': False})
            
            if res.success or res.fun < -best_score:
                # Check if this is a valid improvement
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
                    best_h = build_heuristic(res.x)
        except Exception:
            pass

    # If parametric optimization didn't find anything good, or if we have time,
    # do a local search on the matrix itself starting from the best parametric solution.
    if time.time() < deadline - 10:
        if best_h is None:
            # Fallback to standard
            best_h = np.where(prizes[None, :] / (distance_matrix + 0.1) > 0, 
                              prizes[None, :] / (distance_matrix + 0.1), 1e-9)
            np.fill_diagonal(best_h, 1e-9)
            best_score = _simulate_aco(prizes, distance_matrix, max_len, best_h)
        
        # Local Jitter Search on the Matrix
        # We perturb the log-heuristic to allow multiplicative changes
        log_h = np.log(np.where(best_h > 0, best_h, 1e-9))
        
        current_score = best_score
        current_log_h = log_h.copy()
        
        # Number of steps depends on time
        step_size = 0.1
        max_steps = 50
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Random perturbation
            # Focus perturbation on edges that are likely used? 
            # For simplicity, global small perturbation.
            noise = np.random.normal(0, step_size, size=(n, n))
            # Don't perturb diagonal
            np.fill_diagonal(noise, 0)
            
            new_log_h = current_log_h + noise
            new_h = np.exp(new_log_h)
            new_h = np.where(new_h > 0, new_h, 1e-9)
            np.fill_diagonal(new_h, 1e-9)
            
            new_score = _simulate_aco(prizes, distance_matrix, max_len, new_h)
            
            if new_score > current_score:
                current_score = new_score
                current_log_h = new_log_h
                best_h = new_h
                best_score = new_score
                # Reduce step size slightly on success (greedy)
                step_size *= 0.9
            else:
                # Keep step size or increase slightly? 
                # Standard random walk: keep or increase.
                # Let's keep it simple.
                pass
                
    return best_h
