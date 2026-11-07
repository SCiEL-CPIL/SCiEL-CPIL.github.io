"""Model-and-Optimize approach for Heuristic ACO Orienteering."""
import numpy as np
import time
import random

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 325.0  # Safety margin before 330s kill switch
    
    N = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    d_j0 = distance_matrix[:, 0].copy()
    
    # Precompute: round-trip cost from i to j via 0 = d[i,j] + d[j,0]
    # This is the "effective cost" of visiting j from i and being able to return
    round_trip_cost = distance_matrix + d_j0[None, :]  # shape (N, N)
    
    def evaluate_heuristic(params, seed):
        """
        Run ACO with a specific heuristic parameterization and return best prize.
        params: [alpha, beta, gamma, delta, theta]
        Heuristic: eta_ij = ( P_j^alpha / ( (D_ij + eps)^beta * (D_j0 + eps)^gamma * (RT_ij + eps)^theta ) )^delta
        where RT_ij = D_ij + D_j0 (round-trip awareness)
        """
        np.random.seed(seed)
        
        alpha, beta, gamma, delta, theta = params
        
        # Compute Heuristic Matrix
        eps = 1e-9
        
        P = np.maximum(prizes, eps)
        D = np.maximum(distance_matrix, eps)
        Dj0 = np.maximum(d_j0, eps)
        RT = np.maximum(round_trip_cost, eps)
        
        num = np.power(P[None, :], alpha)
        den = np.power(D, beta) * np.power(Dj0[None, :], gamma) * np.power(RT, theta)
        
        eta_base = num / den
        eta = np.power(eta_base, delta)
        
        # Enforce positivity and replace non-positive with 1e-9
        eta = np.where(eta > 0, eta, 1e-9)
        
        # ACO Simulation - optimized
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
                    # Find eligible nodes: unvisited and d[current,j] + d[j,0] <= remaining
                    remaining = max_len - path_len
                    costs = round_trip_cost[current, :]
                    mask = (~visited) & (costs <= remaining)
                    
                    if not np.any(mask):
                        break
                    
                    eligible_idx = np.where(mask)[0]
                    
                    # Weights: tau[i, j] * eta[i, j]
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

    # Define parameter bounds and initial guesses
    # params: [alpha, beta, gamma, delta, theta]
    initial_x0 = [1.0, 1.0, 1.0, 1.0, 0.0]
    
    # Phase 1: Coarse Grid Search to find a good basin
    # Reduced grid for speed, then refine
    grid_alpha = [0.5, 1.0, 2.0, 3.0]
    grid_beta = [0.5, 1.0, 1.5, 2.0]
    grid_gamma = [0.0, 0.5, 1.0, 1.5]
    grid_delta = [0.5, 1.0, 1.5]
    grid_theta = [0.0, 0.5, 1.0]
    
    candidates = []
    for a in grid_alpha:
        for b in grid_beta:
            for g in grid_gamma:
                for d in grid_delta:
                    for t in grid_theta:
                        candidates.append([a, b, g, d, t])
    
    # Evaluate with seed 0 only for speed in phase 1
    best_score_grid = -1
    best_params_grid = initial_x0
    
    # Limit grid search: 4*4*4*3*3 = 576 points, use time budget
    for p in candidates:
        if time.time() - start_time > time_limit - 80:
            break
        score = evaluate_heuristic(p, 0)
        if score > best_score_grid:
            best_score_grid = score
            best_params_grid = p
    
    # Phase 2: Local Optimization using Nelder-Mead
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy:
        # Objective function to minimize (negative of mean score)
        def objective(params):
            if time.time() - start_time > time_limit - 10:
                return 0.0
            # Use 2 seeds for stability during optimization
            return -multi_seed_eval(params, [0, 1])
        
        # Start from best grid point
        x0 = np.array(best_params_grid, dtype=np.float64)
        
        # Bounds for [alpha, beta, gamma, delta, theta]:
        bounds = [(0.1, 5.0), (0.1, 5.0), (0.0, 5.0), (0.1, 2.0), (0.0, 3.0)]
        
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={'maxiter': 60, 'xatol': 0.03, 'fatol': 0.005}
        )
        
        if time.time() - start_time < time_limit - 25:
            best_params_local = res.x
            best_score_local = -res.fun
        else:
            best_params_local = x0
            best_score_local = -objective(x0)
            
    else:
        # Fallback: simple hill climbing
        best_params_local = list(best_params_grid)
        best_score_local = best_score_grid
        
        for _ in range(50):
            if time.time() - start_time > time_limit - 20:
                break
            new_p = [x + random.uniform(-0.15, 0.15) for x in best_params_local]
            new_p[0] = np.clip(new_p[0], 0.1, 5.0)
            new_p[1] = np.clip(new_p[1], 0.1, 5.0)
            new_p[2] = np.clip(new_p[2], 0.0, 5.0)
            new_p[3] = np.clip(new_p[3], 0.1, 2.0)
            new_p[4] = np.clip(new_p[4], 0.0, 3.0)
            
            s = evaluate_heuristic(new_p, 0)
            if s > best_score_local:
                best_params_local = new_p
                best_score_local = s

    # Phase 3: Final Verification and Selection
    final_candidates = [
        (best_params_grid, "grid"),
        (best_params_local, "local")
    ]
    
    # Add the initial guess just in case
    final_candidates.append((initial_x0, "init"))
    
    best_final_params = best_params_local
    best_final_score = -np.inf
    
    for p, name in final_candidates:
        if time.time() - start_time > time_limit - 5:
            break
        # Evaluate with seeds 0, 1, 2
        score = multi_seed_eval(p, [0, 1, 2])
        if score > best_final_score:
            best_final_score = score
            best_final_params = p
    
    # Phase 4: High-precision polish with more seeds if time permits
    if time.time() - start_time < time_limit - 30:
        # Try a few perturbations around the best found params
        polish_candidates = [best_final_params]
        for _ in range(5):
            if time.time() - start_time > time_limit - 30:
                break
            perturbed = [x + random.gauss(0, 0.05) for x in best_final_params]
            perturbed[0] = np.clip(perturbed[0], 0.1, 5.0)
            perturbed[1] = np.clip(perturbed[1], 0.1, 5.0)
            perturbed[2] = np.clip(perturbed[2], 0.0, 5.0)
            perturbed[3] = np.clip(perturbed[3], 0.1, 2.0)
            perturbed[4] = np.clip(perturbed[4], 0.0, 3.0)
            polish_candidates.append(perturbed)
        
        for p in polish_candidates:
            if time.time() - start_time > time_limit - 10:
                break
            score = multi_seed_eval(p, [0, 1, 2, 3, 4])
            if score > best_final_score:
                best_final_score = score
                best_final_params = p
    
    # Construct final heuristic
    alpha, beta, gamma, delta, theta = best_final_params
    eps = 1e-9
    
    P = np.maximum(prizes, eps)
    D = np.maximum(distance_matrix, eps)
    Dj0 = np.maximum(d_j0, eps)
    RT = np.maximum(round_trip_cost, eps)
    
    num = np.power(P[None, :], alpha)
    den = np.power(D, beta) * np.power(Dj0[None, :], gamma) * np.power(RT, theta)
    
    eta_base = num / den
    eta = np.power(eta_base, delta)
    
    eta = np.where(eta > 0, eta, 1e-9)
    
    # Safety check
    if not np.all(np.isfinite(eta)) or eta.shape != distance_matrix.shape:
        # Fallback to simple prize/distance
        eta = (prizes[None, :] + eps) / (distance_matrix + eps)
        eta = np.where(eta > 0, eta, 1e-9)

    return eta
