import numpy as np
import time

def _run_aco(prizes, distance_matrix, max_len, alpha, beta, n_ants=100, n_iters=10, seed=42):
    """
    Runs a fast ACO simulation to evaluate heuristic quality.
    Returns the best prize found in this simulation.
    """
    n = len(prizes)
    if n == 0:
        return 0.0
    
    # Precompute heuristic for this specific alpha, beta
    # eta_ij = p_j^alpha / (d_ij + d_j0 + eps)^beta
    # Note: p_0 is 0, so eta_i0 is 0 (or 1e-9). 
    # The move is i -> j. j is the destination.
    # Heuristic depends on destination j's prize and distance from i to j plus j to 0.
    
    d_j0 = distance_matrix[:, 0]
    # eff_dist[i, j] = d[i, j] + d[j, 0]
    eff_dist = distance_matrix + d_j0[None, :]
    eff_dist = np.maximum(eff_dist, 1e-9)
    
    # prizes[None, :] is shape (1, n). Broadcasting to (n, n) gives row i, col j = prizes[j]
    prize_matrix = prizes[None, :]
    
    # Handle p=0 case for node 0. Since we never return to 0 as a 'move' in the same way,
    # but j can be 0? No, "moves to an unvisited node j". 0 is visited at start.
    # So j is never 0 in the selection step for the first move? 
    # "Each ant starts at node 0 and moves to an unvisited node j". 
    # So j != 0.
    # However, the heuristic matrix is full. We just need eta[i, j] for valid moves.
    
    # p_j^alpha. If p_j=0, 0^alpha = 0.
    # We use np.power.
    try:
        num = np.power(prize_matrix, alpha)
    except:
        num = prize_matrix.copy()
        num[num < 0] = 0 # Should not happen with uniform [0,1]
        
    den = np.power(eff_dist, beta)
    
    eta = num / den
    # Ensure positive
    eta = np.maximum(eta, 1e-9)
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_prize_global = 0.0
    rng = np.random.default_rng(seed)
    
    for _ in range(n_iters):
        best_prize_iter = 0.0
        
        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            current_len = 0.0
            total_prize = prizes[0] # Prize 0 is 0
            
            while True:
                # Find unvisited nodes j such that current_len + d[current, j] + d[j, 0] <= max_len
                # d[current, j] is distance_matrix[current, :]
                # d[j, 0] is d_j0
                # Condition: current_len + dist_mat[current, :] + d_j0 <= max_len
                
                # Vectorized check
                # dist_mat[current, :] is (n,)
                # d_j0 is (n,)
                # visited is (n,)
                
                # We need indices j where visited[j] is False AND condition holds
                # Also j != current (implied by visited if we mark current as visited? No, current is visited)
                
                # Calculate feasibility mask
                # Note: d[current, current] is 0, but visited[current] is True, so it's excluded.
                
                feasible_mask = ~visited
                # Add length constraint
                # len_so_far + d[i, j] + d[j, 0] <= max_len
                # d[i, j] is distance_matrix[current, :]
                # d[j, 0] is d_j0
                
                # We only care about unvisited nodes
                # Let's compute the required length for each j
                req_len = current_len + distance_matrix[current, :] + d_j0
                
                feasible_mask = feasible_mask & (req_len <= max_len)
                
                # Get indices of feasible nodes
                feasible_indices = np.where(feasible_mask)[0]
                
                if len(feasible_indices) == 0:
                    break
                
                # Calculate probabilities
                # prob[j] = tau[current, j] * eta[current, j]
                # Only for feasible_indices
                
                # Extract relevant tau and eta
                # tau[current, feasible_indices]
                # eta[current, feasible_indices]
                
                t = tau[current, feasible_indices]
                e = eta[current, feasible_indices]
                
                probs = t * e
                sum_probs = np.sum(probs)
                
                if sum_probs == 0:
                    break
                    
                probs /= sum_probs
                
                # Select next node
                # Use rng.choice with p
                # rng.choice(feasible_indices, p=probs)
                # This might be slow in a tight loop. 
                # Alternative: cumulative sum and binary search.
                # But for n=50, choice is probably fine.
                
                next_idx = rng.choice(feasible_indices, p=probs)
                
                # Update state
                current_len = req_len[next_idx]
                visited[next_idx] = True
                total_prize += prizes[next_idx]
                current = next_idx
                
            if total_prize > best_prize_iter:
                best_prize_iter = total_prize
                
        if best_prize_iter > best_prize_global:
            best_prize_global = best_prize_iter
            
        # Pheromone update
        # tau = tau * 0.9 + (best_prize_iter / 10)
        # The task says: "After each iteration, pheromone is multiplied by 0.9 and every entry gains (best prize of that iteration) / 10."
        # This applies to the whole matrix? "every entry".
        # Usually ACO updates on the path, but the prompt says "every entry gains...".
        # This is a very specific rule. It means pheromone becomes uniform-ish quickly?
        # tau_new = tau_old * 0.9 + constant.
        # This converges to constant / 0.1 = 10 * constant.
        # If best_prize is ~1, tau -> 1.
        # If best_prize is ~5, tau -> 5.
        # This effectively scales the pheromone based on global best, but uniformly.
        # So tau[i, j] becomes independent of i, j over time?
        # Yes, if we add a scalar to every entry.
        # Then tau[i, j] * eta[i, j] is proportional to eta[i, j].
        # So the heuristic dominates.
        
        tau = tau * 0.9 + (best_prize_iter / 10.0)
        
    return best_prize_global

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = 320.0 # 300s limit + safety margin
    
    n = len(prizes)
    
    # Define a grid of parameters to search
    # alpha: power on prize. Higher alpha -> more greedy on prize.
    # beta: power on distance. Higher beta -> more greedy on distance (shorter).
    
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0]
    betas = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    best_score = -1
    best_alpha = 1.0
    best_beta = 1.0
    
    # Evaluate each pair
    # We can parallelize? No, standard lib. 
    # We can reduce search space if time is tight.
    
    # Sort pairs by "centrality" to try good ones first?
    # Or just iterate.
    
    for alpha in alphas:
        for beta in betas:
            if time.time() > deadline:
                break
            
            # Run simulation
            # To save time, we can reduce n_ants/n_iters for the search phase
            # and maybe do a final validation with more ants?
            # But the heuristic is static. The ACO inside the evaluator is fixed (20 ants, 30 iters).
            # Our internal ACO is just a proxy.
            # Let's use 50 ants, 5 iters for speed.
            
            score = _run_aco(prizes, distance_matrix, max_len, alpha, beta, n_ants=50, n_iters=5, seed=123)
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_beta = beta
        
        if time.time() > deadline:
            break
            
    # Construct the final heuristic matrix using best_alpha, best_beta
    d_j0 = distance_matrix[:, 0]
    eff_dist = distance_matrix + d_j0[None, :]
    eff_dist = np.maximum(eff_dist, 1e-9)
    
    prize_matrix = prizes[None, :]
    
    num = np.power(prize_matrix, best_alpha)
    den = np.power(eff_dist, best_beta)
    
    eta = num / den
    eta = np.maximum(eta, 1e-9)
    
    return eta
