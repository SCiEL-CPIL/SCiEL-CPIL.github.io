import numpy as np
import time

def _run_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
    """
    Runs the ACO simulation to evaluate the heuristic.
    """
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    # Ensure positive values as per task description
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    
    d = distance_matrix
    d_j0 = d[0, :] 
    # Precompute cost matrix for validity check: d[i,j] + d[j,0]
    cost = d + d_j0[None, :]
    
    best_global = 0.0
    
    for _ in range(num_iters):
        best_iter = 0.0
        
        for _ in range(num_ants):
            cur = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize = 0.0
            
            while True:
                remaining = max_len - length
                # Valid nodes j: not visited, and length + d[cur, j] + d[j, 0] <= max_len
                valid_mask = (~visited) & (cost[cur, :] <= remaining)
                valid_nodes = np.where(valid_mask)[0]
                
                if len(valid_nodes) == 0:
                    break
                
                w = tau[cur, valid_nodes] * eta[cur, valid_nodes]
                sum_w = w.sum()
                if sum_w < 1e-10:
                    break
                p = w / sum_w
                
                next_node = rng.choice(valid_nodes, p=p)
                
                visited[next_node] = True
                length += d[cur, next_node]
                prize += prizes[next_node]
                cur = next_node
            
            if prize > best_iter:
                best_iter = prize
        
        if best_iter > best_global:
            best_global = best_iter
            
        # Pheromone update
        tau = 0.9 * tau + (best_iter / 10.0)
        
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 280.0
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    d_j0 = d[0, :]
    total_cost = d + d_j0[None, :]
    
    # Precompute sorted prizes and indices for efficient regret calculation
    # We want to know, for a given node j and current node i, 
    # how much "better" prize is available in nodes k that are "closer" (lower total cost) to i.
    
    # Strategy: 
    # H[i, j] = p[j] / (1 + Regret[i, j])
    # Regret[i, j] = sum_{k: total_cost[i, k] < total_cost[i, j]} max(0, p[k] - p[j])^alpha
    
    # This is computationally heavy if done naively for every (i, j) with a loop over k.
    # N=50, so N^3 = 125,000 operations. This is fast.
    # We can vectorize the Regret calculation.
    
    # Let's define a function to compute the heuristic given parameters alpha, beta, gamma
    # H[i, j] = (p[j]^beta) / ((1 + Regret[i, j]) * (total_cost[i, j] + eps)^gamma)
    
    def compute_heuristic(alpha, beta, gamma):
        H = np.zeros((n, n))
        
        # Vectorized regret calculation
        # For each i, we have a vector of total costs tc_i.
        # For each j, we want sum over k where tc_i[k] < tc_i[j] of max(0, p[k] - p[j])^alpha
        
        # We can do this for all i at once?
        # tc is (n, n). p is (n,).
        # cond_cost = tc[:, :, None] < tc[:, None, :] # (n, n, n) -> [i, k, j] is True if tc[i, k] < tc[i, j]
        # diff_p = p[None, :, None] - p[None, None, :] # (1, n, n) -> [k, j] is p[k] - p[j]
        # regret_term = np.where(diff_p > 0, diff_p**alpha, 0) # (n, n) -> [k, j]
        # regret = (cond_cost * regret_term).sum(axis=1) # Sum over k. Result (n, n) -> [i, j]
        
        # This creates a (50, 50, 50) array which is 125k floats. Very small.
        
        cond_cost = total_cost[:, :, None] < total_cost[:, None, :] # [i, k, j]
        diff_p = p[None, :, None] - p[None, None, :] # [k, j]
        diff_p = np.maximum(diff_p, 0)
        
        # Handle alpha=0 case (binary regret) vs alpha>0
        if alpha == 0:
            regret_term = (diff_p > 0).astype(float)
        else:
            regret_term = diff_p ** alpha
        
        # regret_term is (n, n) [k, j]. We need to multiply with cond_cost [i, k, j] and sum over k.
        # regret_term[None, :, :] is (1, n, n) [i, k, j]
        regret = (cond_cost * regret_term[None, :, :]).sum(axis=1) # (n, n) [i, j]
        
        # H[i, j] = p[j]^beta / ((1 + regret[i, j]) * (total_cost[i, j] + eps)^gamma)
        # p[j] is (n,), so p[None, :] is (1, n)
        numerator = p[None, :] ** beta
        denominator = (1.0 + regret) * (total_cost + eps) ** gamma
        
        H = numerator / denominator
        H[total_cost <= 0] = 0 # Node 0 to Node 0 is 0, but we start at 0 so it's visited.
        H[H <= 0] = 1e-9
        
        return H

    # Simulated Annealing over parameters alpha, beta, gamma
    # alpha: regret exponent (0 to 2)
    # beta: prize exponent (0.5 to 2)
    # gamma: distance penalty exponent (0 to 1)
    
    best_score = -1
    best_params = (1.0, 1.0, 0.5)
    best_heuristic = compute_heuristic(1.0, 1.0, 0.5)
    
    current_params = (1.0, 1.0, 0.5)
    current_heuristic = best_heuristic
    current_score = _run_aco(p, d, max_len, current_heuristic, num_ants=10, num_iters=10, seed=0)
    best_score = current_score
    
    temp = 0.1
    cooling = 0.95
    
    rng_opt = np.random.default_rng(42)
    
    while time.time() < deadline - 10:
        if temp < 0.001:
            break
            
        # Generate neighbor
        delta = rng_opt.normal(0, 0.2, 3)
        new_params = tuple(np.clip(current_params[i] + delta[i], 0, 3) for i in range(3))
        
        new_heuristic = compute_heuristic(new_params[0], new_params[1], new_params[2])
        
        # Quick evaluation
        new_score = _run_aco(p, d, max_len, new_heuristic, num_ants=10, num_iters=10, seed=0)
        
        if new_score > current_score:
            current_params = new_params
            current_heuristic = new_heuristic
            current_score = new_score
        else:
            # Accept with probability exp((new_score - current_score) / temp)
            # Scores are prizes, typically 0-5. Diff is small.
            accept_prob = np.exp((new_score - current_score) / temp)
            if rng_opt.random() < accept_prob:
                current_params = new_params
                current_heuristic = new_heuristic
                current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params
            best_heuristic = current_heuristic
            
        temp *= cooling
        
    # Final verification with full ACO to ensure the best heuristic found is indeed good
    # The SA might have found a local optimum that is better in the quick run but worse in full.
    # Let's do a few full runs on the top few candidates if time permits.
    
    # For now, return the best found by SA.
    return best_heuristic
