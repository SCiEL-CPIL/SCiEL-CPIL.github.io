"""Structural change: prize-ratio heuristic with adaptive parameter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 320.0  # Safety margin (evaluator kills at 330s)
    
    N = len(prizes)
    d_j0 = distance_matrix[:, 0].copy()
    
    # Fast ACO simulation with correct state tracking
    def run_aco_fast(eta, seed, n_ants=20, n_iters=30):
        rng = np.random.RandomState(seed)
        tau = np.ones((N, N))
        
        best_prize_global = 0.0
        
        for it in range(n_iters):
            best_prize_iter = 0.0
            
            # State for all ants
            current = np.zeros(n_ants, dtype=int)
            path_len = np.zeros(n_ants)
            prize_total = np.zeros(n_ants)
            visited = np.zeros((n_ants, N), dtype=bool)
            visited[:, 0] = True
            active = np.ones(n_ants, dtype=bool)
            
            for step in range(N - 1):
                if not np.any(active):
                    break
                
                # For each active ant, find eligible nodes
                d_cur = distance_matrix[current, :]  # (n_ants, N)
                total_len = path_len[:, None] + d_cur + d_j0[None, :]
                eligible_mask = (~visited) & (total_len <= max_len)
                
                # Compute probabilities for eligible nodes
                prob_matrix = tau[current, :] * eta[current, :]
                prob_matrix[~eligible_mask] = 0.0
                
                # Normalize probabilities for each ant
                row_sums = prob_matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                prob_matrix = prob_matrix / row_sums
                
                # Sample next node for each ant
                cumsum = np.cumsum(prob_matrix, axis=1)
                r = rng.random_sample((n_ants, 1))
                next_nodes = np.argmax(cumsum >= r, axis=1)
                
                # Check if any ant has no eligible node
                no_eligible = (prob_matrix.sum(axis=1) == 0)
                next_nodes[no_eligible] = -1
                
                # Update state
                new_active = active & (next_nodes != -1)
                
                # Store old current before updating
                old_current = current.copy()
                
                current[new_active] = next_nodes[new_active]
                path_len[new_active] += distance_matrix[old_current[new_active], next_nodes[new_active]]
                prize_total[new_active] += prizes[next_nodes[new_active]]
                visited[new_active, next_nodes[new_active]] = True
                
                active = new_active
            
            # Update best_prize_iter
            if np.any(active):
                best_prize_iter = np.max(prize_total[active])
            else:
                best_prize_iter = 0.0
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            
            # Update pheromone
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global

    # Heuristic: prize[j] / (d[i,j] + d[j,0])^alpha
    # This is structurally different: it directly encodes the cost of visiting j
    # and returning to 0, weighted by the prize
    def make_heuristic_ratio(alpha, beta):
        p = np.maximum(prizes, 1e-12)[None, :]
        d_ij = np.maximum(distance_matrix, 1e-12)
        d_j0_t = np.maximum(d_j0, 1e-12)[None, :]
        
        # Cost to visit j and return to 0
        cost = d_ij + d_j0_t
        eta = p / np.power(cost, alpha)
        
        # Optional: add a small bonus for high prizes
        if beta > 0:
            eta *= np.power(np.maximum(prizes, 1e-12)[None, :], beta)
        
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    # Alternative: prize[j] * exp(-alpha * (d[i,j] + d[j,0]))
    def make_heuristic_exp(alpha, beta):
        p = np.maximum(prizes, 1e-12)[None, :]
        d_ij = distance_matrix
        d_j0_t = d_j0[None, :]
        
        cost = d_ij + d_j0_t
        eta = p * np.exp(-alpha * cost)
        
        if beta > 0:
            eta *= np.power(np.maximum(prizes, 1e-12)[None, :], beta)
        
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    # Another alternative: prize[j] / (1 + d[i,j])^alpha / (1 + d[j,0])^beta
    def make_heuristic_sep(alpha, beta):
        p = np.maximum(prizes, 1e-12)[None, :]
        d_ij = 1 + distance_matrix
        d_j0_t = 1 + d_j0[None, :]
        
        eta = p / (np.power(d_ij, alpha) * np.power(d_j0_t, beta))
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    # Search over parameters
    best_score = -1
    best_eta = None
    
    rng = np.random.RandomState(123)
    
    # Try a grid of parameters first, then random search
    param_configs = []
    
    # Ratio heuristic
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for beta in [0.0, 0.5, 1.0]:
            param_configs.append(('ratio', alpha, beta))
    
    # Exp heuristic
    for alpha in [1.0, 2.0, 3.0, 4.0]:
        for beta in [0.0, 0.5, 1.0]:
            param_configs.append(('exp', alpha, beta))
    
    # Sep heuristic
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for beta in [0.5, 1.0, 1.5, 2.0]:
            param_configs.append(('sep', alpha, beta))
    
    # Evaluate each config
    for config in param_configs:
        if time.time() - start_time > time_limit - 10:
            break
        
        h_type, alpha, beta = config
        
        if h_type == 'ratio':
            eta = make_heuristic_ratio(alpha, beta)
        elif h_type == 'exp':
            eta = make_heuristic_exp(alpha, beta)
        else:
            eta = make_heuristic_sep(alpha, beta)
        
        # Evaluate with multiple seeds to get a robust score
        scores = []
        for seed in [0, 1, 2]:
            if time.time() - start_time > time_limit - 5:
                break
            s = run_aco_fast(eta, seed)
            scores.append(s)
        
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_eta = eta
    
    # If we have time, do some random search
    if best_eta is not None and time.time() - start_time < time_limit - 15:
        for _ in range(50):
            if time.time() - start_time > time_limit - 5:
                break
            
            h_type = rng.choice(['ratio', 'exp', 'sep'])
            alpha = rng.uniform(0.3, 3.0)
            beta = rng.uniform(0.0, 1.5)
            
            if h_type == 'ratio':
                eta = make_heuristic_ratio(alpha, beta)
            elif h_type == 'exp':
                eta = make_heuristic_exp(alpha, beta)
            else:
                eta = make_heuristic_sep(alpha, beta)
            
            scores = []
            for seed in [0, 1]:
                if time.time() - start_time > time_limit - 3:
                    break
                s = run_aco_fast(eta, seed)
                scores.append(s)
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_eta = eta
    
    # Fallback
    if best_eta is None:
        best_eta = make_heuristic_ratio(1.0, 0.0)
    
    # Final safety check
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        cost = distance_matrix + d_j0[None, :]
        cost = np.maximum(cost, 1e-12)
        best_eta = prizes[None, :] / cost
        best_eta = np.where(best_eta > 0, best_eta, 1e-9)
    
    best_eta = np.where(best_eta > 0, best_eta, 1e-9)
    
    return best_eta

# EVOLVE-BLOCK-END
