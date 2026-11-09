import numpy as np
import time

def _run_aco_vectorized(prizes, dist, max_len, heuristic, seed):
    """
    Run ACO with vectorized feasibility and probability checks.
    Returns the best total prize found.
    """
    n = len(prizes)
    rng = np.random.RandomState(seed)
    
    # Pheromone matrix
    pheromone = np.ones((n, n), dtype=float)
    # Heuristic matrix, ensuring positive values
    heur = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Pre-compute distances to origin for quick feasibility checks
    # dist_to_origin[j] = dist[j, 0]
    dist_to_origin = dist[:, 0]
    
    best_overall = 0.0
    
    for it in range(30):
        best_iter = 0.0
        
        for ant in range(20):
            # State
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            current = 0
            length = 0.0
            total_prize = 0.0
            
            while True:
                # Vectorized feasibility check
                # Condition: not visited AND length + dist[current, j] + dist[j, 0] <= max_len
                feasible_mask = (~visited_mask) & (length + dist[current, :] + dist_to_origin <= max_len)
                
                # Get indices of feasible nodes
                feasible_indices = np.where(feasible_mask)[0]
                
                if len(feasible_indices) == 0:
                    break
                
                # Calculate probabilities
                # p[j] = pheromone[current, j] * heur[current, j]
                # We only care about feasible_indices
                probs = pheromone[current, feasible_indices] * heur[current, feasible_indices]
                s = probs.sum()
                if s <= 0:
                    break
                probs = probs / s
                
                # Select next node
                j = int(rng.choice(feasible_indices, p=probs))
                
                # Update state
                length += dist[current, j]
                total_prize += prizes[j]
                visited_mask[j] = True
                current = j
            
            if total_prize > best_iter:
                best_iter = total_prize
            if total_prize > best_overall:
                best_overall = total_prize
        
        # Pheromone update
        # pheromone = pheromone * 0.9 + best_iter / 10.0
        pheromone *= 0.9
        pheromone += best_iter / 10.0
    
    return best_overall

def _make_heuristic(prizes, dist, max_len, params):
    """
    Generate heuristic matrix based on parameters.
    Supports power-law and exponential decay forms.
    """
    eps = 1e-9
    p = prizes[None, :].astype(float)
    d = dist.astype(float)
    ret = dist[:, 0][None, :].astype(float)
    
    a = params.get('d_exp', 1.0)
    b = params.get('ret_weight', 0.0)
    c = params.get('p_exp', 1.0)
    k = params.get('p_add', 0.0)
    
    if params.get('exp_decay', None) is not None:
        beta = params['exp_decay']
        H = (p + k) ** c * np.exp(-beta * (d + b * ret))
    else:
        denom = (d + b * ret + eps) ** a
        numer = (p + k) ** c
        H = numer / denom
    
    # Ensure finite and positive
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def _make_heuristic_composite(prizes, dist, max_len, params):
    """
    Composite heuristic that combines multiple terms.
    Term 1: prize / (forward_dist + return_weight * return_dist)
    Term 2: prize / total_cost where total_cost accounts for path budget
    """
    eps = 1e-9
    p = prizes[None, :].astype(float)
    d = dist.astype(float)
    ret = dist[:, 0][None, :].astype(float)
    
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 0.5)
    gamma = params.get('gamma', 1.0)
    delta = params.get('delta', 0.0)
    
    # Term 1: prize / (d + beta * ret)
    cost1 = d + beta * ret + eps
    term1 = (p + delta) / cost1
    
    # Term 2: prize / (d * ret + eps) - penalizes nodes that are far in both directions
    cost2 = d * ret + eps
    term2 = (p + delta) / cost2
    
    # Term 3: prize * exp(-gamma * (d + beta * ret))
    term3 = (p + delta) * np.exp(-gamma * (d + beta * ret))
    
    H = alpha * term1 + (1.0 - alpha) * (beta * term2 + (1.0 - beta) * term3)
    
    # Ensure finite and positive
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def _seed_heuristic(prizes, dist):
    """
    Fallback heuristic: Prize / Dist
    """
    H = (prizes[None, :] + 1e-9) / (dist + 1e-9)
    return np.where(np.isfinite(H) & (H > 0), H, 1e-9)

def heuristics(prizes, distance_matrix, max_len):
    try:
        start_time = time.time()
        # Conservative deadline to ensure we return well before the 330s limit
        deadline = start_time + 300.0
        
        prizes = np.asarray(prizes, dtype=float)
        dist = np.asarray(distance_matrix, dtype=float)
        n = len(prizes)
        
        # Define a diverse set of candidate heuristics
        # Mix of power-law, exponential decay, and composite forms
        # Focus on parameters that emphasize prize density and return distance
        candidates = [
            # Power-law forms
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0},
            {'d_exp': 2.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0},
            {'d_exp': 1.0, 'ret_weight': 1.0, 'p_exp': 1.0, 'p_add': 0.0},
            {'d_exp': 1.5, 'ret_weight': 0.5, 'p_exp': 1.5, 'p_add': 0.0},
            {'d_exp': 1.0, 'ret_weight': 0.0, 'p_exp': 2.0, 'p_add': 0.0},
            {'d_exp': 0.5, 'ret_weight': 0.5, 'p_exp': 1.0, 'p_add': 0.0},
            {'d_exp': 1.0, 'ret_weight': 0.5, 'p_exp': 1.0, 'p_add': 0.1},
            # Exponential decay forms
            {'exp_decay': 2.0, 'ret_weight': 0.0, 'p_exp': 1.0, 'p_add': 0.0},
            {'exp_decay': 3.0, 'ret_weight': 0.5, 'p_exp': 1.0, 'p_add': 0.0},
            {'exp_decay': 1.5, 'ret_weight': 1.0, 'p_exp': 1.5, 'p_add': 0.0},
            # Composite forms
            {'composite': True, 'alpha': 0.7, 'beta': 0.5, 'gamma': 1.0, 'delta': 0.0},
            {'composite': True, 'alpha': 0.5, 'beta': 0.3, 'gamma': 1.5, 'delta': 0.0},
            {'composite': True, 'alpha': 0.8, 'beta': 0.7, 'gamma': 0.8, 'delta': 0.05},
        ]
        
        num_seeds = 3
        results = []
        
        # Evaluate candidates
        for params in candidates:
            if time.time() > deadline - 60.0:
                break
                
            if params.get('composite', False):
                H = _make_heuristic_composite(prizes, dist, max_len, params)
            else:
                H = _make_heuristic(prizes, dist, max_len, params)
            scores = []
            
            # Run ACO with a few seeds to estimate quality
            for s in range(num_seeds):
                if time.time() > deadline - 60.0:
                    break
                score = _run_aco_vectorized(prizes, dist, max_len, H, s)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                results.append((mean_score, params, H, scores))
        
        if not results:
            # Fallback to seed heuristic if no time was left for search
            return _seed_heuristic(prizes, dist)
        
        # Sort by mean score
        results.sort(key=lambda x: -x[0])
        best_mean, best_params, best_H, best_scores = results[0]
        
        # If we have time, refine the top candidate with more seeds
        if time.time() < deadline - 30.0:
            # Try to add more seeds to the best candidate
            current_seed_count = len(best_scores)
            target_seed_count = current_seed_count + 5
            
            for s in range(current_seed_count, target_seed_count):
                if time.time() > deadline - 30.0:
                    break
                score = _run_aco_vectorized(prizes, dist, max_len, best_H, s)
                best_scores.append(score)
            
            new_mean = np.mean(best_scores)
            # Update best if improved (it will be, or same)
            best_mean = new_mean
        
        # Final safety check for validity
        best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
        return best_H
        
    except Exception:
        # Robust fallback
        try:
            prizes = np.asarray(prizes, dtype=float)
            dist = np.asarray(distance_matrix, dtype=float)
            return _seed_heuristic(prizes, dist)
        except Exception:
            try:
                n = len(prizes)
            except Exception:
                n = 50
            return np.ones((n, n))
