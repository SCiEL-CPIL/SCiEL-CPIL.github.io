import numpy as np
import time

def _run_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    
    d = distance_matrix
    d_j0 = d[0, :] 
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
            
        tau = 0.9 * tau + (best_iter / 10.0)
        
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    # Allow up to 120 seconds per instance to be safe within 300s total for 3 instances + overhead
    # Actually, the evaluator calls this 3 times. 300s total means ~100s per call.
    # Let's set a hard deadline of 95s for this function call to be safe.
    deadline = start_time + 95.0
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    d_j0 = d[0, :]
    
    # Generate candidates
    candidates = []
    
    # 1. Standard
    candidates.append(p[None, :] / (d + eps))
    
    # 2. Return Aware (most likely best)
    candidates.append(p[None, :] / (d + d_j0[None, :] + eps))
    
    # 3. Prize Powers
    for alpha in [0.5, 1.5, 2.0, 3.0]:
        p_pow = np.where(p > eps, p**alpha, 0)
        candidates.append(p_pow[None, :] / (d + d_j0[None, :] + eps))
        
    # 4. Distance Powers
    for beta in [0.5, 1.5, 2.0]:
        denom = (d + d_j0[None, :] + eps)**beta
        candidates.append(p[None, :] / denom)
        
    # 5. Hybrid
    candidates.append(p[None, :] / (d + eps) * (1 / (d_j0[None, :] + eps)))
    candidates.append(p[None, :] / (np.maximum(d, d_j0[None, :]) + eps))
    
    # 6. Scaled
    p_max = np.max(p)
    if p_max > 0:
        p_norm = p / p_max
        candidates.append(p_norm[None, :] / (d + d_j0[None, :] + eps))
        
    # 7. Greedy-like: prize / total_cost where total_cost is path cost if we go i->j->0
    # This is similar to 2 but let's try variations
    # Maybe prize / (d[i,j] + d[j,0] + d[0,i])? No, d[0,i] is constant for row i? No, i varies.
    # d[i,j] + d[j,0] is the cost of the step.
    
    # Let's add some more diverse candidates
    # Logarithmic prize
    p_log = np.log(p + 1e-6)
    candidates.append(p_log[None, :] / (d + d_j0[None, :] + eps))
    
    # Inverse distance only (ignoring prize magnitude, just closeness)
    candidates.append(1.0 / (d + d_j0[None, :] + eps))
    
    # Prize * Inverse Distance
    candidates.append(p[None, :] / (d + d_j0[None, :] + eps)) # Already have this
    
    # Evaluate candidates
    best_score = -1
    best_heuristic = candidates[0]
    
    # Use a fixed seed for evaluation
    eval_seed = 0
    
    # Sort candidates by a quick metric? Or just evaluate all if time permits.
    # There are about 15 candidates.
    # Each eval takes ~0.5-1s. 15 * 1s = 15s. Well within 95s.
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Validate
        if cand.shape != (n, n):
            continue
        if not np.all(np.isfinite(cand)):
            continue
            
        score = _run_aco(p, d, max_len, cand, num_ants=20, num_iters=30, seed=eval_seed)
        
        if score > best_score:
            best_score = score
            best_heuristic = cand
            
    # If we have more time, we can try to optimize parameters further
    # e.g., grid search on alpha and beta
    if time.time() < deadline - 10:
        # Try a few more specific combinations
        for alpha in np.arange(0.5, 3.5, 0.5):
            if time.time() > deadline - 5:
                break
            p_pow = np.where(p > eps, p**alpha, 0)
            for beta in np.arange(0.5, 2.5, 0.5):
                if time.time() > deadline - 5:
                    break
                denom = (d + d_j0[None, :] + eps)**beta
                cand = p_pow[None, :] / denom
                if cand.shape != (n, n) or not np.all(np.isfinite(cand)):
                    continue
                score = _run_aco(p, d, max_len, cand, num_ants=20, num_iters=30, seed=eval_seed)
                if score > best_score:
                    best_score = score
                    best_heuristic = cand
                    
    # Final cleanup
    best_heuristic = best_heuristic.copy()
    best_heuristic[best_heuristic <= 0] = 1e-9
    
    return best_heuristic
