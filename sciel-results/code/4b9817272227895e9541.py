import numpy as np
import time

def _run_aco_fast(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
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

def _run_aco_quick(prizes, distance_matrix, max_len, heuristic, num_ants=10, num_iters=15, seed=0):
    """Faster version for initial candidate screening"""
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

def _compute_local_density(prizes, distance_matrix, k=5):
    """Compute prize-weighted average distance to top-k nearest neighbors for each node."""
    n = len(prizes)
    eps = 1e-9
    local_density = np.ones(n, dtype=float)
    
    for j in range(n):
        dists_to_j = distance_matrix[:, j].copy()
        dists_to_j[j] = np.inf
        
        k_actual = min(k, n - 1)
        if k_actual > 0:
            nearest_indices = np.argpartition(dists_to_j, k_actual)[:k_actual]
            nearest_dists = dists_to_j[nearest_indices]
            nearest_prizes = prizes[nearest_indices]
            
            total_prize = np.sum(nearest_prizes)
            if total_prize > eps:
                local_density[j] = np.sum(nearest_prizes * nearest_dists) / total_prize
            else:
                local_density[j] = np.mean(nearest_dists) if len(nearest_dists) > 0 else 1.0
    
    local_density = np.where(np.isfinite(local_density) & (local_density > 0), local_density, 1.0)
    return local_density

def _make_candidate(prizes, distance_matrix, max_len, gamma, beta, alpha, use_density, density):
    """Generate a heuristic candidate with given parameters."""
    n = len(prizes)
    eps = 1e-9
    d = distance_matrix
    d_to_start = d[:, 0].copy()
    
    eff_cost = d + d_to_start[None, :]
    eff_cost_safe = eff_cost + eps
    
    prize_row = prizes[None, :]
    
    # Prize efficiency component
    if use_density:
        prize_eff = ((prize_row + eps) / density[None, :]) ** alpha
    else:
        prize_eff = (prize_row + eps) ** alpha
    
    prize_eff = (prize_eff / eff_cost_safe) ** gamma
    
    # Distance penalty
    dist_penalty = (1.0 / (1.0 + d / max_len)) ** beta
    
    eta = prize_eff * dist_penalty
    eta[:, 0] = 1e-9
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 295.0
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    # Compute local density of value
    local_density = _compute_local_density(p, d, k=5)
    
    # Generate diverse candidates with different parameter combinations
    candidates = []
    
    # Parameter grid: gamma (efficiency power), beta (distance penalty), alpha (prize power)
    gamma_vals = [1.5, 2.0, 2.5, 3.0]
    beta_vals = [1.0, 1.5, 2.0]
    alpha_vals = [1.0, 1.5, 2.0]
    
    # Without density
    for g in gamma_vals:
        for b in beta_vals:
            for a in alpha_vals:
                cand = _make_candidate(p, d, max_len, g, b, a, use_density=False, density=None)
                if cand.shape == (n, n) and np.all(np.isfinite(cand)):
                    candidates.append(cand)
    
    # With density
    for g in gamma_vals:
        for b in beta_vals:
            for a in alpha_vals:
                cand = _make_candidate(p, d, max_len, g, b, a, use_density=True, density=local_density)
                if cand.shape == (n, n) and np.all(np.isfinite(cand)):
                    candidates.append(cand)
    
    # Also add the classic return-aware heuristic
    d_j0 = d[0, :]
    eff_cost_simple = d + d_j0[None, :] + eps
    for g in [1.0, 2.0, 2.5, 3.0]:
        cand = (p[None, :] / eff_cost_simple) ** g
        cand[:, 0] = 1e-9
        cand = np.where(np.isfinite(cand) & (cand > 0), cand, 1e-9)
        candidates.append(cand)
    
    if not candidates:
        return np.ones((n, n))
    
    # Phase 1: Quick screening
    quick_results = []
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 40:
            break
        score = _run_aco_quick(p, d, max_len, cand, num_ants=10, num_iters=15, seed=0)
        quick_results.append((score, i))
    
    quick_results.sort(reverse=True)
    top_candidates = [candidates[i] for _, i in quick_results[:10]]
    
    # Phase 2: Full evaluation of top candidates
    best_score = -1
    best_heuristic = top_candidates[0] if top_candidates else candidates[0]
    
    for cand in top_candidates:
        if time.time() > deadline - 20:
            break
        score = _run_aco_fast(p, d, max_len, cand, num_ants=20, num_iters=30, seed=0)
        if score > best_score:
            best_score = score
            best_heuristic = cand
    
    # Phase 3: Fine-tune around the best candidate
    if time.time() < deadline - 15:
        best_base = best_heuristic.copy()
        best_base[best_base <= 0] = 1e-9
        
        # Scale variations
        for scale in [0.5, 0.75, 1.25, 1.5, 2.0]:
            if time.time() > deadline - 10:
                break
            cand = best_base * scale
            if cand.shape == (n, n) and np.all(np.isfinite(cand)):
                score = _run_aco_fast(p, d, max_len, cand, num_ants=20, num_iters=30, seed=0)
                if score > best_score:
                    best_score = score
                    best_heuristic = cand
        
        # Small perturbations
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            noise = rng.normal(0, 0.1, size=(n, n))
            cand = best_base + noise
            cand = np.maximum(cand, 1e-9)
            if cand.shape == (n, n) and np.all(np.isfinite(cand)):
                score = _run_aco_fast(p, d, max_len, cand, num_ants=20, num_iters=30, seed=0)
                if score > best_score:
                    best_score = score
                    best_heuristic = cand
    
    # Final cleanup
    best_heuristic = best_heuristic.copy()
    best_heuristic[best_heuristic <= 0] = 1e-9
    
    return best_heuristic
