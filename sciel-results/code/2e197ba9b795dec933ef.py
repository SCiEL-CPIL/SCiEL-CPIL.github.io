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

def _compute_regret_heuristic(prizes, distance_matrix, max_len, budget_frac=0.5):
    """
    Regret-based heuristic: attractiveness of moving to j from i is inversely
    proportional to the opportunity cost (max prize of other reachable nodes).
    """
    n = len(prizes)
    eps = 1e-9
    d = distance_matrix
    p = prizes
    
    budget = budget_frac * max_len
    eta = np.ones((n, n), dtype=float)
    
    for i in range(n):
        reachable_mask = (d[i, :] <= budget) & (np.arange(n) != i)
        reachable_indices = np.where(reachable_mask)[0]
        
        if len(reachable_indices) == 0:
            for j in range(n):
                if j != i:
                    eta[i, j] = p[j] + eps
            continue
        
        reachable_prizes = p[reachable_indices]
        
        for j in range(n):
            if j == i:
                eta[i, j] = eps
                continue
            
            if j in reachable_indices:
                other_mask = reachable_indices != j
                if np.any(other_mask):
                    regret = np.max(reachable_prizes[other_mask])
                else:
                    regret = 0.0
            else:
                regret = 0.0
            
            eta[i, j] = (p[j] + eps) / (1.0 + regret)
    
    d_j0 = d[0, :]
    eff_cost = d + d_j0[None, :]
    dist_penalty = 1.0 / (1.0 + eff_cost / max_len)
    
    eta = eta * dist_penalty
    eta[:, 0] = eps
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    return eta

def _compute_regret_heuristic_v2(prizes, distance_matrix, max_len, k_neighbors=10):
    """
    Regret heuristic v2: heuristic[i,j] = prize[j] / (1 + count of higher-prize k-neighbors)
    """
    n = len(prizes)
    eps = 1e-9
    d = distance_matrix
    p = prizes
    
    eta = np.ones((n, n), dtype=float)
    
    for i in range(n):
        dists = d[i, :].copy()
        dists[i] = np.inf
        
        k_actual = min(k_neighbors, n - 1)
        if k_actual <= 0:
            continue
        
        nearest_idx = np.argpartition(dists, k_actual)[:k_actual]
        nearest_prizes = p[nearest_idx]
        
        for j in range(n):
            if j == i:
                eta[i, j] = eps
                continue
            
            n_higher = np.sum(nearest_prizes > p[j])
            eta[i, j] = (p[j] + eps) / (1.0 + n_higher)
    
    d_j0 = d[0, :]
    eff_cost = d + d_j0[None, :]
    dist_penalty = 1.0 / (1.0 + eff_cost / max_len)
    
    eta = eta * dist_penalty
    eta[:, 0] = eps
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    return eta

def _compute_regret_heuristic_v3(prizes, distance_matrix, max_len, threshold_frac=0.7):
    """
    Regret heuristic v3: heuristic[i,j] = prize[j] / (1 + sum of prizes of closer reachable nodes)
    """
    n = len(prizes)
    eps = 1e-9
    d = distance_matrix
    p = prizes
    
    budget = threshold_frac * max_len
    eta = np.ones((n, n), dtype=float)
    
    for i in range(n):
        dists_from_i = d[i, :]
        reachable_mask = (dists_from_i <= budget) & (np.arange(n) != i)
        reachable_idx = np.where(reachable_mask)[0]
        
        if len(reachable_idx) == 0:
            for j in range(n):
                if j != i:
                    eta[i, j] = p[j] + eps
            continue
        
        dists_to_reachable = dists_from_i[reachable_idx]
        prizes_of_reachable = p[reachable_idx]
        
        for j in range(n):
            if j == i:
                eta[i, j] = eps
                continue
            
            dist_to_j = dists_from_i[j]
            closer_mask = dists_to_reachable < dist_to_j
            if j in reachable_idx:
                j_pos = np.where(reachable_idx == j)[0]
                if len(j_pos) > 0:
                    closer_mask[j_pos[0]] = False
            
            opportunity_cost = np.sum(prizes_of_reachable[closer_mask])
            eta[i, j] = (p[j] + eps) / (1.0 + opportunity_cost)
    
    d_j0 = d[0, :]
    eff_cost = d + d_j0[None, :]
    dist_penalty = 1.0 / (1.0 + eff_cost / max_len)
    
    eta = eta * dist_penalty
    eta[:, 0] = eps
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    return eta

def _make_regret_candidate(prizes, distance_matrix, max_len, variant=1, **kwargs):
    """Generate a regret-based heuristic candidate."""
    n = len(prizes)
    eps = 1e-9
    
    if variant == 1:
        eta = _compute_regret_heuristic(prizes, distance_matrix, max_len, **kwargs)
    elif variant == 2:
        eta = _compute_regret_heuristic_v2(prizes, distance_matrix, max_len, **kwargs)
    elif variant == 3:
        eta = _compute_regret_heuristic_v3(prizes, distance_matrix, max_len, **kwargs)
    else:
        eta = np.ones((n, n))
    
    eta[:, 0] = eps
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    return eta

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 295.0
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    # Generate regret-based heuristic candidates
    candidates = []
    
    # Variant 1: Regret based on max prize of reachable alternatives
    for bf in [0.3, 0.5, 0.7, 0.9]:
        cand = _make_regret_candidate(p, d, max_len, variant=1, budget_frac=bf)
        if cand.shape == (n, n) and np.all(np.isfinite(cand)):
            candidates.append(cand)
    
    # Variant 2: Regret based on count of higher-prize neighbors
    for kn in [5, 10, 15, 20]:
        cand = _make_regret_candidate(p, d, max_len, variant=2, k_neighbors=kn)
        if cand.shape == (n, n) and np.all(np.isfinite(cand)):
            candidates.append(cand)
    
    # Variant 3: Regret based on sum of prizes of closer reachable nodes
    for tf in [0.4, 0.6, 0.8, 1.0]:
        cand = _make_regret_candidate(p, d, max_len, variant=3, threshold_frac=tf)
        if cand.shape == (n, n) and np.all(np.isfinite(cand)):
            candidates.append(cand)
    
    # Classic prize/cost heuristic as baseline
    d_j0 = d[0, :]
    eff_cost = d + d_j0[None, :] + eps
    for g in [1.0, 2.0, 3.0]:
        cand = (p[None, :] / eff_cost) ** g
        cand[:, 0] = eps
        cand = np.where(np.isfinite(cand) & (cand > 0), cand, eps)
        candidates.append(cand)
    
    if not candidates:
        return np.ones((n, n))
    
    # Phase 1: Quick screening
    quick_results = []
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 60:
            break
        score = _run_aco_quick(p, d, max_len, cand, num_ants=10, num_iters=15, seed=0)
        quick_results.append((score, i))
    
    quick_results.sort(reverse=True)
    top_candidates = [candidates[i] for _, i in quick_results[:10]]
    
    # Phase 2: Full evaluation of top candidates
    best_score = -1
    best_heuristic = top_candidates[0] if top_candidates else candidates[0]
    
    for cand in top_candidates:
        if time.time() > deadline - 30:
            break
        score = _run_aco_fast(p, d, max_len, cand, num_ants=20, num_iters=30, seed=0)
        if score > best_score:
            best_score = score
            best_heuristic = cand
    
    # Phase 3: Fine-tune
    if time.time() < deadline - 20:
        best_base = best_heuristic.copy()
        best_base[best_base <= 0] = eps
        
        for scale in [0.5, 0.75, 1.25, 1.5]:
            if time.time() > deadline - 15:
                break
            cand = best_base * scale
            if cand.shape == (n, n) and np.all(np.isfinite(cand)):
                score = _run_aco_fast(p, d, max_len, cand, num_ants=20, num_iters=30, seed=0)
                if score > best_score:
                    best_score = score
                    best_heuristic = cand
    
    # Final cleanup
    best_heuristic = best_heuristic.copy()
    best_heuristic[best_heuristic <= 0] = eps
    
    return best_heuristic
