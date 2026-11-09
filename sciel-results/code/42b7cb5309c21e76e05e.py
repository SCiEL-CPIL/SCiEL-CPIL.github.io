import numpy as np
import time

def _run_aco(prizes, dist, max_len, heuristic_matrix, seed, n_ants=20, n_iters=30):
    """Simulate ACO with given heuristic, return best prize found."""
    n = len(prizes)
    rng = np.random.RandomState(seed)
    pheromone = np.ones((n, n))
    best_global = 0.0
    
    for _ in range(n_iters):
        best_iter = 0.0
        for _ in range(n_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize_sum = prizes[0]
            
            while True:
                # Find candidates
                candidates = []
                probs = []
                for j in range(1, n):
                    if not visited[j]:
                        if length + dist[current, j] + dist[j, 0] <= max_len:
                            candidates.append(j)
                            h = heuristic_matrix[current, j]
                            if not np.isfinite(h) or h <= 0:
                                h = 1e-9
                            p = pheromone[current, j] * h
                            probs.append(p)
                
                if not candidates:
                    break
                
                total_p = sum(probs)
                if total_p <= 0:
                    break
                
                probs_arr = np.array(probs)
                probs_norm = probs_arr / total_p
                r = rng.rand()
                cumsum = np.cumsum(probs_norm)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(candidates):
                    idx = len(candidates) - 1
                
                next_node = candidates[idx]
                length += dist[current, next_node]
                visited[next_node] = True
                prize_sum += prizes[next_node]
                current = next_node
        
        best_iter = max(best_iter, prize_sum)
        pheromone = pheromone * 0.9 + best_iter / 10.0
        best_global = max(best_global, best_iter)
    
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    n = len(prizes)
    d_j0 = distance_matrix[:, 0]
    
    # Define candidate heuristic generators
    def gen_prize_over_eff_dist(p, d, ml):
        """Prize / (d_ij + d_j0) - accounts for return cost"""
        eff = d + d_j0[None, :]
        return np.maximum(p[None, :] / (eff + 1e-9), 1e-9)
    
    def gen_prize_over_dist(p, d, ml):
        """Prize / d_ij - simple distance penalty"""
        return np.maximum(p[None, :] / (d + 1e-9), 1e-9)
    
    def gen_prize_exp_decay(p, d, ml):
        """Prize * exp(-d_ij) - exponential decay"""
        return np.maximum(p[None, :] * np.exp(-(d + 1e-9)), 1e-9)
    
    def gen_prize_over_total(p, d, ml):
        """Prize / (d_ij + d_0j + d_0i) - total trip cost approximation"""
        d_0 = d[0, :]
        total = d + d_0[None, :] + d_0[:, None]
        return np.maximum(p[None, :] / (total + 1e-9), 1e-9)
    
    def gen_prize_sq_dist(p, d, ml):
        """Prize / d_ij^2 - sharper distance penalty"""
        return np.maximum(p[None, :] / ((d + 1e-9) ** 2), 1e-9)
    
    def gen_combined(p, d, ml):
        """Blend: 0.5 * prize/eff_dist + 0.3 * prize/dist + 0.2 * prize*exp(-dist)"""
        h1 = gen_prize_over_eff_dist(p, d, ml)
        h2 = gen_prize_over_dist(p, d, ml)
        h3 = gen_prize_exp_decay(p, d, ml)
        # Normalize each to [0,1] range for blending
        def norm(h):
            mn, mx = h.min(), h.max()
            if mx - mn < 1e-12:
                return h
            return (h - mn) / (mx - mn)
        return np.maximum(0.5 * norm(h1) + 0.3 * norm(h2) + 0.2 * norm(h3) + 1e-9, 1e-9)
    
    candidates = [
        gen_prize_over_eff_dist,
        gen_prize_over_dist,
        gen_prize_exp_decay,
        gen_prize_over_total,
        gen_prize_sq_dist,
        gen_combined,
    ]
    
    # Evaluate each candidate using a quick ACO simulation with a fixed seed
    test_seed = 42
    best_score = -np.inf
    best_matrix = None
    
    for gen in candidates:
        if time.time() > deadline - 5:
            break
        
        try:
            h_mat = gen(prizes, distance_matrix, max_len)
            # Sanitize
            h_mat = np.where(np.isfinite(h_mat) & (h_mat > 0), h_mat, 1e-9)
            
            # Quick evaluation with reduced iterations for speed
            score = _run_aco(prizes, distance_matrix, max_len, h_mat, test_seed, 
                            n_ants=10, n_iters=10)
            
            if score > best_score:
                best_score = score
                best_matrix = h_mat
        except Exception:
            continue
    
    if best_matrix is None:
        # Fallback
        best_matrix = gen_prize_over_eff_dist(prizes, distance_matrix, max_len)
        best_matrix = np.where(np.isfinite(best_matrix) & (best_matrix > 0), best_matrix, 1e-9)
    
    return best_matrix
