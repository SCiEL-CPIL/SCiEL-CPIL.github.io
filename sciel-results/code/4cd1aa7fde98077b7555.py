"""Structural change: detour-based heuristic with adaptive ACO evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 325.0
    
    N = len(prizes)
    prizes_arr = np.asarray(prizes, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)
    d_j0 = dist[:, 0].copy()
    
    # Precompute: for edge i->j, the detour cost is d[i,j] + d[j,0]
    # The "efficiency" of visiting j from i is prize[j] / (d[i,j] + d[j,0])
    # But we also want to penalize long detours more heavily
    
    def run_aco(eta, seed, n_ants=20, n_iters=30):
        """Fast ACO simulation."""
        rng = np.random.RandomState(seed)
        tau = np.ones((N, N))
        best_global = 0.0
        
        for it in range(n_iters):
            best_iter = 0.0
            
            for ant in range(n_ants):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                
                while True:
                    remaining = ~visited
                    threshold = max_len - path_len
                    # Eligibility: d[current, j] + d[j, 0] <= threshold
                    eligible = remaining & (dist[current, :] + d_j0 <= threshold + 1e-12)
                    eligible_idx = np.where(eligible)[0]
                    
                    if len(eligible_idx) == 0:
                        break
                    
                    weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                    total_w = weights.sum()
                    if total_w <= 0:
                        break
                    
                    probs = weights / total_w
                    r = rng.random_sample()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible_idx):
                        idx = len(eligible_idx) - 1
                    
                    next_node = eligible_idx[idx]
                    path_len += dist[current, next_node]
                    prize_total += prizes_arr[next_node]
                    visited[next_node] = True
                    current = next_node
                
                if prize_total > best_iter:
                    best_iter = prize_total
            
            if best_iter > best_global:
                best_global = best_iter
            
            tau = tau * 0.9
            tau += best_iter / 10.0
        
        return best_global
    
    # New structural approach: detour-aware heuristic
    # The key insight: when at node i, moving to j costs d[i,j] in path length
    # and then we must return from j to 0, costing d[j,0].
    # The "detour penalty" for going to j from i is d[i,j] + d[j,0].
    # But we can also think of it as: the marginal cost of adding j to the path
    # when currently at i is d[i,j], and the opportunity cost is d[j,0] vs d[i,0].
    
    # Heuristic family 1: prize / (detour_cost)^alpha
    # Heuristic family 2: prize * exp(-beta * detour_cost)
    # Heuristic family 3: prize^gamma / (d[i,j]^alpha + d[j,0]^beta)
    # Heuristic family 4: prize / (d[i,j] + d[j,0] - d[i,0])^alpha  (relative detour)
    
    def make_detour(alpha, gamma):
        """prize^gamma / (d[i,j] + d[j,0])^alpha"""
        detour = dist + d_j0[None, :]
        detour = np.maximum(detour, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], gamma) / np.power(detour, alpha)
        return np.where(eta > 0, eta, 1e-9)
    
    def make_exp_detour(beta, gamma):
        """prize^gamma * exp(-beta * (d[i,j] + d[j,0]))"""
        detour = dist + d_j0[None, :]
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], gamma) * np.exp(-beta * np.maximum(detour, 0))
        return np.where(eta > 0, eta, 1e-9)
    
    def make_relative_detour(alpha, gamma):
        """prize^gamma / (d[i,j] + d[j,0] - d[i,0] + eps)^alpha"""
        # Relative detour: how much extra distance does j add compared to staying at i?
        rel_detour = dist + d_j0[None, :] - d_j0[:, None]
        rel_detour = np.maximum(rel_detour, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], gamma) / np.power(rel_detour, alpha)
        return np.where(eta > 0, eta, 1e-9)
    
    def make_mixed(alpha, beta, gamma):
        """prize^gamma / (d[i,j]^alpha + d[j,0]^beta)"""
        denom = np.power(np.maximum(dist, 1e-12), alpha) + np.power(np.maximum(d_j0, 1e-12)[None, :], beta)
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], gamma) / denom
        return np.where(eta > 0, eta, 1e-9)
    
    def make_hybrid(alpha, beta, gamma, delta):
        """prize^gamma * (1/(d[i,j]+d[j,0])^alpha) * exp(-beta*d[i,j]) * (1/(1+delta*d[j,0]))"""
        detour = dist + d_j0[None, :]
        eta = (np.power(np.maximum(prizes_arr, 1e-12)[None, :], gamma) *
               np.power(np.maximum(detour, 1e-12), -alpha) *
               np.exp(-beta * np.maximum(dist, 0)) /
               (1 + delta * np.maximum(d_j0, 0))[None, :])
        return np.where(eta > 0, eta, 1e-9)
    
    # Generate candidates from multiple families
    candidates = []
    
    # Family 1: simple detour
    for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        for gamma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            candidates.append(('detour', alpha, gamma))
    
    # Family 2: exponential detour
    for beta in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        for gamma in [0.5, 1.0, 1.5, 2.0, 2.5]:
            candidates.append(('exp_detour', beta, gamma))
    
    # Family 3: relative detour
    for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for gamma in [0.5, 1.0, 1.5, 2.0, 2.5]:
            candidates.append(('rel_detour', alpha, gamma))
    
    # Family 4: mixed powers
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for beta in [0.5, 1.0, 1.5, 2.0]:
            for gamma in [0.5, 1.0, 1.5, 2.0]:
                candidates.append(('mixed', alpha, beta, gamma))
    
    # Family 5: hybrid
    for alpha in [0.5, 1.0, 1.5]:
        for beta in [0.5, 1.0, 2.0]:
            for gamma in [0.5, 1.0, 1.5]:
                for delta in [0.5, 1.0, 2.0]:
                    candidates.append(('hybrid', alpha, beta, gamma, delta))
    
    def get_eta(cand):
        ctype = cand[0]
        if ctype == 'detour':
            _, alpha, gamma = cand
            return make_detour(alpha, gamma)
        elif ctype == 'exp_detour':
            _, beta, gamma = cand
            return make_exp_detour(beta, gamma)
        elif ctype == 'rel_detour':
            _, alpha, gamma = cand
            return make_relative_detour(alpha, gamma)
        elif ctype == 'mixed':
            _, alpha, beta, gamma = cand
            return make_mixed(alpha, beta, gamma)
        elif ctype == 'hybrid':
            _, alpha, beta, gamma, delta = cand
            return make_hybrid(alpha, beta, gamma, delta)
        else:
            return make_detour(1.0, 1.0)
    
    # Phase 1: Quick screen with seed 0
    quick_results = []
    
    for cand in candidates:
        if time.time() - start_time > time_limit - 120:
            break
        eta = get_eta(cand)
        s = run_aco(eta, 0)
        quick_results.append((s, cand, eta))
    
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Evaluate top candidates with multiple seeds
    top_k = min(20, len(quick_results))
    best_score = -1
    best_eta = quick_results[0][2] if quick_results else make_detour(1.0, 1.0)
    best_cand = quick_results[0][1] if quick_results else ('detour', 1.0, 1.0)
    
    for s_quick, cand, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 40:
            break
        scores = [s_quick]
        for seed in [1, 2, 3, 4, 5]:
            if time.time() - start_time > time_limit - 35:
                break
            s = run_aco(eta, seed)
            scores.append(s)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_eta = eta
            best_cand = cand
    
    # Phase 3: Local refinement around best candidate
    if time.time() - start_time < time_limit - 45:
        best_type = best_cand[0]
        refinement_candidates = []
        
        if best_type == 'detour':
            _, balpha, bgamma = best_cand
            for dalpha in [-0.2, -0.1, 0.1, 0.2, 0.3, 0.5]:
                for dgamma in [-0.2, -0.1, 0.1, 0.2, 0.3, 0.5]:
                    na = balpha + dalpha
                    ng = bgamma + dgamma
                    if na > 0.1 and ng > 0.1:
                        refinement_candidates.append(('detour', na, ng))
        
        elif best_type == 'exp_detour':
            _, bbeta, bgamma = best_cand
            for dbeta in [-0.3, -0.1, 0.1, 0.3, 0.5, 1.0]:
                for dgamma in [-0.2, -0.1, 0.1, 0.2, 0.3, 0.5]:
                    nb = bbeta + dbeta
                    ng = bgamma + dgamma
                    if nb > 0.1 and ng > 0.1:
                        refinement_candidates.append(('exp_detour', nb, ng))
        
        elif best_type == 'rel_detour':
            _, balpha, bgamma = best_cand
            for dalpha in [-0.2, -0.1, 0.1, 0.2, 0.3, 0.5]:
                for dgamma in [-0.2, -0.1, 0.1, 0.2, 0.3, 0.5]:
                    na = balpha + dalpha
                    ng = bgamma + dgamma
                    if na > 0.1 and ng > 0.1:
                        refinement_candidates.append(('rel_detour', na, ng))
        
        elif best_type == 'mixed':
            _, balpha, bbeta, bgamma = best_cand
            for dalpha in [-0.2, -0.1, 0.1, 0.2, 0.3]:
                for dbeta in [-0.2, -0.1, 0.1, 0.2, 0.3]:
                    for dgamma in [-0.2, -0.1, 0.1, 0.2, 0.3]:
                        na = balpha + dalpha
                        nb = bbeta + dbeta
                        ng = bgamma + dgamma
                        if na > 0.1 and nb > 0.1 and ng > 0.1:
                            refinement_candidates.append(('mixed', na, nb, ng))
        
        elif best_type == 'hybrid':
            _, balpha, bbeta, bgamma, bdelta = best_cand
            for dalpha in [-0.2, -0.1, 0.1, 0.2]:
                for dbeta in [-0.2, -0.1, 0.1, 0.2]:
                    for dgamma in [-0.2, -0.1, 0.1, 0.2]:
                        for ddelta in [-0.2, -0.1, 0.1, 0.2]:
                            na = balpha + dalpha
                            nb = bbeta + dbeta
                            ng = bgamma + dgamma
                            nd = bdelta + ddelta
                            if na > 0.1 and nb > 0.01 and ng > 0.1 and nd > 0.01:
                                refinement_candidates.append(('hybrid', na, nb, ng, nd))
        
        for cand in refinement_candidates:
            if time.time() - start_time > time_limit - 25:
                break
            eta = get_eta(cand)
            s = run_aco(eta, 0)
            if s > best_score * 0.9:
                scores = [s]
                for seed in [1, 2, 3]:
                    if time.time() - start_time > time_limit - 20:
                        break
                    s2 = run_aco(eta, seed)
                    scores.append(s2)
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_eta = eta
    
    # Final safety check
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != dist.shape:
        detour = dist + d_j0[None, :]
        detour = np.maximum(detour, 1e-12)
        best_eta = prizes_arr[None, :] / detour
        best_eta = np.where(best_eta > 0, best_eta, 1e-9)
    
    best_eta = np.where(best_eta > 0, best_eta, 1e-9)
    
    return best_eta

# EVOLVE-BLOCK-END
