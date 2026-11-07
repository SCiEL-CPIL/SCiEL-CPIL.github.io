"""Improved heuristic search: richer family, faster ACO, adaptive multi-phase optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 326.0  # Safety margin (evaluator kills at 330s)
    
    N = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)
    
    # Precompute distances from each node to node 0
    d_j0 = dist[:, 0].copy()
    
    # Precompute total cost for eligibility checks
    total_cost = dist + d_j0[None, :]  # (N, N): d[i,j] + d[j,0]
    
    # Precompute slack: remaining budget after round-trip via j from 0
    slack = np.maximum(max_len - dist[0, :] - d_j0, 0.0)  # (N,)
    
    # Precompute log-prizes for stability
    log_prizes = np.log(np.maximum(prizes, 1e-12))
    
    def run_aco_fast(eta, seed, n_ants=20, n_iters=30):
        rng = np.random.RandomState(seed)
        tau = np.ones((N, N))
        
        best_prize_global = 0.0
        
        for it in range(n_iters):
            best_prize_iter = 0.0
            
            for ant in range(n_ants):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                
                while True:
                    remaining = max_len - path_len
                    eligible_mask = (~visited) & (total_cost[current, :] <= remaining)
                    eligible = np.where(eligible_mask)[0]
                    
                    if len(eligible) == 0:
                        break
                    
                    probs = tau[current, eligible] * eta[current, eligible]
                    total_prob = probs.sum()
                    if total_prob <= 0:
                        break
                    
                    probs = probs / total_prob
                    r = rng.random_sample()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible):
                        idx = len(eligible) - 1
                    
                    next_node = eligible[idx]
                    path_len += dist[current, next_node]
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
    
    def make_heuristic(params):
        """
        params: dict with keys:
          - 'p': prize exponent
          - 'q': d_ij exponent  
          - 'r': d_j0 exponent
          - 's': overall denominator exponent
          - 'alpha': weight on d_j0 in denominator
          - 'slack_w': slack weight
          - 'use_log': whether to use log-prizes
          - 'gamma': additional penalty for far nodes
        """
        p = params.get('p', 1.0)
        q = params.get('q', 1.0)
        r = params.get('r', 1.0)
        s = params.get('s', 1.0)
        alpha = params.get('alpha', 1.0)
        slack_w = params.get('slack_w', 0.0)
        use_log = params.get('use_log', False)
        gamma = params.get('gamma', 0.0)
        
        # Prize component
        if use_log:
            prize_comp = np.exp(p * log_prizes)[None, :]
        else:
            prize_comp = np.power(np.maximum(prizes, 1e-12)[None, :], p)
        
        # Distance denominator
        d_ij_safe = np.maximum(dist, 1e-12)
        d_j0_safe = np.maximum(d_j0, 1e-12)[None, :]
        
        denom_base = np.power(d_ij_safe, q) + alpha * np.power(d_j0_safe, r)
        denom_base = np.maximum(denom_base, 1e-12)
        
        # Slack component
        if slack_w > 0:
            slack_safe = np.maximum(slack, 1e-12)
            slack_comp = np.power(slack_safe[None, :], slack_w)
        else:
            slack_comp = np.ones((1, N))
        
        # Gamma penalty for far nodes
        if gamma > 0:
            max_d = np.max(dist)
            gamma_comp = np.exp(-gamma * dist / max_d)
        else:
            gamma_comp = np.ones((N, N))
        
        eta = prize_comp * slack_comp * gamma_comp / np.power(denom_base, s)
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    # Phase 1: Grid search over a focused parameter space
    candidates = []
    
    # Core grid: vary p, q, r, s, alpha
    for p in [0.5, 1.0, 1.5, 2.0, 2.5]:
        for q in [0.5, 1.0, 1.5, 2.0]:
            for r in [0.5, 1.0, 1.5, 2.0]:
                for s in [1.0, 1.5, 2.0, 2.5]:
                    for alpha in [0.5, 1.0, 1.5, 2.0]:
                        candidates.append({
                            'p': p, 'q': q, 'r': r, 's': s,
                            'alpha': alpha, 'slack_w': 0.0,
                            'use_log': False, 'gamma': 0.0
                        })
    
    # Add slack-weighted variants
    for p in [1.0, 1.5, 2.0]:
        for slack_w in [0.5, 1.0, 1.5, 2.0]:
            for s in [1.0, 1.5, 2.0]:
                candidates.append({
                    'p': p, 'q': 1.0, 'r': 1.0, 's': s,
                    'alpha': 1.0, 'slack_w': slack_w,
                    'use_log': False, 'gamma': 0.0
                })
    
    # Add log-prize variants
    for p in [0.5, 1.0, 1.5]:
        for q in [1.0, 1.5, 2.0]:
            for s in [1.0, 1.5, 2.0]:
                candidates.append({
                    'p': p, 'q': q, 'r': 1.0, 's': s,
                    'alpha': 1.0, 'slack_w': 0.0,
                    'use_log': True, 'gamma': 0.0
                })
    
    # Add gamma-penalty variants
    for p in [1.0, 1.5, 2.0]:
        for gamma in [0.5, 1.0, 2.0]:
            for s in [1.0, 1.5]:
                candidates.append({
                    'p': p, 'q': 1.0, 'r': 1.0, 's': s,
                    'alpha': 1.0, 'slack_w': 0.0,
                    'use_log': False, 'gamma': gamma
                })
    
    # Phase 1: Quick screen with seed 0
    quick_results = []
    n_evaluated = 0
    for cand in candidates:
        if time.time() - start_time > time_limit - 80:
            break
        eta = make_heuristic(cand)
        score = run_aco_fast(eta, 0)
        quick_results.append((score, cand, eta))
        n_evaluated += 1
    
    # Sort by quick score descending
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Evaluate top candidates with multiple seeds
    top_k = min(10, len(quick_results))
    best_score = -1
    best_eta = quick_results[0][2] if quick_results else make_heuristic({'p': 1.0, 'q': 1.0, 'r': 1.0, 's': 1.0, 'alpha': 1.0})
    best_cand = quick_results[0][1] if quick_results else {'p': 1.0, 'q': 1.0, 'r': 1.0, 's': 1.0, 'alpha': 1.0}
    
    for s_quick, cand, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 30:
            break
        scores = [s_quick]
        for seed in [1, 2, 3, 4, 5]:
            if time.time() - start_time > time_limit - 25:
                break
            s = run_aco_fast(eta, seed)
            scores.append(s)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_eta = eta
            best_cand = cand
    
    # Phase 3: Local refinement around best candidate
    if time.time() - start_time < time_limit - 35:
        refinement_candidates = []
        bp = best_cand.get('p', 1.0)
        bq = best_cand.get('q', 1.0)
        br = best_cand.get('r', 1.0)
        bs = best_cand.get('s', 1.0)
        balpha = best_cand.get('alpha', 1.0)
        bslack = best_cand.get('slack_w', 0.0)
        blog = best_cand.get('use_log', False)
        bgamma = best_cand.get('gamma', 0.0)
        
        # Perturb each parameter
        for dp in [-0.15, 0, 0.15]:
            for dq in [-0.15, 0, 0.15]:
                for dr in [-0.15, 0, 0.15]:
                    for ds in [-0.15, 0, 0.15]:
                        for dalpha in [-0.2, 0, 0.2]:
                            np_val = bp + dp
                            nq_val = bq + dq
                            nr_val = br + dr
                            ns_val = bs + ds
                            nalpha = balpha + dalpha
                            if np_val > 0.1 and nq_val > 0.1 and nr_val > 0.1 and ns_val > 0.5 and nalpha > 0.1:
                                refinement_candidates.append({
                                    'p': np_val, 'q': nq_val, 'r': nr_val, 's': ns_val,
                                    'alpha': nalpha, 'slack_w': bslack,
                                    'use_log': blog, 'gamma': bgamma
                                })
        
        # Also try adding slack if not present
        if bslack == 0:
            for sw in [0.5, 1.0, 1.5]:
                refinement_candidates.append({
                    'p': bp, 'q': bq, 'r': br, 's': bs,
                    'alpha': balpha, 'slack_w': sw,
                    'use_log': blog, 'gamma': bgamma
                })
        
        for cand in refinement_candidates:
            if time.time() - start_time > time_limit - 25:
                break
            eta = make_heuristic(cand)
            s = run_aco_fast(eta, 0)
            if s > best_score * 0.95:  # Only pursue promising candidates
                scores = [s]
                for seed in [1, 2, 3]:
                    if time.time() - start_time > time_limit - 20:
                        break
                    s2 = run_aco_fast(eta, seed)
                    scores.append(s2)
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_eta = eta
                    best_cand = cand
    
    # Phase 4: Final diverse candidates
    if time.time() - start_time < time_limit - 30:
        diverse = [
            {'p': 2.0, 'q': 0.5, 'r': 1.0, 's': 1.5, 'alpha': 1.0, 'slack_w': 1.0, 'use_log': False, 'gamma': 0.0},
            {'p': 1.5, 'q': 1.0, 'r': 0.5, 's': 2.0, 'alpha': 1.5, 'slack_w': 0.5, 'use_log': True, 'gamma': 0.0},
            {'p': 2.5, 'q': 1.5, 'r': 1.5, 's': 1.0, 'alpha': 0.5, 'slack_w': 0.0, 'use_log': False, 'gamma': 1.0},
            {'p': 1.0, 'q': 0.5, 'r': 0.5, 's': 2.5, 'alpha': 2.0, 'slack_w': 1.5, 'use_log': False, 'gamma': 0.5},
            {'p': 3.0, 'q': 1.0, 'r': 1.0, 's': 1.5, 'alpha': 1.0, 'slack_w': 0.0, 'use_log': True, 'gamma': 0.0},
            {'p': 1.5, 'q': 2.0, 'r': 2.0, 's': 1.0, 'alpha': 0.5, 'slack_w': 1.0, 'use_log': False, 'gamma': 0.0},
        ]
        for cand in diverse:
            if time.time() - start_time > time_limit - 25:
                break
            eta = make_heuristic(cand)
            s = run_aco_fast(eta, 0)
            if s > best_score:
                scores = [s]
                for seed in [1, 2]:
                    if time.time() - start_time > time_limit - 20:
                        break
                    s2 = run_aco_fast(eta, seed)
                    scores.append(s2)
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_eta = eta
    
    # Final safety check
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != dist.shape:
        denom = dist + d_j0[None, :]
        denom = np.maximum(denom, 1e-12)
        best_eta = prizes[None, :] / denom
        best_eta = np.where(best_eta > 0, best_eta, 1e-9)
    
    # Ensure no non-positive values
    best_eta = np.where(best_eta > 0, best_eta, 1e-9)
    
    return best_eta

# EVOLVE-BLOCK-END
