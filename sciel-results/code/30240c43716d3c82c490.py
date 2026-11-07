"""Heuristic matrix for ant colony optimization of orienteering - opportunity cost approach."""
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 320.0  # Safety margin before 330s kill
    
    N = len(prizes)
    d_j0 = distance_matrix[:, 0].copy()
    prizes_safe = np.maximum(prizes, 1e-9)
    
    def run_aco_fast(eta, seed):
        """Run the fixed ACO protocol and return best prize over all iterations."""
        np.random.seed(seed)
        tau = np.ones((N, N))
        best_prize_global = 0.0
        
        for it in range(30):
            best_prize_iter = 0.0
            
            for ant in range(20):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                
                while True:
                    remaining = max_len - path_len
                    costs = distance_matrix[current, :] + d_j0
                    eligible_mask = (~visited) & (costs <= remaining)
                    
                    if not np.any(eligible_mask):
                        break
                    
                    eligible_idx = np.where(eligible_mask)[0]
                    weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                    total_w = weights.sum()
                    if total_w <= 0:
                        break
                    
                    probs = weights / total_w
                    r = np.random.rand()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible_idx):
                        idx = len(eligible_idx) - 1
                    
                    next_node = eligible_idx[idx]
                    path_len += distance_matrix[current, next_node]
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

    def make_opportunity_cost_heuristic(alpha=1.0, beta=1.0, gamma=1.0):
        """
        Opportunity cost heuristic:
        For each (i, j):
          - cost_to_visit = d_ij + d_j0 (total length consumed by going to j and returning)
          - budget_fraction = cost_to_visit / max_len (fraction of budget consumed)
          - opportunity_cost = prizes[j] * (1 - budget_fraction)^gamma  [value preserved after visit]
          - Also penalize by how much of the budget is consumed:
            penalty = (budget_fraction)^beta
          - Combined: prizes[j]^alpha * (1 - budget_fraction)^gamma / (1 + penalty)
        
        Actually, let's think more carefully:
        - We want to prioritize nodes that give high prize while consuming little budget
        - The "opportunity cost" of visiting j is the budget we lose: d_ij + d_j0
        - So the attractiveness should be prize_j / (opportunity_cost)^alpha, but also
          we should consider how much budget remains after visiting j
        
        New formulation:
        - remaining_after = max_len - (d_ij + d_j0)
        - slack_ratio = remaining_after / max_len  (how much budget is left)
        - eta[i,j] = prizes[j]^alpha * slack_ratio^beta / (d_ij + d_j0)^gamma
        
        This rewards high prizes, good slack, and low cost.
        """
        p = np.maximum(prizes, 1e-9)[None, :]
        dij = np.maximum(distance_matrix, 1e-9)
        dj0 = np.maximum(d_j0, 1e-9)[None, :]
        
        cost = dij + dj0  # total cost to visit j from i and return to 0
        remaining = np.maximum(max_len - cost, 0.0)
        slack_ratio = remaining / max_len
        
        # Base: prize^alpha * slack^beta / cost^gamma
        eta = np.power(p, alpha) * np.power(slack_ratio, beta) / np.power(cost, gamma)
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def make_prize_cost_heuristic(alpha=1.0, beta=1.0):
        """
        Simple prize/cost heuristic with different exponent structure.
        eta[i,j] = prizes[j]^alpha / (d_ij^beta + d_j0^beta)
        """
        p = np.maximum(prizes, 1e-9)[None, :]
        dij = np.maximum(distance_matrix, 1e-9)
        dj0 = np.maximum(d_j0, 1e-9)[None, :]
        
        eta = np.power(p, alpha) / (np.power(dij, beta) + np.power(dj0, beta))
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def make_rank_based_heuristic(top_k=10):
        """
        Rank-based: for each node j, rank by prize. 
        For each (i,j), give high attractiveness if j has high prize and low cost.
        eta[i,j] = rank_prize[j] / (d_ij + d_j0)
        """
        p = np.maximum(prizes, 1e-9)
        # Rank prizes: highest prize gets rank N, lowest gets rank 1
        sorted_indices = np.argsort(p)
        ranks = np.zeros(N)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1  # 1 to N
        
        p_ranked = ranks[None, :]
        dij = np.maximum(distance_matrix, 1e-9)
        dj0 = np.maximum(d_j0, 1e-9)[None, :]
        
        eta = p_ranked / (dij + dj0)
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    # Candidate heuristics
    candidates = []
    
    # Opportunity cost family
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for beta in [0.5, 1.0, 1.5, 2.0]:
            for gamma in [0.5, 1.0, 1.5]:
                candidates.append(('oc', alpha, beta, gamma))
    
    # Prize/cost family
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for beta in [0.5, 1.0, 1.5, 2.0]:
            candidates.append(('pc', alpha, beta, 1.0))
    
    # Rank-based
    candidates.append(('rank', 1.0, 1.0, 1.0))
    
    # Phase 1: Quick screen with 1 seed
    quick_results = []
    for cand in candidates:
        if time.time() - start_time > time_limit - 60:
            break
        
        ctype, a, b, g = cand
        if ctype == 'oc':
            eta = make_opportunity_cost_heuristic(a, b, g)
        elif ctype == 'pc':
            eta = make_prize_cost_heuristic(a, b)
        else:
            eta = make_rank_based_heuristic()
        
        s = run_aco_fast(eta, 0)
        quick_results.append((s, cand, eta))
    
    if not quick_results:
        # Fallback
        eta = make_opportunity_cost_heuristic(1.0, 1.0, 1.0)
        return eta
    
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Evaluate top candidates with more seeds
    top_k = min(10, len(quick_results))
    best_avg = -1
    best_eta = quick_results[0][2]
    
    for s_quick, cand, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 20:
            break
        scores = [s_quick]
        for seed in [1, 2]:
            if time.time() - start_time > time_limit - 15:
                break
            s = run_aco_fast(eta, seed)
            scores.append(s)
        avg = np.mean(scores)
        if avg > best_avg:
            best_avg = avg
            best_eta = eta
    
    # Phase 3: Local refinement around best
    if time.time() - start_time < time_limit - 30:
        best_entry = quick_results[0]
        btype, ba, bb, bg = best_entry[1]
        
        # Generate nearby candidates
        nearby = []
        for da in [-0.25, -0.1, 0.1, 0.25]:
            for db in [-0.25, -0.1, 0.1, 0.25]:
                for dg in [-0.25, -0.1, 0.1, 0.25]:
                    na = ba + da
                    nb = bb + db
                    ng = bg + dg
                    if na > 0.1 and nb > 0.1 and ng > 0.1:
                        nearby.append((btype, na, nb, ng))
        
        seen = set()
        for cand in nearby:
            key = (cand[0], round(cand[1], 3), round(cand[2], 3), round(cand[3], 3))
            if key in seen:
                continue
            seen.add(key)
            
            if time.time() - start_time > time_limit - 10:
                break
            
            ctype, a, b, g = cand
            if ctype == 'oc':
                eta = make_opportunity_cost_heuristic(a, b, g)
            elif ctype == 'pc':
                eta = make_prize_cost_heuristic(a, b)
            else:
                continue
            
            s = run_aco_fast(eta, 0)
            if s > best_avg * 0.95:
                scores = [s]
                for seed in [1, 2]:
                    if time.time() - start_time > time_limit - 8:
                        break
                    s2 = run_aco_fast(eta, seed)
                    scores.append(s2)
                avg = np.mean(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_eta = eta
    
    # Final safety check
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = make_opportunity_cost_heuristic(1.0, 1.0, 1.0)
    
    return best_eta
