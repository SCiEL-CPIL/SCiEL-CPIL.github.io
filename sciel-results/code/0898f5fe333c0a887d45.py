import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, eta, seed):
    """Run the fixed ACO protocol and return best prize over all iterations."""
    np.random.seed(seed)
    N = len(prizes)
    tau = np.ones((N, N))
    best_prize_global = 0.0
    d_j0 = distance_matrix[:, 0]
    
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

def _make_opportunity_cost_heuristic(prizes, distance_matrix, max_len, 
                                       prize_exp=1.0, cost_exp=1.0, slack_exp=1.0,
                                       base_cost=0.1):
    """
    Opportunity cost heuristic that models the trade-off between prize value,
    travel cost, and remaining budget slack.
    
    eta[i,j] = prize[j]^a / (cost[i,j] + base)^b * slack[i,j]^c
    
    where:
    - cost[i,j] = d[i,j] + d[j,0] (total cost of visiting j from i and returning)
    - slack[i,j] = max(0, 1 - cost[i,j]/max_len) (fraction of budget remaining)
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    
    # Total cost of going i -> j -> 0
    C = distance_matrix + d_j0[None, :]
    C_safe = np.maximum(C, 1e-9)
    
    # Slack: fraction of budget remaining after visiting j
    slack = np.maximum(1.0 - C / max_len, 0.0)
    slack_safe = np.maximum(slack, 1e-9)
    
    # Base heuristic: prize^a / (cost + base)^b * slack^c
    prize_term = np.power(prizes_safe[None, :], prize_exp)
    cost_term = np.power(C_safe + base_cost, cost_exp)
    slack_term = np.power(slack_safe, slack_exp)
    
    eta = prize_term / cost_term * slack_term
    
    # Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta

def _make_rank_slack_heuristic(prizes, distance_matrix, max_len, 
                                 prize_rank_weight=1.0, slack_weight=1.0):
    """
    Rank-based heuristic combining prize rank and slack.
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    C = distance_matrix + d_j0[None, :]
    C_safe = np.maximum(C, 1e-9)
    
    # Prize ranks (higher is better)
    prize_ranks = np.argsort(np.argsort(prizes_safe))
    
    # Slack: how much budget remains
    slack = np.maximum(1.0 - C / max_len, 0.0)
    slack_safe = np.maximum(slack, 1e-9)
    
    # Combine: high prize rank * high slack
    eta = np.power(prize_ranks[None, :] + 1, prize_rank_weight) * np.power(slack_safe, slack_weight)
    
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta

def _make_hybrid_heuristic(prizes, distance_matrix, max_len, 
                            prize_exp=1.0, cost_exp=1.0, slack_exp=1.0,
                            base_cost=0.1, rank_weight=0.5):
    """
    Hybrid heuristic combining opportunity cost and rank-based approaches.
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    C = distance_matrix + d_j0[None, :]
    C_safe = np.maximum(C, 1e-9)
    
    slack = np.maximum(1.0 - C / max_len, 0.0)
    slack_safe = np.maximum(slack, 1e-9)
    
    prize_ranks = np.argsort(np.argsort(prizes_safe))
    
    # Opportunity cost component
    prize_term = np.power(prizes_safe[None, :], prize_exp)
    cost_term = np.power(C_safe + base_cost, cost_exp)
    slack_term = np.power(slack_safe, slack_exp)
    oc_component = prize_term / cost_term * slack_term
    
    # Rank component
    rank_component = np.power(prize_ranks[None, :] + 1, 1.0) * np.power(slack_safe, 1.0)
    
    # Combine
    eta = (1 - rank_weight) * oc_component + rank_weight * rank_component
    
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 320.0  # Safety margin before 330s kill
    
    N = len(prizes)
    
    # Generate candidate heuristics with a focused grid
    candidates = []
    
    # Opportunity cost heuristics with varying exponents
    for prize_exp in [0.5, 1.0, 1.5, 2.0]:
        for cost_exp in [0.5, 1.0, 1.5, 2.0]:
            for slack_exp in [0.5, 1.0, 1.5, 2.0, 3.0]:
                for base_cost in [0.05, 0.1, 0.2, 0.5]:
                    candidates.append(('oc', prize_exp, cost_exp, slack_exp, base_cost))
    
    # Rank-slack heuristics
    for prize_rank_weight in [0.5, 1.0, 1.5, 2.0]:
        for slack_weight in [0.5, 1.0, 1.5, 2.0]:
            candidates.append(('rs', prize_rank_weight, 0.0, slack_weight, 0.0))
    
    # Hybrid heuristics
    for prize_exp in [1.0, 1.5]:
        for cost_exp in [1.0, 1.5]:
            for slack_exp in [1.0, 1.5, 2.0]:
                for base_cost in [0.1, 0.2]:
                    for rank_weight in [0.3, 0.5, 0.7]:
                        candidates.append(('hybrid', prize_exp, cost_exp, slack_exp, base_cost, rank_weight))
    
    # Quick screening with seed 0
    quick_results = []
    for c in candidates:
        if time.time() - start_time > time_limit - 80:
            break
        
        if c[0] == 'oc':
            _, prize_exp, cost_exp, slack_exp, base_cost = c
            eta = _make_opportunity_cost_heuristic(prizes, distance_matrix, max_len,
                                                    prize_exp, cost_exp, slack_exp, base_cost)
        elif c[0] == 'rs':
            _, prize_rank_weight, _, slack_weight, _ = c
            eta = _make_rank_slack_heuristic(prizes, distance_matrix, max_len,
                                              prize_rank_weight, slack_weight)
        elif c[0] == 'hybrid':
            _, prize_exp, cost_exp, slack_exp, base_cost, rank_weight = c
            eta = _make_hybrid_heuristic(prizes, distance_matrix, max_len,
                                          prize_exp, cost_exp, slack_exp, base_cost, rank_weight)
        else:
            eta = np.ones((N, N))
        
        s = _simulate_aco(prizes, distance_matrix, max_len, eta, 0)
        quick_results.append((s, c, eta))
    
    if not quick_results:
        return _make_opportunity_cost_heuristic(prizes, distance_matrix, max_len, 1.0, 1.0, 1.0, 0.1)
    
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Evaluate top candidates with multiple seeds
    best_avg = -1
    best_eta = None
    top_k = min(20, len(quick_results))
    
    for s_quick, c, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 30:
            break
        
        scores = [s_quick]
        for seed in [1, 2, 3]:
            if time.time() - start_time > time_limit - 25:
                break
            s = _simulate_aco(prizes, distance_matrix, max_len, eta, seed)
            scores.append(s)
        
        avg = np.mean(scores)
        if avg > best_avg:
            best_avg = avg
            best_eta = eta
    
    # Local refinement around the best if time permits
    if best_eta is not None and time.time() - start_time < time_limit - 40:
        # Determine which type was best and refine around it
        # Find the best candidate parameters
        best_c = None
        for s_quick, c, eta in quick_results[:top_k]:
            if np.array_equal(eta, best_eta):
                best_c = c
                break
        
        if best_c is not None:
            if best_c[0] == 'oc':
                _, pe, ce, se, bc = best_c
                # Refine around best opportunity cost parameters
                for pe2 in [pe - 0.2, pe, pe + 0.2]:
                    for ce2 in [ce - 0.2, ce, ce + 0.2]:
                        for se2 in [se - 0.3, se, se + 0.3]:
                            for bc2 in [bc * 0.8, bc, bc * 1.2]:
                                if time.time() - start_time > time_limit - 15:
                                    break
                                if pe2 <= 0 or ce2 <= 0 or se2 <= 0 or bc2 <= 0:
                                    continue
                                eta2 = _make_opportunity_cost_heuristic(prizes, distance_matrix, max_len,
                                                                         pe2, ce2, se2, bc2)
                                s2 = _simulate_aco(prizes, distance_matrix, max_len, eta2, 0)
                                if s2 > best_avg * 0.98:
                                    scores2 = [s2]
                                    for seed in [1, 2]:
                                        if time.time() - start_time > time_limit - 10:
                                            break
                                        s3 = _simulate_aco(prizes, distance_matrix, max_len, eta2, seed)
                                        scores2.append(s3)
                                    avg2 = np.mean(scores2)
                                    if avg2 > best_avg:
                                        best_avg = avg2
                                        best_eta = eta2
            elif best_c[0] == 'rs':
                _, prw, _, sw, _ = best_c
                for prw2 in [prw - 0.2, prw, prw + 0.2]:
                    for sw2 in [sw - 0.2, sw, sw + 0.2]:
                        if time.time() - start_time > time_limit - 15:
                            break
                        if prw2 <= 0 or sw2 <= 0:
                            continue
                        eta2 = _make_rank_slack_heuristic(prizes, distance_matrix, max_len, prw2, sw2)
                        s2 = _simulate_aco(prizes, distance_matrix, max_len, eta2, 0)
                        if s2 > best_avg * 0.98:
                            scores2 = [s2]
                            for seed in [1, 2]:
                                if time.time() - start_time > time_limit - 10:
                                    break
                                s3 = _simulate_aco(prizes, distance_matrix, max_len, eta2, seed)
                                scores2.append(s3)
                            avg2 = np.mean(scores2)
                            if avg2 > best_avg:
                                best_avg = avg2
                                best_eta = eta2
            elif best_c[0] == 'hybrid':
                _, pe, ce, se, bc, rw = best_c
                for pe2 in [pe - 0.2, pe, pe + 0.2]:
                    for ce2 in [ce - 0.2, ce, ce + 0.2]:
                        for se2 in [se - 0.3, se, se + 0.3]:
                            for rw2 in [max(0.1, rw - 0.1), rw, min(0.9, rw + 0.1)]:
                                if time.time() - start_time > time_limit - 15:
                                    break
                                if pe2 <= 0 or ce2 <= 0 or se2 <= 0:
                                    continue
                                eta2 = _make_hybrid_heuristic(prizes, distance_matrix, max_len,
                                                               pe2, ce2, se2, bc, rw2)
                                s2 = _simulate_aco(prizes, distance_matrix, max_len, eta2, 0)
                                if s2 > best_avg * 0.98:
                                    scores2 = [s2]
                                    for seed in [1, 2]:
                                        if time.time() - start_time > time_limit - 10:
                                            break
                                        s3 = _simulate_aco(prizes, distance_matrix, max_len, eta2, seed)
                                        scores2.append(s3)
                                    avg2 = np.mean(scores2)
                                    if avg2 > best_avg:
                                        best_avg = avg2
                                        best_eta = eta2
    
    # Final safety check
    if best_eta is None or not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = _make_opportunity_cost_heuristic(prizes, distance_matrix, max_len, 1.0, 1.0, 1.0, 0.1)
    
    return best_eta
