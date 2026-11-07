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

def _make_regret_heuristic(prizes, distance_matrix, max_len, alpha=1.0, beta=1.0):
    """
    Regret-based heuristic: entry (i, j) approximates the counterfactual loss
    if the ant at i does NOT visit j next. This is approximated by:
    prize[j] * P(j is the greedy-optimal next step from i)
    where P is computed via a soft-max over the "attractiveness" of each candidate.
    
    We define attractiveness of j from i as: prize[j] / (d[i,j] + d[j,0] + eps)
    Then P(j optimal) ~ exp(alpha * att_j) / sum_k exp(alpha * att_k)
    Regret(i,j) = prize[j] * P(j optimal) * (1 + beta * slack_j)
    where slack_j = max(0, max_len - d[i,j] - d[j,0]) / max_len
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    
    # Cost of going i -> j -> 0
    C = distance_matrix + d_j0[None, :]
    C_safe = np.maximum(C, 1e-9)
    
    # Attractiveness: prize / cost
    att = prizes_safe[None, :] / C_safe
    
    # Soft-max probability that j is the best choice from i
    # We use alpha to control "temperature"
    att_scaled = alpha * att
    # Subtract max for numerical stability
    att_max = np.max(att_scaled, axis=1, keepdims=True)
    exp_att = np.exp(att_scaled - att_max)
    P_opt = exp_att / np.sum(exp_att, axis=1, keepdims=True)
    
    # Slack: how much room is left if we take j
    slack = np.maximum(0.0, max_len - C) / max_len
    slack = np.maximum(slack, 1e-9)
    
    # Regret: prize * P_opt * (1 + beta * slack)
    eta = prizes_safe[None, :] * P_opt * (1.0 + beta * slack)
    
    # Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta

def _make_simple_heuristic(prizes, distance_matrix, max_len, type_id, p1, p2, p3):
    """Generate various simple heuristic types."""
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    C = distance_matrix + d_j0[None, :]
    C_safe = np.maximum(C, 1e-9)
    max_prize = np.max(prizes_safe)
    max_cost = np.max(C_safe)
    
    try:
        if type_id == 0:  # Power Law
            prize_norm = prizes_safe / (max_prize + 1e-9)
            cost_norm = C_safe / (max_cost + 1e-9)
            eta = np.power(prize_norm[None, :], p2) / np.power(cost_norm, p2)
        elif type_id == 1:  # Exponential
            prize_norm = prizes_safe / (max_prize + 1e-9)
            cost_norm = C_safe / (max_cost + 1e-9)
            eta = np.exp(p1 * prize_norm[None, :] - p2 * cost_norm)
        elif type_id == 2:  # Ratio with Slack
            slack = np.maximum(1.0 - C / max_len, 0.0)
            slack = np.maximum(slack, 1e-9)
            eta = (prizes_safe[None, :] / C_safe) * np.power(slack, p1)
        elif type_id == 3:  # Rank-based
            prize_ranks = np.argsort(np.argsort(prizes_safe))
            cost_ranks = np.argsort(np.argsort(C_safe))
            cost_ranks_inv = (N - 1) - cost_ranks
            eta = np.power(prize_ranks[None, :] + 1, p1) / np.power(cost_ranks_inv + 1, p2)
        elif type_id == 4:  # Hybrid
            eta = np.power(prizes_safe[None, :], p1) / (C_safe + p3)
        else:
            eta = np.ones((N, N))
    except Exception:
        eta = np.ones((N, N))
    
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 325.0
    
    N = len(prizes)
    
    # Generate candidate heuristics
    candidates = []
    
    # Regret-based heuristics with different alpha, beta
    for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for beta in [0.0, 0.5, 1.0, 2.0]:
            candidates.append(('regret', alpha, beta))
    
    # Simple heuristics
    # Ensure all tuples have length 5 to match unpacking: (_, type_id, p1, p2, p3)
    for p2 in [0.5, 1.0, 1.5, 2.0, 3.0]:
        candidates.append(('simple', 0, 1.0, p2, 0.0))
    
    for p1 in [1.0, 2.0, 3.0, 5.0]:
        for p2 in [1.0, 2.0, 3.0, 5.0]:
            candidates.append(('simple', 1, p1, p2, 0.0))
    
    for p1 in [0.5, 1.0, 2.0]:
        candidates.append(('simple', 2, p1, 0.0, 0.0))
    
    for p1 in [1.0, 2.0]:
        for p2 in [1.0, 2.0]:
            candidates.append(('simple', 3, p1, p2, 0.0))
    
    for p1 in [1.0, 2.0]:
        for p2 in [1.0, 2.0]:
            for p3 in [0.1, 0.5, 1.0]:
                candidates.append(('simple', 4, p1, p2, p3))
    
    # Evaluate candidates
    best_avg = -1
    best_eta = None
    
    # First pass: Quick screen with 1 seed
    quick_results = []
    for c in candidates:
        if time.time() - start_time > time_limit - 60:
            break
        
        if c[0] == 'regret':
            _, alpha, beta = c
            eta = _make_regret_heuristic(prizes, distance_matrix, max_len, alpha, beta)
        else:
            _, type_id, p1, p2, p3 = c
            eta = _make_simple_heuristic(prizes, distance_matrix, max_len, type_id, p1, p2, p3)
        
        s = _simulate_aco(prizes, distance_matrix, max_len, eta, 0)
        quick_results.append((s, c, eta))
    
    if not quick_results:
        # Fallback
        return _make_regret_heuristic(prizes, distance_matrix, max_len, 1.0, 1.0)
    
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Second pass: Evaluate top candidates with more seeds
    top_k = min(15, len(quick_results))
    for s_quick, c, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 20:
            break
        scores = [s_quick]
        for seed in [1, 2]:
            if time.time() - start_time > time_limit - 15:
                break
            s = _simulate_aco(prizes, distance_matrix, max_len, eta, seed)
            scores.append(s)
        avg = np.mean(scores)
        if avg > best_avg:
            best_avg = avg
            best_eta = eta
    
    # Third pass: Local refinement around best if it's regret-based
    if best_eta is not None and time.time() - start_time < time_limit - 30:
        # Try to refine regret parameters if the best was regret-based
        # We need to track which candidate was best
        # For simplicity, just try a few more regret variants
        for alpha in [0.8, 1.2, 1.5, 2.5, 4.0]:
            for beta in [0.3, 0.7, 1.3, 1.7]:
                if time.time() - start_time > time_limit - 10:
                    break
                eta = _make_regret_heuristic(prizes, distance_matrix, max_len, alpha, beta)
                s = _simulate_aco(prizes, distance_matrix, max_len, eta, 0)
                if s > best_avg * 0.95:
                    scores = [s]
                    for seed in [1, 2]:
                        if time.time() - start_time > time_limit - 8:
                            break
                        s2 = _simulate_aco(prizes, distance_matrix, max_len, eta, seed)
                        scores.append(s2)
                    avg = np.mean(scores)
                    if avg > best_avg:
                        best_avg = avg
                        best_eta = eta
    
    # Final safety check
    if best_eta is None or not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = _make_regret_heuristic(prizes, distance_matrix, max_len, 1.0, 1.0)
    
    return best_eta
