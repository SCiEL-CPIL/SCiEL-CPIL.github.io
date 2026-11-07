import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290.0

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    eps = 1e-9
    
    d_0 = distance_matrix[0, :]
    d_k0 = distance_matrix[:, 0]
    
    # Two-step lookahead: for edge (i, j), estimate best total prize if ant at i visits j next.
    # After visiting j, the ant can optionally visit one more node k before returning to 0.
    # Feasibility: d[0,i] + d[i,j] + d[j,k] + d[k,0] <= max_len  (for the two-step case)
    # Or: d[0,i] + d[i,j] + d[j,0] <= max_len  (for the one-step case, i.e., k=0 meaning return directly)
    
    # Cost to go from i to j and then to k and back to 0:
    # cost_ijk = d[i,j] + d[j,k] + d[k,0]
    # Budget available at i: max_len - d[0,i]
    
    budget_at_i = max_len - d_0  # shape (n,)
    
    # cost_ij = d[i,j] + d[j,0]  (one-step: go to j then return)
    cost_ij = distance_matrix + d_k0[None, :]  # shape (n, n)
    
    # cost_ijk = d[i,j] + d[j,k] + d[k,0]
    # d[i,j] is distance_matrix[i,j], d[j,k] is distance_matrix[j,k], d[k,0] is d_k0[k]
    # This is a 3D tensor: cost_ijk[i,j,k] = d[i,j] + d[j,k] + d[k,0]
    # = distance_matrix[i,j] + distance_matrix[j,k] + d_k0[k]
    
    # Shape: (n, n, n)
    # distance_matrix[:, :, None] -> d[i,j,0]
    # distance_matrix[None, :, :] -> d[0,j,k]  
    # d_k0[None, None, :] -> d[0,0,k]
    
    cost_ijk = distance_matrix[:, :, None] + distance_matrix[None, :, :] + d_k0[None, None, :]
    
    # Feasibility for two-step: cost_ijk[i,j,k] <= budget_at_i[i]
    feasible_2step = cost_ijk <= budget_at_i[:, None, None]  # shape (n, n, n)
    
    # For each (i,j), the best additional prize from a second node k:
    # best_k_prize[i,j] = max over k of prizes[k] * feasible_2step[i,j,k]
    # Also consider the option of NOT visiting a second node (just return): prize = 0
    
    prizes_3d = prizes[None, None, :]  # shape (1, 1, n)
    masked_prizes = np.where(feasible_2step, prizes_3d, 0.0)
    best_second_prize = np.max(masked_prizes, axis=2)  # shape (n, n)
    
    # Total prize if we visit j from i:
    # total_prize[i,j] = prizes[j] + best_second_prize[i,j]
    # But only if the one-step is feasible (cost_ij[i,j] <= budget_at_i[i])
    feasible_1step = cost_ij <= budget_at_i[:, None]  # shape (n, n)
    
    total_prize = np.where(feasible_1step, prizes[None, :] + best_second_prize, 0.0)
    
    # Slack: how much budget remains after the minimal path through j
    # slack_1step[i,j] = budget_at_i[i] - cost_ij[i,j]  (if feasible)
    slack_1step = np.maximum(budget_at_i[:, None] - cost_ij, 0.0)
    
    # Heuristic: combine total prize with slack (more slack = more flexible future choices)
    # Use a power to emphasize high-prize paths while rewarding slack
    # eta[i,j] = total_prize[i,j]^alpha * (1 + slack_1step[i,j])^beta
    # Also penalize if j is not feasible at all
    
    alpha = 1.5
    beta = 0.5
    
    # Ensure total_prize is non-negative
    total_prize = np.maximum(total_prize, 0.0)
    
    heuristic = (total_prize ** alpha) * ((1.0 + slack_1step) ** beta)
    
    # Where infeasible, set to very small value
    heuristic[~feasible_1step] = eps
    heuristic[~np.isfinite(heuristic)] = eps
    heuristic[heuristic <= 0] = eps
    
    if time.time() > deadline:
        return (prizes[None, :] + eps) / (distance_matrix + eps)
        
    return heuristic
