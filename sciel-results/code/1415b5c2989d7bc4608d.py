import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ACO orienteering.
    
    Structural Change:
    1. Global Topological Awareness (Centrality):
       Instead of treating each move in isolation, we compute a 'centrality' score 
       for each node based on its proximity to other high-prize nodes. This encourages 
       ants to visit nodes that serve as hubs for collecting multiple prizes efficiently, 
       moving beyond simple local greedy ratios.
       
    2. Smoothed Budget Penalty:
       We retain the exponential penalty for budget consumption but combine it with 
       the centrality to create a smoother, more informative gradient. The heuristic 
       rewards moves that are cheap, go to valuable nodes, and lead to other valuable nodes.
       
    3. Feasibility Masking:
       Strict masking of infeasible moves (d[i,j] + d[j,0] > max_len).
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    eps = 1e-9
    
    # 1. Compute Total Cost: d[i,j] + d[j,0]
    # This represents the length consumed by moving to j and then returning to depot.
    # Shape (n, n)
    total_cost = distance_matrix + distance_matrix[:, 0:1]
    
    # 2. Static Feasibility Mask
    feasible_mask = total_cost <= max_len
    
    # 3. Compute Centrality Scores
    # C[j] = sum over all k of (prize[k] / (d[j,k] + eps))
    # This measures how "central" node j is to the prize distribution.
    # We exclude node 0 from contributing to centrality if its prize is 0, 
    # but the formula naturally handles prize[0]=0.
    # We add a small offset to distance to avoid division by zero for self-loops, 
    # though self-loops are irrelevant for movement.
    inv_dist = 1.0 / (distance_matrix + eps)
    # Weighted sum of inverse distances by prize
    centrality = np.sum(inv_dist * prizes[None, :], axis=1)
    
    # Normalize centrality to [0, 1] range for stability, then scale it up
    # to have a meaningful impact (e.g., multiply by 2.0 so max boost is 3x)
    max_cent = np.max(centrality)
    if max_cent > eps:
        centrality_norm = centrality / max_cent
    else:
        centrality_norm = np.zeros_like(centrality)
    
    # Boost factor: 1.0 for low centrality, up to 3.0 for high centrality
    boost = 1.0 + 2.0 * centrality_norm
    
    # 4. Base Heuristic Components
    # Prize value at destination
    prize_val = prizes[None, :]  # Shape (n, n), broadcasted
    
    # Local step cost
    local_cost = distance_matrix + eps
    
    # Total budget commitment
    commit_cost = total_cost + eps
    
    # Exponential budget penalty
    # Penalize moves that consume a large fraction of max_len
    # lambda = 2.0 provides a moderate penalty
    lambda_pen = 2.0
    budget_penalty = np.exp(-lambda_pen * total_cost / max_len)
    
    # Combine components
    # eta[i,j] = (prize[j] * boost[j]) / (d[i,j] * (d[i,j] + d[j,0])) * penalty
    # The denominator d[i,j] * (d[i,j] + d[j,0]) strongly prefers short steps 
    # and low total commitment.
    
    eta = (prize_val * boost[None, :]) / (local_cost * commit_cost) * budget_penalty
    
    # 5. Apply Feasibility Mask
    # Infeasible moves get very low attractiveness
    eta = np.where(feasible_mask, eta, 1e-9)
    
    # Ensure node 0 is not a destination (prize is 0, but explicitly mask to be safe)
    eta[:, 0] = 1e-9
    
    # 6. Normalization
    # Scale so max is 1.0
    max_eta = np.max(eta)
    if max_eta > eps:
        eta = eta / max_eta
    else:
        eta = np.full_like(eta, 1e-9)
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return eta
    
    return eta
