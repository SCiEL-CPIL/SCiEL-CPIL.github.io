import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ACO orienteering.
    
    Structural Change:
    1. Greedy Prize-Density Base: 
       The core heuristic is prize[j] / (d[i,j] + d[j,0]). This directly models
       the value obtained per unit of total length budget consumed (move + return).
    2. Local Proximity Factor:
       Multiplied by 1 / d[i,j] to strongly prefer short incremental steps, 
       preserving budget for subsequent moves. This is retained from the 
       high-performing current structure.
    3. Exponential Budget Gradient:
       An exponential penalty exp(-k * (d[i,j] + d[j,0]) / max_len) is applied.
       This smoothly discourages moves that consume a large fraction of the 
       total length budget, providing a gradient for exploration rather than 
       a hard cliff. This integrates the 'budget awareness' from Inspiration 2 
       into the 'greedy ratio' framework.
    4. Feasibility Mask:
       Edges where d[i,j] + d[j,0] > max_len are strictly infeasible and set 
       to 1e-9.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    eps = 1e-9
    
    # 1. Compute Total Cost: d[i,j] + d[j,0]
    # This represents the length consumed by moving to j and then returning to depot.
    total_cost = distance_matrix + distance_matrix[:, 0:1]  # Shape (n, n)
    
    # 2. Static Feasibility Mask
    # If total_cost > max_len, the move is impossible regardless of current path length.
    feasible_mask = total_cost <= max_len
    
    # 3. Greedy Prize-Density Base
    # Value per unit of total cost.
    # Avoid division by zero by adding eps to denominator.
    base_ratio = prizes[None, :] / (total_cost + eps)  # Shape (n, n)
    
    # 4. Local Proximity Factor
    # Prefer moving to nodes close to current node i.
    local_proximity = 1.0 / (distance_matrix + eps)  # Shape (n, n)
    
    # 5. Exponential Budget Gradient
    # Smoothly penalize moves that use a large fraction of max_len.
    # k=1.5 provides a moderate penalty.
    k = 1.5
    budget_penalty = np.exp(-k * total_cost / max_len)  # Shape (n, n)
    
    # Combine components
    # eta[i,j] = (prize_j / (d_ij + d_j0)) * (1 / d_ij) * exp(-k * (d_ij + d_j0) / max_len)
    eta = base_ratio * local_proximity * budget_penalty
    
    # Apply feasibility mask
    eta = np.where(feasible_mask, eta, 1e-9)
    
    # Ensure node 0 is not a destination (prize is 0 anyway, but ensures no self-loop issues if prize[0] was non-zero in some variant)
    eta[:, 0] = 1e-9
    
    # 6. Normalization
    # Scale so max is 1.0 to keep pheromone and heuristic on similar scales.
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
