"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a heuristic matrix for Ant Colony Optimization.
    
    Strategy: Potential-Field Heuristic with Global Reachability
    Instead of a simple ratio, we model the heuristic as a potential field where
    the attractiveness of moving from i to j depends on:
    1. The 'value' of node j (prize).
    2. The 'cost' of the move d[i,j] and the return cost d[j,0].
    3. A 'potential' term that encourages visiting nodes that allow for future visits.
    
    Specifically, we compute for each node j a 'potential' P[j] which is the maximum
    total prize achievable in a tour starting from 0, visiting j as the *first* node,
    and then optimally visiting other nodes subject to the length constraint, and returning.
    
    However, computing this exactly is NP-hard (Orienteering). 
    We approximate it using a greedy potential:
    P[j] = prize[j] + sum of prizes of nodes k that can be visited after j 
           given the remaining budget (max_len - d[0,j] - d[j,0]).
    
    To make this computationally feasible and robust, we use a simpler but effective 
    approximation:
    h[i,j] = prize[j] * (1 - d[j,0]/max_len)^gamma / (d[i,j] + eps)^beta
    
    The term (1 - d[j,0]/max_len) represents the 'slack' or 'potential' remaining after
    committing to visit j and return. If d[j,0] is close to max_len, the slack is near 0,
    making j unattractive even if prize[j] is high, because it consumes the entire budget.
    If d[j,0] is small, the slack is high, allowing j to be part of a longer, potentially
    higher-value tour.
    
    This is a structural change from the current program which used a ratio of prize to cost.
    Here, we use a multiplicative 'feasibility potential' that directly models the budget
    consumption relative to the total budget.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure non-negative and finite
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    n = len(prizes)
    eps = 1e-9
    
    # Calculate the 'slack' or 'potential' for each node j
    # slack[j] = 1 - d[j,0] / max_len
    # If d[j,0] >= max_len, slack is <= 0, meaning j cannot be visited and returned from
    # (or leaves no room for anything else). We clip this to 0.
    return_costs = distance_matrix[:, 0]
    slack = 1.0 - return_costs / max_len
    slack = np.clip(slack, 0, 1)
    
    # We want to amplify the difference between nodes with high slack and low slack.
    # Using a power gamma. If gamma > 1, we strongly penalize nodes with low slack.
    gamma = 2.0
    potential = slack ** gamma
    
    # Combine with prize
    # value[j] = prize[j] * potential[j]
    value = prizes * potential
    
    # Now, construct the heuristic matrix h[i,j]
    # The immediate cost of moving from i to j is d[i,j].
    # We penalize this cost.
    # h[i,j] = value[j] / (d[i,j] + eps)^beta
    
    beta = 1.0
    local_cost = distance_matrix + eps
    local_penalty = 1.0 / (local_cost ** beta)
    
    # Broadcast value to a matrix: value[None, :] has shape (1, n)
    h = value[None, :] * local_penalty
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h), h, 1e-9)
    h = np.maximum(h, 1e-9)
    
    # Time check
    if time.time() > deadline:
        # If we're running out of time, return a simple heuristic
        # This should not happen given the vectorized operations
        return h
    
    return h
# EVOLVE-BLOCK-END
