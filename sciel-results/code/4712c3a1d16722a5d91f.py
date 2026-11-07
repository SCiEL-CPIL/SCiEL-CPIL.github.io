"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a heuristic matrix for Ant Colony Optimization.
    
    Strategy: Feasibility-Aware Global Heuristic
    The heuristic h[i, j] is designed to guide ants towards nodes j that:
    1. Have high prizes.
    2. Are "safe" to visit, meaning there is enough remaining length budget 
       to potentially visit other high-value nodes and return to the start.
       
    We approximate the "risk" of visiting j by considering the minimum additional 
    cost required to return to 0 from j (which is d[j, 0]) and the cost to get 
    to j from i (d[i, j]). 
    However, a more powerful signal for the *global* tour is how much "slack" 
    is left. If we visit j, the remaining budget for the rest of the tour is 
    max_len - (current_len + d[i,j] + d[j,0]).
    Since we don't know current_len at heuristic computation time (it's computed 
    once before the colony runs), we must use a static proxy.
    
    A better static proxy for "goodness" of an edge (i, j) in a length-constrained 
    orienteering problem is the prize of j divided by the "total cost" of including j 
    in a round trip from 0, which is roughly d[i,j] + d[j,0] if we assume i is near 0, 
    but more generally, we want to avoid edges that are too long relative to the budget.
    
    Let's use a "Penalized Prize" approach:
    h[i, j] = prize[j] * exp(-lambda * (d[i,j] + d[j,0]) / max_len)
    
    This exponentially penalizes edges that consume a large fraction of the total budget.
    The parameter lambda controls the strength of the feasibility constraint.
    A high lambda makes ants prefer short, efficient loops. A low lambda makes them 
    greedy for prizes regardless of distance.
    
    Given max_len=3 and points in unit square, distances are ~0.1-1.4.
    d[i,j]+d[j,0] is typically 0.2-2.8.
    Ratio is 0.07-0.93.
    exp(-1.0 * 0.5) ~ 0.6.
    exp(-3.0 * 0.5) ~ 0.22.
    
    Let's try lambda = 2.0 to strongly discourage long detours that might prevent 
    visiting multiple nodes.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Basic sanity checks
    if prizes.ndim != 1 or distance_matrix.ndim != 2:
        raise ValueError("Invalid dimensions")
        
    N = len(prizes)
    if distance_matrix.shape != (N, N):
        raise ValueError("Distance matrix shape mismatch")

    # Clip values to ensure no negative distances or prizes
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    # Vectorized computation
    # d[i, j] is distance_matrix[i, j]
    # d[j, 0] is distance_matrix[j, 0]
    
    # We want h[i, j] to be high if prize[j] is high and the cost (d[i,j] + d[j,0]) is low.
    # Cost matrix C[i, j] = d[i, j] + d[j, 0]
    # d[j, 0] is a column vector (N, 1) if we take distance_matrix[:, 0][:, None]
    # d[i, j] is the full matrix (N, N)
    
    # d_j0 shape (N, 1)
    d_j0 = distance_matrix[:, 0][:, None]
    # d_ij shape (N, N)
    d_ij = distance_matrix
    
    # Total cost to insert j after i and return to 0 (partially)
    # Note: The actual constraint is len_so_far + d[i,j] + d[j,0] <= max_len.
    # The term (d[i,j] + d[j,0]) is the *incremental* cost added to the path length 
    # when moving from i to j and *if* we were to return to 0 immediately.
    # This is a good proxy for the "budget consumption" of this step.
    
    cost = d_ij + d_j0
    
    # Normalize cost by max_len to make lambda scale-invariant
    norm_cost = cost / max_len
    
    # Exponential penalty
    # We want to suppress edges that are "too expensive".
    # h[i, j] = prize[j] * exp(-lambda * norm_cost)
    # prize[j] is (1, N)
    p = prizes[None, :]
    
    lambda_val = 2.5  # Tuned for strong feasibility preference
    
    # Calculate penalty
    penalty = np.exp(-lambda_val * norm_cost)
    
    h = p * penalty
    
    # Ensure h is finite and positive
    # If prize is 0, h is 0. The problem statement says "Heuristic entries that are 
    # not positive become 1e-9".
    # So we replace 0 and negative (none expected) with 1e-9.
    
    # Check for non-finite values
    if not np.all(np.isfinite(h)):
         h = np.nan_to_num(h, nan=1e-9, posinf=1e9, neginf=1e-9)
         
    # Apply the "not positive becomes 1e-9" rule
    h = np.where(h > 0, h, 1e-9)
    
    # Time check (should be instantaneous, but good practice)
    if time.time() > deadline:
         return h

    return h
# EVOLVE-BLOCK-END
