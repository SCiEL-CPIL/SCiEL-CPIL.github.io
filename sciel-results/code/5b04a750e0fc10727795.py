"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Build a heuristic matrix that encodes the true value of taking
    an edge (i, j) in the orienteering tour.
    
    For each edge (i, j), we estimate:
    1. The prize at node j (what we gain by visiting j).
    2. The effective cost of the edge, considering both the step i->j 
       and the eventual need to return to 0 (using d[i,j] + d[j,0] as a 
       proxy for the total distance commitment of visiting j from i).
    3. Feasibility: The edge must be part of a potential tour that fits 
       within max_len. We use the lower bound path length from 0 to i, 
       plus the step i->j, plus the return j->0.
    4. Slack bonus: Edges that leave more slack for future moves are 
       boosted more aggressively.
    """
    start_time = time.time()
    DEADLINE = start_time + 290.0  # 300s limit, keep 10s margin
    
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Ensure shape correctness
    if distance_matrix.shape != (n, n):
        # Return a default valid matrix
        default = np.ones((n, n), dtype=np.float64) * 1e-9
        default[0, 0] = 0.0
        return default
    
    eps = 1e-9
    safe_dist = distance_matrix + eps
    
    # 1. Compute lower bound distance from start (0) to every node i
    # This is the minimum possible distance to reach i.
    min_dist_from_0 = distance_matrix[0, :]
    
    # 2. Compute the minimum possible tour length if we take edge (i, j)
    # Path: 0 -> ... -> i -> j -> 0
    # Lower bound: min_dist_from_0[i] + d[i, j] + d[j, 0]
    min_tour_len = min_dist_from_0[:, None] + distance_matrix + distance_matrix[None, 0]
    
    # Feasibility mask: If the lower bound exceeds max_len, the edge is 
    # definitely infeasible.
    feasible = min_tour_len <= max_len
    
    # 3. Base Heuristic: Prize over Effective Cost
    # We use (d[i, j] + d[j, 0]) as the cost. This penalizes nodes that are 
    # far from the origin, which is crucial for the "return to 0" constraint.
    # It also penalizes long steps from i to j.
    effective_cost = distance_matrix + distance_matrix[None, 0] + eps
    base_heuristic = prizes[None, :] / effective_cost
    
    # Apply feasibility mask
    heuristic = np.where(feasible, base_heuristic, eps)
    
    # 4. Slack Boost
    # Slack = max_len - min_tour_len
    # We want to strongly favor edges that leave significant slack.
    # A linear boost was used before. Let's try a slightly more aggressive 
    # boost for high slack, but keep it stable.
    slack = max_len - min_tour_len
    slack = np.maximum(slack, 0.0)
    
    # Normalize slack to [0, 1] range relative to max_len for scaling
    # If max_len is small, slack is small. 
    # Boost = 1 + (slack / max_len) * 2.0  (Twice the previous boost strength)
    # This makes the difference between a "tight" edge and a "loose" edge 
    # more pronounced, helping the ant avoid dead-ends earlier.
    boost = 1.0 + (slack / max_len) * 2.0
    boost = np.where(feasible, boost, 1.0)
    
    heuristic = heuristic * boost
    
    # 5. Cleanup: Ensure positivity and finiteness
    heuristic = np.where(heuristic > 0, heuristic, eps)
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    
    # 6. Diagonal and Node 0 handling
    # Diagonal: self-loops are not allowed/visited
    np.fill_diagonal(heuristic, eps)
    
    # Node 0: Prize is 0. 
    # Edges to 0 (j=0) should have low attractiveness because prize[0]=0.
    # base_heuristic[:, 0] = 0 / (d[i,0] + d[0,0]) = 0. 
    # So heuristic[:, 0] is already eps (or 0*boost=0 -> eps).
    # This is correct: we don't want to go back to 0 until the end.
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - start_time:
        # Return what we have
        pass
    
    return heuristic
# EVOLVE-BLOCK-END
