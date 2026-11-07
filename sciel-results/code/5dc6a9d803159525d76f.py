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
    2. Whether the edge is 'feasible' in the sense that there exists a path from 0
       to i with length < max_len - d[i,j] - d[j,0].
    3. The opportunity cost: how many other high-value nodes become unreachable
       after taking this edge and returning to 0 (or continuing).
    
    A direct formula:
    attractiveness[i, j] = prize[j] / (d[i, j] + epsilon) * feasibility_bonus[i, j]
    
    Where feasibility_bonus is 1 if the edge can be part of a valid tour, else 0.
    But since the ant will check feasibility at runtime, the heuristic should
    prioritize edges that are both high-reward and likely to be feasible.
    
    A better direct approach:
    For each node j, compute the maximum prize collectable if we visit j and then
    greedily visit other reachable nodes within the remaining budget.
    Use this as the heuristic weight for edges leading to j.
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
    
    # Clip non-positive entries to 1e-9 as required
    # But first, compute the heuristic
    
    # Direct approach: For each edge (i, j), the heuristic should reflect:
    # 1. The prize value at j
    # 2. The cost (distance) of taking this edge
    # 3. The feasibility of this edge in a tour starting at 0
    
    # Compute for each node j, the minimum distance from 0 to j (via any path, 
    # but since we're in a complete graph with triangle inequality likely, 
    # direct distance is a lower bound)
    min_dist_from_0 = distance_matrix[0, :]
    
    # For each edge (i, j), an ant at i can go to j if:
    #   dist_so_far + d[i,j] + d[j,0] <= max_len
    # The heuristic should favor edges where:
    # - prize[j] is high
    # - d[i,j] is small (cost is low)
    # - The edge is likely to be feasible (i.e., there exists some path to i 
    #   such that the total tour stays within max_len)
    
    # Estimate feasibility for edge (i, j):
    # The minimum possible distance to reach i from 0 is min_dist_from_0[i]
    # So the minimum total tour length if we take edge (i,j) is:
    #   min_dist_from_0[i] + d[i,j] + d[j,0]
    # If this is <= max_len, the edge is potentially feasible.
    
    min_tour_len = min_dist_from_0[:, None] + distance_matrix + distance_matrix[None, 0]
    feasible = min_tour_len <= max_len
    
    # Now build the heuristic:
    # attractiveness[i, j] = prize[j] / (d[i,j] + eps) for feasible edges, else 1e-9
    # But we can do better: weight by how much slack we have after taking this edge.
    
    eps = 1e-9
    safe_dist = distance_matrix + eps
    
    # Base heuristic: prize over cost
    base_heuristic = prizes[None, :] / safe_dist
    
    # Apply feasibility: non-feasible edges get 1e-9
    heuristic = np.where(feasible, base_heuristic, eps)
    
    # Additional refinement: prefer edges that leave more slack
    # Slack = max_len - (min_dist_from_0[i] + d[i,j] + d[j,0])
    slack = max_len - min_tour_len
    slack = np.maximum(slack, 0.0)
    
    # Boost edges with more slack (they are more flexible)
    # Use a mild boost: multiply by (1 + slack / max_len) for feasible edges
    boost = 1.0 + slack / max_len
    boost = np.where(feasible, boost, 1.0)
    
    heuristic = heuristic * boost
    
    # Ensure all entries are positive (replace non-positive with 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, eps)
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    
    # The diagonal (self-loops) should be non-attractive
    # Actually, ants don't visit visited nodes, and node 0 is the start.
    # But to be safe, set diagonal to 1e-9
    np.fill_diagonal(heuristic, eps)
    
    # Edge case: node 0 has prize 0, so edges from/to 0 should reflect that
    # But prize[0] = 0, so base_heuristic for j=0 is already 0/eps = 0 -> 1e-9
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - start_time:
        # Return what we have
        pass
    
    return heuristic
# EVOLVE-BLOCK-END
