"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Calculates a priority score for each vertex to maximize the size of the
    greedy independent set.
    
    Heuristic:
    1. Primary: Prefer vertices with lower degree (fewer neighbors to block).
    2. Secondary: Prefer vertices whose neighbors have higher degrees.
       Rationale: Picking a vertex 'v' blocks its neighbors. If the blocked
       neighbors have high degrees, it is beneficial because those high-degree
       vertices would have blocked many other vertices if they had been picked.
       By blocking them early, we 'save' their neighbors from being blocked.
       This is captured by maximizing the sum of degrees of neighbors.
    3. Tertiary: Tie-breaker using inverse degree of neighbors.
       Rationale: Among vertices with similar degree and neighbor-degree-sum,
       prefer those whose neighbors have high degrees (low inverse degree).
       Blocking a high-degree vertex is "cheaper" than blocking a low-degree one.
    """
    start_time = time.time()
    
    # Convert to integer array for efficient arithmetic
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Calculate sum of degrees of neighbors for each vertex
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    
    # Calculate sum of inverse degrees of neighbors for each vertex
    inv_deg = 1.0 / (degrees + 1.0)
    neighbor_inv_degree_sums = (adj @ inv_deg).astype(np.float64)
    
    # Define weights for the cost function.
    # We want to MINIMIZE: degree, neighbor_inv_degree_sums
    # and MAXIMIZE: neighbor_degree_sums
    #
    # Scale analysis for N=200:
    # degree: [0, 199]
    # neighbor_degree_sums: [0, ~40000]
    # neighbor_inv_degree_sums: [0, 200]
    #
    # We want degree to be the primary factor.
    # A difference of 1 in degree should outweigh any difference in other terms.
    # Max neighbor_degree_sums diff is ~40000.
    # Max neighbor_inv_degree_sums diff is ~200.
    # So if we weight neighbor_degree_sums by 1, its contribution is too large.
    # We need to normalize or weight it down so that degree dominates.
    #
    # Let's define cost = w1 * degree - w2 * neighbor_degree_sums + w3 * neighbor_inv_degree_sums
    # We want w1 * degree to dominate.
    # If we set w1 = 1, then w2 * neighbor_degree_sums must be < 1 for any degree diff of 1.
    # Max neighbor_degree_sums is ~40000. So w2 should be < 1/40000 = 2.5e-5.
    # Let's try w2 = 1e-5. Then max contribution is 0.4.
    # Max neighbor_inv_degree_sums is 200. Let's set w3 = 1e-5. Max contribution is 0.002.
    #
    # Alternatively, we can use a large multiplier M for degree and subtract the others.
    # cost = M * degree - neighbor_degree_sums + neighbor_inv_degree_sums
    # If M is large enough, degree dominates.
    # Max neighbor_degree_sums is 40000. Max neighbor_inv_degree_sums is 200.
    # So if M > 40200, degree dominates.
    # Let's use M = 100000.
    
    M = 100000.0
    
    # cost = M * degree - neighbor_degree_sums + neighbor_inv_degree_sums
    # We want to minimize cost, so priority = -cost
    costs = M * degrees - neighbor_degree_sums + neighbor_inv_degree_sums
    priorities = -costs
    
    # Deadline guard
    if time.time() - start_time > 110:
        return priorities.astype(float)
        
    return priorities.astype(float)
# EVOLVE-BLOCK-END
