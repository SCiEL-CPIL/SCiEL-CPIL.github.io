import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    Strategy:
    1. Primary: Lowest Degree (vertices with fewer neighbors are better candidates).
    2. Secondary: Vertices connected to high-degree neighbors are preferred.
       Rationale: Picking a vertex that blocks a "hub" (high-degree neighbor) is more
       efficient than picking a vertex that blocks a "leaf" (low-degree neighbor),
       because hubs block many other vertices if selected. By blocking the hub, we
       "free" its other neighbors to be selected later (if they are not blocked by
       other picks).
    
    Score = - (Weight_Deg * Degree + Weight_Nbr * Sum(1 / (Deg(Nbr) + 1)))
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Calculate degrees
    deg = adj.sum(axis=1)
    
    # 2. Calculate sum of inverse degrees of neighbors
    # inv_deg[i] = 1 / (deg[i] + 1)
    inv_deg = 1.0 / (deg + 1.0)
    
    # neighbor_inv_sum[i] = sum_{j in N(i)} inv_deg[j]
    # This is a matrix-vector product
    neighbor_inv_sum = adj @ inv_deg
    
    # 3. Combine scores
    # We want higher score for lower degree and lower neighbor_inv_sum.
    # Weighting:
    # Degree range: [0, n-1]
    # neighbor_inv_sum range: [0, n-1] (actually max is ~0.5*n)
    # To make Degree the primary key, we scale it up.
    # If deg difference is 1, it should outweigh max neighbor_inv_sum difference.
    # Max neighbor_inv_sum diff is < n.
    # So weight 1.0 for degree might be too low if n is large?
    # For n=200, max sum ~ 100.
    # If deg diff is 1, and sum diff is 100, sum dominates.
    # We want deg to dominate.
    # So scale deg by n? Or just a large constant.
    # Let's use weight 10.0 for degree.
    # Score = - (10.0 * deg + 1.0 * neighbor_inv_sum)
    
    score = -(10.0 * deg + 1.0 * neighbor_inv_sum)
    
    # Safety deadline check
    if time.time() - start_time > 140:
        # Fallback to simple LDF
        return -deg
        
    return score
