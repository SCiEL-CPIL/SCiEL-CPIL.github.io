"""Guided local search penalty for TSP; Targeted 2-opt escape heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Applies a targeted penalty to the edges of the current local optimum.
    Unlike global penalties, this focuses on breaking the specific structure
    of the current tour by penalizing edges that are 'trapped' in the local optimum.
    
    The penalty is additive and proportional to the usage count of the edge,
    but scaled by a factor that encourages breaking edges that have been 
    part of local optima repeatedly.
    
    Args:
    edge_distance: np.ndarray, true distance matrix.
    local_opt_tour: list or np.ndarray, the current tour as a sequence of node indices.
    edge_n_used: np.ndarray, symmetric matrix of usage counts.
    
    Returns:
    np.ndarray: Modified distance matrix for the next 2-opt phase.
    """
    # Copy the true distances
    new_dist = edge_distance.copy()
    
    # Ensure inputs are numpy arrays
    tour = np.asarray(local_opt_tour)
    n = len(tour)
    
    if n < 2:
        return new_dist

    # Identify edges in the current tour
    # The tour is a sequence of nodes [v0, v1, ..., vn-1]
    # Edges are (v0, v1), (v1, v2), ..., (vn-2, vn-1), (vn-1, v0)
    
    # We want to penalize these edges to force 2-opt to break them.
    # However, a simple uniform penalty might not be enough if 2-opt can easily 
    # swap them for slightly better ones that are also in the basin.
    
    # Strategy: 
    # 1. Calculate a base penalty for each edge in the current tour.
    # 2. The penalty is higher for edges that have been used more frequently 
    #    (indicated by edge_n_used).
    # 3. To be "guided", we can also consider the geometry: 
    #    If an edge is long, it's a prime candidate for removal.
    #    If an edge is short, it might be structurally necessary unless heavily penalized.
    
    # Let's use an additive penalty: new_dist[i,j] += lambda * count[i,j]
    # But only for edges in the current tour? 
    # No, the standard GLS penalizes *all* edges in the current local optimum.
    # The current program penalizes ALL edges in the matrix based on count.
    # This new strategy will ONLY penalize the edges in the CURRENT tour.
    # This is a structural change: local vs global penalty application.
    
    # Why? 
    # Global penalties accumulate on edges that might be good globally but were 
    # just part of a bad tour. By only penalizing the current tour's edges, 
    # we force the search to escape the *current* local optimum specifically, 
    # without biasing the entire search space based on history that might be 
    # irrelevant to the current state.
    
    # Let's refine: 
    # The evaluator calls this function, then runs 2-opt on the returned matrix.
    # If we only penalize current tour edges, 2-opt will likely break one or two 
    # of them to find a shorter tour in the *modified* metric. 
    # Then, on the true metric, this new tour might be better than the previous best.
    
    # Penalty magnitude:
    # We need it strong enough to force a break, but not so strong that it 
    # makes the tour arbitrarily long in the modified metric (which might 
    # lead to a bad starting point for the true 2-opt).
    
    # Let's use a penalty that scales with the number of times the edge has been 
    # used in previous local optima (edge_n_used).
    
    # Get counts for current tour edges
    # edge_n_used is a symmetric matrix.
    
    # Iterate through edges in the current tour
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        
        # Get the count for this edge
        count = edge_n_used[u, v]
        
        # If count is 0, it's a new edge. We still penalize it to force 
        # exploration away from the current structure, but maybe less?
        # Actually, if count is 0, it hasn't been a trap before. 
        # But it IS in the current local optimum. 
        # Standard GLS penalizes all edges in the current local optimum.
        
        # Let's apply a penalty: P = alpha * (1 + count)
        # alpha needs to be tuned. 
        # Distances are in [0, sqrt(2)] ~ [0, 1.41].
        # A penalty of 0.5 * (1 + count) might be reasonable.
        # If count is 0, penalty is 0.5.
        # If count is 5, penalty is 3.0.
        
        # However, if we penalize ALL current edges equally, 2-opt might just 
        # swap them in a way that doesn't change the tour much.
        
        # Let's try a different structural approach:
        # "Edge-breaking" penalty.
        # We add a penalty to the edges in the current tour.
        # To ensure 2-opt breaks the tour, the penalty should make the 
        # current tour edges significantly more expensive than alternatives.
        
        # Let's set the penalty to be proportional to the edge length itself?
        # No, that preserves relative order.
        
        # Let's stick to a count-based additive penalty but ONLY on current tour edges.
        
        penalty = 0.5 * (1.0 + count)
        
        # Apply to symmetric matrix
        new_dist[u, v] += penalty
        new_dist[v, u] += penalty

    # Ensure symmetry
    new_dist = (new_dist + new_dist.T) / 2.0
    
    # Safety check
    if not np.all(np.isfinite(new_dist)):
        return edge_distance.copy()

    return new_dist
# EVOLVE-BLOCK-END
