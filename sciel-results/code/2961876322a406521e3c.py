"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty with a repulsive force field around the current tour.
    
    Instead of only penalizing edges in the current tour, we create a continuous
    penalty field where edges geometrically close to the tour are also penalized,
    discouraging 2-opt from making small perturbations that keep the tour in
    the same region of the solution space.
    """
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    # Extract tour edges (i, i+1) cyclically
    tour_edges = []
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edges.append((a, b))
    
    # Build arrays of tour edge endpoints
    # tour_edges[i] = (a_i, b_i)
    # We want to compute for each pair (i,j) the minimum closeness to any tour edge
    
    # Precompute: for each tour edge (a, b), we have distances from all points to a and b
    # closeness(i,j; a,b) = min(d(i,a)+d(j,b), d(i,b)+d(j,a))
    
    # To vectorize: 
    # Let A = array of 'a' endpoints, B = array of 'b' endpoints for all tour edges
    # For each (i,j), we need min over all tour edges of min(d(i,a)+d(j,b), d(i,b)+d(j,a))
    
    # This is expensive to do naively. Let's think of a smarter approach.
    
    # Alternative: For each point i, find which tour edges are "near" i.
    # Actually, let's just do it with broadcasting but carefully.
    
    # Number of tour edges = n
    # For each tour edge (a, b), define for all pairs (i,j):
    #   closeness_ij = min(D[i,a] + D[j,b], D[i,b] + D[j,a])
    # We want min over all tour edges.
    
    # This is n (tour edges) * n^2 (pairs) = n^3 operations, which for n=100 is 1e6, fine.
    
    D = edge_distance
    m = len(tour_edges)
    
    # Arrays of endpoints
    a_arr = np.array([e[0] for e in tour_edges], dtype=int)
    b_arr = np.array([e[1] for e in tour_edges], dtype=int)
    
    # For each tour edge index t, with endpoints a_t, b_t:
    #   For all (i,j): closeness_t[i,j] = min(D[i,a_t] + D[j,b_t], D[i,b_t] + D[j,a_t])
    # We want min over t.
    
    # Let's compute this in a loop over tour edges, keeping the running min.
    # This is O(n^3) which is 1e6 for n=100, acceptable.
    
    # Initialize min_closeness to infinity
    min_closeness = np.full((n, n), np.inf)
    
    for t in range(m):
        a = a_arr[t]
        b = b_arr[t]
        # D[:, a] is vector of distances from all points to a
        # D[:, b] is vector of distances from all points to b
        da = D[:, a]  # shape (n,)
        db = D[:, b]  # shape (n,)
        
        # closeness[i,j] = min(da[i] + db[j], db[i] + da[j])
        # da[:, None] + db[None, :] gives da[i] + db[j] for all i,j
        # db[:, None] + da[None, :] gives db[i] + da[j] for all i,j
        c1 = da[:, None] + db[None, :]  # shape (n, n)
        c2 = db[:, None] + da[None, :]  # shape (n, n)
        c = np.minimum(c1, c2)
        
        min_closeness = np.minimum(min_closeness, c)
    
    # Now min_closeness[i,j] is the minimum closeness of edge (i,j) to any tour edge.
    # Edges in the tour itself will have closeness 0 (when i=a, j=b or i=b, j=a).
    
    # Apply a penalty that decays with closeness.
    # Use an exponential decay: penalty_factor = 1 + alpha * exp(-beta * min_closeness)
    # But we want to be careful: edges far from the tour should get little penalty.
    
    # Scale: typical edge lengths in unit square ~ 0.1-0.3
    # Use beta such that exp(-beta * 0.1) is still significant, but exp(-beta * 0.5) is small.
    # Let's use beta = 10, so exp(-1) ~ 0.37 at distance 0.1, exp(-5) ~ 0.007 at 0.5.
    
    # Also incorporate edge_n_used: edges that have been used more should get higher penalty.
    # Normalize edge_n_used
    max_used = edge_n_used.max()
    if max_used > 0:
        norm_used = edge_n_used / max_used
    else:
        norm_used = np.zeros_like(edge_n_used)
    
    # Penalty: base penalty from the force field, modulated by usage
    # force_field_penalty = exp(-beta * min_closeness)
    # total_penalty = 1 + gamma * (0.5 + 0.5 * norm_used) * force_field_penalty
    
    # Let's tune: gamma should be large enough to matter but not destroy the tour.
    # Start with gamma = 0.5, beta = 10.
    
    beta = 10.0
    gamma = 0.5
    
    force_field = np.exp(-beta * min_closeness)
    
    # Combine: edges in the tour (closeness ~ 0) get force_field ~ 1
    # Edges far from tour get force_field ~ 0
    # Usage modulates: heavily used edges get more penalty
    
    penalty_factor = 1.0 + gamma * (0.5 + 0.5 * norm_used) * force_field
    
    result = D * penalty_factor
    
    # Ensure symmetry and finite values
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)
    
    # Replace any non-finite values
    result = np.nan_to_num(result, nan=0.0, posinf=D.max(), neginf=0.0)
    
    return result
# EVOLVE-BLOCK-END
