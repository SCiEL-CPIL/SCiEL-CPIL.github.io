"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search on TSP.
    Returns a modified distance matrix that penalizes frequently used edges,
    encouraging the subsequent 2-opt to explore different tour structures.
    
    Uses a combination of:
    1. Global usage-based penalty (smooth saturation)
    2. Local penalty on current tour edges with adaptive strength
    3. Regret-based weighting: edges that are 'expensive' to break get less penalty
    4. K-NN structural bias: extra penalty on tour edges not in either endpoint's
       nearest-neighbor set, targeting geometrically implausible edges
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s limit
    
    n = edge_distance.shape[0]
    tour = np.asarray(local_opt_tour)
    if len(tour) < 2:
        return edge_distance.copy()
    
    # Build list of tour edges with their true distances and compute regret
    # Regret = how much longer the next-best alternative is compared to this edge
    tour_edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        d = edge_distance[u, v]
        tour_edges.append((d, u, v))
    
    # For each tour edge, compute the regret: second-best distance from u to any node
    # except v, minus the current edge distance
    # This tells us how "valuable" each edge is in the tour
    regrets = np.zeros(n)
    for idx, (d, u, v) in enumerate(tour_edges):
        # Get all distances from u, excluding v
        dists_from_u = edge_distance[u].copy()
        dists_from_u[v] = np.inf
        # The second best is the min of remaining (which is the best alternative)
        best_alt = np.min(dists_from_u)
        # Regret: how much we'd lose by breaking this edge
        # If regret is high, the edge is critical; if low, it's easy to replace
        regrets[idx] = best_alt - d
    
    # Normalize regrets to [0, 1] range
    if np.max(regrets) > 0:
        regret_norm = regrets / (np.max(regrets) + 1e-10)
    else:
        regret_norm = np.zeros(n)
    
    # Compute K-NN structural bias: for each node, find its k nearest neighbors
    # An edge (u,v) is "structurally poor" if v is not in u's K-NN set AND u is not in v's K-NN set
    k = 5  # number of nearest neighbors to consider
    knn_sets = [set() for _ in range(n)]
    for u in range(n):
        dists = edge_distance[u].copy()
        dists[u] = np.inf
        # Get indices of k nearest neighbors
        nn_indices = np.argsort(dists)[:k]
        knn_sets[u] = set(nn_indices.tolist())
    
    # Create penalty matrix
    penalty = np.zeros_like(edge_distance)
    
    # Global penalty: proportional to usage count with smooth saturation
    alpha_global = 0.25
    penalty_global = alpha_global * (edge_n_used / (1.0 + edge_n_used))
    penalty += penalty_global
    
    # Local penalty: extra penalty for edges in the current tour
    # Weight by inverse regret: edges with low regret (easy to break) get MORE penalty
    # This encourages breaking easy-to-replace edges first
    alpha_local_base = 0.4
    for idx, (d, u, v) in enumerate(tour_edges):
        # Inverse regret weighting: low regret -> high penalty
        # Use (1 - regret_norm) so that edges with regret 0 get full penalty
        weight = 1.0 - regret_norm[idx]
        # Add a minimum penalty even for high-regret edges
        local_penalty = alpha_local_base * (0.3 + 0.7 * weight)
        
        # K-NN structural bias: if edge is not in either endpoint's K-NN set,
        # apply additional penalty to target geometrically implausible edges
        if v not in knn_sets[u] and u not in knn_sets[v]:
            # This edge is structurally poor - apply extra penalty
            local_penalty *= 1.5
        
        penalty[u, v] += local_penalty
        penalty[v, u] += local_penalty
    
    # Apply penalty to the distance matrix using multiplicative form
    # This ensures the penalty scales with the original distance
    modified_distance = edge_distance * (1.0 + penalty)
    
    # Ensure symmetry
    modified_distance = (modified_distance + modified_distance.T) / 2.0
    
    # Ensure finite values
    modified_distance = np.nan_to_num(modified_distance, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Safety: check elapsed time
    if time.time() > deadline:
        # Return a simple perturbation if we're running out of time
        return edge_distance * (1.0 + 0.1 * (edge_n_used / (1.0 + edge_n_used)))
    
    return modified_distance
# EVOLVE-BLOCK-END
