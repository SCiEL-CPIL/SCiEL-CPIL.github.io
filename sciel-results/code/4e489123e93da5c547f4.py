"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fast_2opt(tour, dist_matrix, max_passes=10):
    """
    Fast 2-opt implementation.
    Returns the improved tour and the set of edges that were removed (broken)
    during the optimization process.
    """
    n = len(tour)
    if n <= 2:
        return tour, set()
    
    # Precompute tour edges for quick lookup? 
    # Actually, we just need to know which edges were flipped.
    # In 2-opt, we remove edges (i, i+1) and (j, j+1) and add (i, j) and (i+1, j+1).
    # The "broken" edges are the ones removed.
    
    broken_edges = set()
    current_tour = list(tour)
    
    # To speed up, we can use numpy for the gain calculation if we vectorize,
    # but for n=100, a simple loop is fast enough.
    # However, vectorizing the gain calculation is a good idea.
    
    for _ in range(max_passes):
        improved = False
        
        # Calculate gains for all pairs i < j
        # We want to find i, j such that:
        # dist[i][j] + dist[i+1][j+1] < dist[i][i+1] + dist[j][j+1]
        # Note: indices are modulo n.
        
        best_gain = 0
        best_i, best_j = -1, -1
        
        # Vectorized gain calculation
        # i ranges from 0 to n-2 (we consider i < j, and we can fix i=0..n-2, j=i+1..n-1)
        # Actually, standard 2-opt considers all i < j.
        # Let's use a vectorized approach for the gain matrix.
        
        # d1[i, j] = dist[tour[i], tour[j]]
        # d2[i, j] = dist[tour[i+1], tour[j+1]]
        # d3[i, j] = dist[tour[i], tour[i+1]]
        # d4[i, j] = dist[tour[j], tour[j+1]]
        
        # This is O(n^2) memory and time per pass. n=100, so 10k operations. Very fast.
        
        # Extract tour nodes
        t = np.array(current_tour)
        
        # Create index arrays
        i_idx = np.arange(n)
        j_idx = np.arange(n)
        
        # We only care about i < j.
        # Let's create a mask for i < j.
        # To save memory, we can just loop over i and vectorize over j.
        
        for i in range(n - 1):
            # j must be > i
            # Edges to remove: (t[i], t[i+1]) and (t[j], t[j+1])
            # Edges to add: (t[i], t[j]) and (t[i+1], t[j+1])
            
            # j ranges from i+1 to n-1 (and also j=0 is handled by i=n-1? No, i goes to n-2)
            # Wait, if i = n-1, j must be 0? 
            # Standard 2-opt usually iterates i from 0 to n-2 and j from i+1 to n-1.
            # This covers all unique pairs of edges.
            
            # Vectorize over j
            # j candidates: i+1 to n-1
            # If i+1 > n-1, skip.
            if i + 1 >= n:
                continue
                
            js = np.arange(i + 1, n)
            
            # Nodes
            u = t[i]
            v = t[i+1]
            
            # For each j in js:
            # Edge 1 to remove: (u, v) -> dist[u, v] (constant for this i)
            # Edge 2 to remove: (t[j], t[j+1])
            # Edge 1 to add: (u, t[j])
            # Edge 2 to add: (v, t[j+1])
            
            # Note: t[j+1] when j=n-1 is t[0].
            # So we need to handle the wrap-around for t[j+1].
            
            t_j = t[js]
            t_j_next = t[(js + 1) % n]
            
            # Distances
            d_remove_1 = dist_matrix[u, v] # Constant
            d_remove_2 = dist_matrix[t_j, t_j_next]
            
            d_add_1 = dist_matrix[u, t_j]
            d_add_2 = dist_matrix[v, t_j_next]
            
            gains = (d_remove_1 + d_remove_2) - (d_add_1 + d_add_2)
            
            # Find max gain
            max_idx = np.argmax(gains)
            max_gain = gains[max_idx]
            
            if max_gain > 1e-7:
                j = js[max_idx]
                
                # Record broken edges
                broken_edges.add((min(u, v), max(u, v)))
                broken_edges.add((min(t[j], t[j+1]), max(t[j], t[j+1])))
                
                # Perform the flip
                # Reverse segment t[i+1] to t[j]
                current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
                improved = True
                
        if not improved:
            break
            
    return current_tour, broken_edges

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Lookahead Simulation Strategy for Guided Local Search.
    
    1. Simulate 2-opt on the true distance matrix to find which edges
       are "critical" (i.e., the ones 2-opt would break to improve the tour).
    2. Apply a strong penalty to ONLY those critical edges in the returned matrix.
    3. This forces the evaluator's 2-opt to break the specific structural
       bottleneck of the current local optimum, rather than randomly perturbing
       the landscape.
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()

    # 1. Simulate 2-opt to identify critical edges
    # We use a slightly higher max_passes for simulation to be thorough,
    # but 10 is sufficient as per the evaluator's constraint.
    _, critical_edges = _fast_2opt(local_opt_tour, edge_distance, max_passes=10)
    
    # 2. Create perturbed distance matrix
    perturbed_dist = edge_distance.copy()
    
    if not critical_edges:
        # If no improvement is found by 2-opt (global optimum?), 
        # we should still perturb to avoid getting stuck in the same loop.
        # In this case, penalize the edges of the current tour slightly.
        # Or just return the original. 
        # If we return original, the evaluator's 2-opt will do nothing,
        # and the tour stays the same. This is a dead end.
        # So we must perturb.
        # Let's penalize all tour edges slightly.
        penalty = 1.1
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            perturbed_dist[u, v] *= penalty
            perturbed_dist[v, u] *= penalty
        return perturbed_dist

    # 3. Apply penalty to critical edges
    # The penalty should be strong enough to ensure these edges are broken
    # by the evaluator's 2-opt.
    # Since 2-opt accepts a move if the new length < old length,
    # making the old edges very expensive ensures they are removed.
    
    # A multiplier of 2.0 or 3.0 is usually sufficient to force removal
    # unless the alternative is extremely bad, which is unlikely for critical edges
    # that were part of a local optimum.
    
    penalty_factor = 2.5
    
    for u, v in critical_edges:
        perturbed_dist[u, v] *= penalty_factor
        perturbed_dist[v, u] *= penalty_factor
        
    return perturbed_dist
# EVOLVE-BLOCK-END
