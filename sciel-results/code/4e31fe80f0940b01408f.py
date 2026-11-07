"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure non-negative and finite
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    N = len(prizes)
    
    # Strategy: Cluster-Centric Heuristic
    # Instead of valuing a node solely by its own prize and immediate distance,
    # we value it by its potential to serve as a hub for visiting multiple
    # high-value neighbors. This exploits the fact that in orienteering,
    # visiting a cluster of nearby high-value nodes is far more efficient
    # than visiting isolated nodes due to the triangle inequality and
    # budget constraints.
    
    # 1. Compute a "Cluster Value" for each node.
    # For each node j, look at its K nearest neighbors (excluding node 0).
    # Sum the prizes of these neighbors, weighted by their proximity.
    # This creates a signal that boosts nodes located in dense high-value regions.
    
    K = 5  # Number of neighbors to consider for cluster estimation
    sigma = 0.2  # Decay factor for distance in cluster value
    
    cluster_values = np.zeros(N)
    
    # We need to be efficient. N=50, so O(N^2) is fine.
    # For each j, find K nearest neighbors from 1..N-1.
    
    for j in range(1, N):
        # Get distances from j to all other nodes
        dists = distance_matrix[j, 1:N]
        
        # Find indices of K smallest distances
        # We need to map these back to original indices
        # Since dists is a view of distance_matrix[j, 1:], 
        # the index k in dists corresponds to node k+1.
        
        if len(dists) < K:
            k_indices = np.arange(len(dists))
        else:
            k_indices = np.argsort(dists)[:K]
            
        # Get the actual node indices
        neighbor_nodes = k_indices + 1
        
        # Calculate weighted sum of prizes for these neighbors
        # Weight = exp(-distance / sigma)
        for idx in neighbor_nodes:
            d_jk = distance_matrix[j, idx]
            weight = np.exp(-d_jk / sigma)
            cluster_values[j] += prizes[idx] * weight
            
    # 2. Combine individual prize with cluster value.
    # V_j = prize_j + lambda * cluster_value_j
    # This makes "hub" nodes more attractive.
    lambda_factor = 1.0
    total_value = prizes + lambda_factor * cluster_values
    
    # Ensure non-negative
    total_value = np.maximum(total_value, 0)
    
    # 3. Construct the heuristic matrix.
    # h[i, j] should be high if total_value[j] is high and the cost of moving i->j and returning is low.
    # Cost of moving i->j and "committing" to return from j is d[i, j] + d[j, 0].
    # We use a power law to sharpen the preference for high value / low cost.
    
    alpha = 1.5
    beta = 1.0
    eps = 1e-9
    
    # Vectorized calculation
    # total_value is shape (N,)
    # We need h[i, j] = (total_value[j] / (d[i, j] + d[j, 0] + eps))^alpha
    
    # d[j, 0] is the return cost for node j.
    return_costs = distance_matrix[:, 0]
    
    # Create matrix for denominator: d[i, j] + d[j, 0]
    # d[i, j] is distance_matrix
    # d[j, 0] needs to be broadcasted to columns.
    # return_costs shape (N,). We want h[i, j] to depend on return_costs[j].
    # So we add return_costs[None, :] to distance_matrix.
    
    denom = distance_matrix + return_costs[None, :] + eps
    
    # Numerator is total_value[None, :] broadcasted to rows.
    numer = total_value[None, :]
    
    # Calculate ratio
    ratio = numer / denom
    
    # Apply power
    h = ratio ** alpha
    
    # Handle node 0 (prize 0, but might have cluster value? No, prize[0]=0).
    # The ant starts at 0. It moves to j.
    # h[0, j] is important.
    # Also, the ant should not move to 0? 
    # The problem says "moves to an unvisited node j". 
    # Usually, the heuristic for moving to 0 is irrelevant if 0 is visited at start, 
    # but if the ant can return to 0 only at the end? 
    # "until no node qualifies". 
    # The constraint is len + d[i,j] + d[j,0] <= max_len.
    # If j=0, d[i,0] + d[0,0] = d[i,0]. 
    # But 0 is already visited. So j != 0.
    # We can set h[:, 0] to a very small value to discourage moving back to 0 mid-tour, 
    # though the "unvisited" constraint in the solver handles this.
    # However, if the solver allows moving to 0, it would be bad. 
    # Let's just let the formula handle it. prize[0]=0, so total_value[0] might be > 0 if neighbors have high prize.
    # But moving to 0 is usually not allowed in the middle. 
    # To be safe, we can suppress column 0.
    h[:, 0] = 1e-9
    
    # Ensure positive
    h = np.maximum(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
