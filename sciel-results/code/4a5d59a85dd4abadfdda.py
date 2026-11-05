# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Computes a priority score for each vertex based on the k-core decomposition.
    
    Approach:
    1. Compute the core number for each vertex.
    2. The core number represents the "centrality" or "density" of the neighborhood.
    3. Vertices with lower core numbers are more peripheral.
    4. Selecting peripheral vertices first (low core number) allows the greedy
       algorithm to peel away the outer layers of the graph, often resulting in
       a larger independent set than selecting high-degree central vertices first.
    5. Priority is set to -core_number, so lower core numbers yield higher priority.
    6. Ties are broken by degree (lower degree gets slightly higher priority).
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
        
    N = adjacency.shape[0]
    
    # Compute initial degrees
    deg = adjacency.sum(axis=1).astype(int)
    
    # Compute Core Numbers
    # O(N^2) algorithm suitable for N=200
    cur_deg = deg.copy()
    processed = np.zeros(N, dtype=bool)
    core_numbers = np.zeros(N, dtype=int)
    
    for _ in range(N):
        # Find the unprocessed vertex with the minimum current degree
        # We can do this by masking processed vertices with a large number
        masked_deg = np.where(processed, np.inf, cur_deg)
        v = np.argmin(masked_deg)
        
        if masked_deg[v] == np.inf:
            break
            
        core_numbers[v] = cur_deg[v]
        processed[v] = True
        
        # Update degrees of unprocessed neighbors
        nbs = np.where(adjacency[v] == 1)[0]
        for u in nbs:
            if not processed[u]:
                cur_deg[u] -= 1
                
    # Priority: 
    # We want to pick vertices with LOW core numbers first.
    # So priority should be HIGH for LOW core numbers.
    # priority = -core_numbers
    # Add a small tie-breaker based on degree (lower degree is better/safer)
    # degree is already int, core is int.
    # Let's use: priority = -core_numbers - 1e-6 * deg
    # This ensures that if core numbers are equal, lower degree is preferred.
    
    scores = -core_numbers.astype(float) - 1e-6 * deg.astype(float)
    
    return scores
# EVOLVE-BLOCK-END
