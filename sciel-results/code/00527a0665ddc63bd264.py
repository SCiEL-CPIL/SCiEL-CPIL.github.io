"""Program for Maximum Independent Set via parallel discrete-time exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores to vertices based on a parallel discrete-time exclusion process.
    In each round, all remaining vertices with the minimum number of active neighbors are
    considered eligible and assigned to the current round. Their neighbors' active neighbor
    counts are decremented. This captures dynamic interference where vertices in dense
    clusters are delayed because their neighbors remain active longer.
    
    Higher score = selected earlier in greedy decoder.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(np.int32)
    
    # Track active (not yet selected) vertices
    remaining = np.ones(n, dtype=bool)
    
    # Number of active neighbors for each vertex
    active_neighbors = adj.sum(axis=1).astype(np.int32)
    
    # Round at which each vertex becomes eligible
    eligibility_round = np.full(n, -1, dtype=np.int32)
    
    current_round = 0
    
    while remaining.any():
        # Find the minimum active neighbor count among remaining vertices
        remaining_indices = np.where(remaining)[0]
        min_neighbors = active_neighbors[remaining_indices].min()
        
        # Identify all remaining vertices with this minimum count
        eligible_mask = remaining & (active_neighbors == min_neighbors)
        eligible_indices = np.where(eligible_mask)[0]
        
        # Assign current round to these vertices
        eligibility_round[eligible_indices] = current_round
        
        # Remove them from remaining
        remaining[eligible_indices] = False
        
        # Update active_neighbors for neighbors of the selected vertices
        # We need to decrement active_neighbors for all vertices that are neighbors of any eligible vertex
        # adj[eligible_indices, :] is a matrix where row i is the neighbors of eligible_indices[i]
        # Summing over axis 0 gives the count of how many eligible neighbors each vertex has
        # But we only care about decrementing the count for the neighbors.
        # The number of active neighbors lost by vertex v is the number of its neighbors that were just selected.
        neighbor_updates = adj[eligible_indices, :].sum(axis=0)
        active_neighbors -= neighbor_updates
        
        current_round += 1
    
    # Assign scores: higher score = earlier selection
    # Vertices with lower eligibility_round should have higher scores.
    # To break ties within the same round, we can use the original degree or index.
    # Using index is simple and deterministic.
    # score = -eligibility_round * large_constant - index
    # Let's use n as the constant to ensure round difference dominates index difference.
    scores = -eligibility_round.astype(float) * n - np.arange(n, dtype=float)
    
    return scores
# EVOLVE-BLOCK-END
