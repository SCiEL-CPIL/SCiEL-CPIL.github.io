"""Program for Maximum Independent Set via dynamic exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores to vertices based on a simulated parallel discrete-time
    exclusion process. Vertices with fewer active neighbors become eligible earlier.
    The score reflects the earliest time step at which a vertex becomes eligible,
    capturing dynamic interference patterns.
    
    Higher score = selected earlier in greedy decoder.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    # Work with integer adjacency for efficiency
    adj = adjacency.astype(np.int32)
    
    # Track active (not yet selected) vertices
    active = np.ones(n, dtype=bool)
    
    # For each vertex, track the number of currently active neighbors
    # Initially, active_neighbors[v] = degree[v]
    active_neighbors = adj.sum(axis=1).astype(np.int32)
    
    # We'll simulate rounds. In each round, vertices with 0 active neighbors are selected.
    # But since we need a total ordering (scores), we track the round at which each vertex
    # becomes eligible (has 0 active neighbors among those still unselected in our simulation).
    
    # However, the greedy decoder selects in descending score order, skipping neighbors of picked.
    # Our simulation should mirror this: when a vertex is "selected" in our simulation,
    # its neighbors lose one active neighbor count.
    
    # To get a deterministic total ordering, we break ties by vertex index.
    # We simulate the process and assign scores based on the round (time step) when
    # a vertex becomes eligible. Lower round = higher priority.
    
    # But wait: in the greedy decoder, we pick the highest score vertex first, then skip its neighbors.
    # Our simulation: in each round, all vertices with 0 active neighbors become eligible.
    # Among those eligible, we need a total order. We'll use: earliest round first, then lower index first.
    # So score = -round * large_constant - index (to make higher score = earlier selection).
    
    # Actually, let's think differently. The greedy decoder processes vertices in descending score order.
    # If we assign score = -eligibility_round * n - vertex_index, then vertices with lower eligibility_round
    # (and lower index for ties) get higher scores and are processed first.
    
    # Let's simulate:
    eligibility_round = np.full(n, -1, dtype=np.int32)
    remaining = np.ones(n, dtype=bool)
    active_neighbors = adj.sum(axis=1).astype(np.int32)
    
    current_round = 0
    while remaining.any():
        # Find vertices that are remaining and have 0 active neighbors
        eligible = remaining & (active_neighbors == 0)
        
        if not eligible.any():
            # No vertices with 0 active neighbors. This can happen if there's a cycle
            # where every remaining vertex has at least one remaining neighbor.
            # In this case, we need to break the deadlock. We'll select the vertex with
            # the fewest active neighbors (and lowest index for ties) as the next eligible.
            remaining_indices = np.where(remaining)[0]
            min_neighbors = active_neighbors[remaining_indices].min()
            candidates = remaining_indices[active_neighbors[remaining_indices] == min_neighbors]
            # Pick the one with lowest index for determinism
            chosen = candidates[0]
            eligible = np.zeros(n, dtype=bool)
            eligible[chosen] = True
        
        # Assign eligibility round to all eligible vertices
        eligibility_round[eligible] = current_round
        
        # Remove eligible vertices from remaining and update active_neighbors of their neighbors
        eligible_indices = np.where(eligible)[0]
        remaining[eligible_indices] = False
        
        # For each neighbor of the selected vertices, decrement active_neighbors
        # adj[eligible_indices, :] gives the neighbors of selected vertices
        neighbor_updates = adj[eligible_indices, :].sum(axis=0)
        active_neighbors -= neighbor_updates
        
        current_round += 1
    
    # Now assign scores: higher score = should be selected earlier
    # score = -eligibility_round * n - vertex_index
    # This ensures vertices with lower eligibility_round get higher scores.
    # For ties in eligibility_round, lower vertex_index gets higher score.
    scores = -eligibility_round.astype(float) * n - np.arange(n, dtype=float)
    
    return scores
# EVOLVE-BLOCK-END
