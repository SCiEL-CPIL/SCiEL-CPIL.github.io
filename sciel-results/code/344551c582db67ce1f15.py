"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Computes a priority score for each vertex to maximize the size of the
    independent set found by a greedy decoder.
    
    Strategy:
    Prioritize vertices whose neighbors have high degrees. 
    Selecting such a vertex 'sacrifices' high-degree neighbors, which are 
    typically more constraining and less likely to form a large independent 
    set on their own compared to low-degree vertices. This effectively 
    clears dense regions of the graph, allowing more low-degree vertices 
    (which have higher 'value' in an independent set) to remain available 
    for selection.
    
    Score(v) = Sum of degrees of all neighbors of v.
    """
    start_time = time.time()
    
    # Convert to float for matrix multiplication efficiency if needed, 
    # though bool/int works too.
    adj = np.asarray(adjacency, dtype=np.float32)
    
    # Calculate degrees
    degrees = adj.sum(axis=1)
    
    # Calculate sum of neighbor degrees for each vertex
    # This is a matrix-vector product: Adj * Degrees
    scores = adj @ degrees
    
    # Deadline guard (structural requirement)
    # In this implementation, the calculation is O(N^2) and extremely fast 
    # for N=200, so this is rarely triggered, but included per directive.
    if time.time() - start_time > 110.0:
        # Fallback to simple degree-based score if we somehow ran out of time
        # (highly unlikely for this operation)
        return -degrees
    
    return scores
# EVOLVE-BLOCK-END
