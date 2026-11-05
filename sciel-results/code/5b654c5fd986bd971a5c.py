"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    Strategy: Simulate Lowest Degree First (LDF) peeling to determine a dynamic
    priority based on the order vertices would be removed in an LDF heuristic.
    This captures the sequential dependency of the greedy choice better than static features.
    """
    start_time = time.time()
    
    # Ensure float matrix for linear algebra
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
        
    # Simulate LDF Peeling
    # active: boolean mask of vertices still in the graph
    # deg: current degree of each vertex in the active subgraph
    active = np.ones(N, dtype=bool)
    deg = A.sum(axis=1).astype(int)
    
    # peeling_step[i] will store the step at which vertex i is removed
    # Lower step = picked earlier = higher priority
    peeling_step = np.zeros(N, dtype=float)
    
    # To speed up min-degree finding, we can just scan the active vertices.
    # N=200, so O(N^2) is 40,000 ops, which is instant.
    
    for step in range(N):
        if not active.any():
            break
            
        # Find active vertex with minimum degree
        # Mask inactive vertices with infinity so they are ignored
        current_deg = np.where(active, deg, np.inf)
        min_idx = np.argmin(current_deg)
        
        # Assign peeling step
        peeling_step[min_idx] = step
        
        # Remove min_idx and its neighbors from active set
        # Neighbors are where A[min_idx] == 1
        neighbors = A[min_idx, :].astype(bool)
        
        # Mark min_idx and neighbors as inactive
        active[min_idx] = False
        active[neighbors] = False
        
        # Update degrees of remaining active vertices
        # Any active vertex connected to min_idx loses 1 degree
        # We can do this by: deg[neighbors & active] -= 1
        # But we need to be careful: 'neighbors' includes min_idx? No, A is symmetric, A[i,i]=0 usually.
        # If self-loops exist, A[i,i]=1. ER graphs usually don't have self-loops, but let's be safe.
        # The mask 'neighbors' indicates who is connected to min_idx.
        # We subtract 1 from the degree of all active neighbors.
        
        # Identify active neighbors
        active_neighbors = neighbors & active
        
        # Update degrees
        deg[active_neighbors] -= 1
        
        # Safety check for time
        if time.time() - start_time > 10.0:
             # Fallback to simple degree-based score if simulation is slow (unlikely)
             return -deg.astype(float)

    # The greedy decoder picks in descending score order.
    # We want vertices with lower peeling_step to be picked first.
    # So score should be inversely related to peeling_step.
    # score = -peeling_step
    # This ensures the greedy decoder picks vertices in the exact same order as the LDF simulation.
    
    score = -peeling_step
    
    return score
# EVOLVE-BLOCK-END
