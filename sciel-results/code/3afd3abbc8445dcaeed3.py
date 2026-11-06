"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Compute priority scores for vertices using local greedy simulation.
    
    Strategy:
    1. For each vertex, simulate the greedy MIS algorithm on its ego network (vertex + neighbors).
    2. Determine if the vertex is "selected" in this local simulation.
    3. Score = (Is Selected? 1 : 0) - (Number of neighbors that would be blocked if selected).
       Actually, a better proxy: 
       If we pick v, we gain 1 but lose all its neighbors.
       The "cost" of picking v is the number of neighbors that have high priority themselves.
       
    Alternative Strategy (More Robust):
    Use a "regret" based score.
    For each vertex v, calculate how many independent set members we would "lose" 
    by picking v versus not picking v, estimated via local structure.
    
    Let's use a simpler, highly effective heuristic often used in greedy MIS:
    Score = 1 / (degree + 1) is standard, but we can improve by considering neighbor degrees.
    
    New Strategy: "Local Competitive Advantage"
    1. Compute a base score for all vertices: s_i = 1 / (deg_i + 1).
    2. For each vertex v, look at its neighbors.
       If v has a neighbor u with much higher s_u, v is "dominated" and should be deprioritized.
       If v has no neighbors with higher s_u, it is a "local champion" and should be prioritized.
       
    Even better: Simulate the greedy process on the full graph but use the scores to break ties 
    dynamically? No, the interface requires static scores.
    
    Let's try: 
    Score(v) = (Number of non-neighbors) / (Number of neighbors + 1)
    This favors vertices in sparse areas with high connectivity to the rest of the graph (which don't matter for independence) 
    but low connectivity to the set we are building.
    
    Actually, the most effective simple heuristic for Greedy MIS is:
    Pick vertex with minimum degree.
    To make it adaptive:
    Score(v) = - (degree(v) + sum of (1/degree(u)) for u in neighbors(v))
    
    Let's implement a "Local Simulation" proxy:
    For each vertex v, simulate picking v. 
    Benefit: 1
    Cost: Number of neighbors that are "strong" (high degree).
    
    Let's use: Score(v) = - (degree(v) * 1.0 + 0.5 * mean_neighbor_degree(v))
    This is similar to the current one but let's try a different family:
    
    FAMILY: Iterative Refinement / Belief Propagation style
    1. Initialize scores s_i = 1.
    2. For T steps:
       For each vertex i:
         s_i = 1 / (1 + sum_{j in N(i)} s_j)
    This is a soft version of the independent set constraint. Vertices with high scoring neighbors get low scores.
    This naturally propagates "conflict" information through the graph.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Ensure symmetry just in case, though input should be symmetric
    # adj = (adj + adj.T) / 2.0
    
    # Initialize scores
    # Start with inverse degree as a strong prior
    degrees = adj.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = 1.0 / (degrees + 1.0)
    
    # Iterative refinement (Belief Propagation / Mean Field)
    # s_i <- 1 / (1 + sum_{j in N(i)} s_j)
    # This encourages vertices to have low scores if their neighbors have high scores.
    
    num_iters = 20
    # Check time budget for iterations
    # Each iteration is O(N^2) due to matrix multiplication if done naively, 
    # but we can use sparsity or just numpy dot which is fast for 200x200.
    
    for _ in range(num_iters):
        if time.time() - start_time > 100:
            break
        
        # Sum of neighbor scores
        neighbor_score_sums = adj.dot(scores)
        
        # Update rule
        new_scores = 1.0 / (1.0 + neighbor_score_sums)
        
        # Damping for stability
        scores = 0.5 * scores + 0.5 * new_scores
        
    # The final scores represent the "probability" or "weight" of a vertex being in the MIS.
    # Higher score = more likely to be picked.
    
    # We can add a small penalty for high degree to break ties in favor of sparse vertices
    # if the BP scores are too uniform.
    # scores = scores / (degrees + 1.0) # This might be too strong.
    
    return scores
# EVOLVE-BLOCK-END
