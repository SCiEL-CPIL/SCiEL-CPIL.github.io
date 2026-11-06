"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(adjacency):
    """
    Assign a score to each vertex based on a 'sacrificial' heuristic.
    We prioritize vertices with low degree that are adjacent to high-degree vertices.
    Selecting such a vertex removes a small number of neighbors, but those neighbors
    likely have high degrees and would have blocked many other vertices if left in.
    This is distinct from eigenvector centrality, focusing on local structural leverage.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    if n == 1:
        return np.array([1.0])
    
    # Calculate degrees
    degrees = adj.sum(axis=1)
    
    # Calculate sum of neighbor degrees for each vertex
    # This is equivalent to the diagonal of A @ A, or row sums of A * degrees
    # sum_deg_neighbors[u] = sum(d[v] for v in neighbors(u))
    # Using matrix multiplication: A @ degrees gives vector where u-th element is sum(d[v] * A[u,v])
    neighbor_degree_sums = adj.dot(degrees)
    
    # Handle isolated vertices (degree 0)
    # If degree is 0, neighbor_degree_sums is 0.
    # We want to prioritize them? 
    # In MIS, isolated vertices are always picked.
    # The greedy decoder picks highest score first.
    # So isolated vertices should have very high scores.
    
    # Heuristic:
    # Score = -degree + lambda * neighbor_degree_sum
    # Let's normalize to avoid scale issues.
    # Typical degree in ER graph with n=200, p=0.1 is ~20.
    # neighbor_degree_sum ~ 20 * 20 = 400.
    
    # We want low degree to be good (negative contribution).
    # We want high neighbor degree sum to be good (positive contribution).
    
    # Let's use a weighted combination.
    # If a vertex has degree 1 and its neighbor has degree 100, picking it is great.
    # Score = -1 + 100 = 99.
    # If a vertex has degree 10 and neighbors have avg degree 10, sum=100.
    # Score = -10 + 100 = 90.
    # So the leaf attached to a hub is preferred. This is correct for MIS.
    
    # However, if we have a cluster of leaves attached to a hub:
    # Hub: degree K, neighbors are leaves (deg 1). Sum = K. Score = -K + K = 0.
    # Leaf: degree 1, neighbor is hub (deg K). Sum = K. Score = -1 + K = K-1.
    # Leaves have higher score than Hub. Leaves are picked first.
    # Once a leaf is picked, the Hub is blocked.
    # This is optimal: we pick all leaves, then Hub is gone. We get K vertices.
    # If we picked Hub first, we get 1 vertex.
    # So this heuristic strongly favors leaves.
    
    # What about a path? 1-2-3
    # d = [1, 2, 1]
    # n_sum[0] = d[1] = 2. Score[0] = -1 + 2 = 1.
    # n_sum[1] = d[0]+d[2] = 2. Score[1] = -2 + 2 = 0.
    # n_sum[2] = d[1] = 2. Score[2] = -1 + 2 = 1.
    # Vertices 0 and 2 have score 1. Vertex 1 has score 0.
    # Greedy picks 0 (or 2). Then 1 is blocked. 2 is blocked if 0 picked? No, 0 and 2 are not neighbors.
    # If pick 0, 1 is blocked. 2 is still available. Score 2 is still 1? 
    # The priority function is static. It doesn't update.
    # The decoder picks highest score. If 0 and 2 have same score, order is arbitrary (or index order).
    # Suppose 0 picked. 1 blocked. 2 picked. Set {0, 2}. Size 2. Optimal.
    
    # This heuristic seems robust.
    
    # To handle scale and ensure isolated vertices (deg 0, sum 0) are picked first:
    # Isolated: Score = 0.
    # Leaf of hub (K=100): Score = 99.
    # So Leaves > Isolated? 
    # If we pick Isolated first, it doesn't block anything.
    # If we pick Leaf first, it blocks Hub.
    # We should pick Isolated first? Or Leaf first?
    # It doesn't matter for size, as long as we pick both.
    # But if we have limited "slots" (we don't, we pick all valid), order matters for blocking.
    # Picking Isolated never blocks anything. Picking Leaf blocks Hub.
    # So we should pick Isolated vertices with highest priority?
    # Actually, if we pick Leaf, we block Hub. If we pick Isolated, we block nothing.
    # Both are safe.
    # But if we have a choice between a Leaf (blocking a Hub that blocks many) and an Isolated (blocking nothing),
    # the Leaf is more "valuable" in terms of preventing future conflicts?
    # No, the Greedy decoder just picks.
    # If we pick Isolated, it remains in the set.
    # If we pick Leaf, Hub is removed from consideration.
    # This is good.
    
    # Let's boost isolated vertices slightly to ensure they are picked early if scores are close?
    # Actually, 0 is lower than K-1. So Leaves are picked before Isolated.
    # Is that bad?
    # If I pick Leaf, I block Hub.
    # If I pick Isolated, I block nothing.
    # Result: Leaf in set, Isolated in set, Hub out.
    # Same result.
    
    # Let's add a small bonus for low degree to break ties or handle deg 0.
    # Score = neighbor_degree_sums - degrees
    # For deg 0, score = 0.
    # For deg 1, neighbor deg 1, score = 0.
    # For deg 1, neighbor deg 100, score = 99.
    
    # Let's return the scores.
    scores = neighbor_degree_sums - degrees
    
    # Optional: add a small randomization or tie-breaker? 
    # The task says "fixed greedy decoder". It likely uses `np.argsort` or similar.
    # Ties might be broken by index.
    
    return scores
# EVOLVE-BLOCK-END
