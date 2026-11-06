"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices.
    Higher score means the vertex is more likely to be selected earlier by the greedy decoder.
    
    Strategy:
    1. Convert adjacency to boolean matrix.
    2. Compute degrees.
    3. For each vertex, compute the "independent set potential":
       - Low degree is good (fewer conflicts).
       - Neighbors with high degrees are good to remove (they block many others).
       - Use a composite score that balances these factors.
       
    We use a direct structural formula:
    score[i] = alpha * (mean_degree_of_neighbors[i] - degree[i]) + beta * (1 / (1 + degree[i]))
    
    This rewards vertices that:
    - Have neighbors with high average degree (removing them unblocks many)
    - Have low own degree (low conflict cost)
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Compute degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # Compute sum of degrees of neighbors for each vertex
    # neighbor_degree_sums[i] = sum(degrees[j] for j in neighbors of i)
    neighbor_degree_sums = adj.astype(float) @ degrees
    
    # Compute mean degree of neighbors
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_degrees = np.where(degrees > 0, neighbor_degree_sums / np.maximum(degrees, 1), 0.0)
    
    # Compute triangle counts: number of triangles each vertex participates in
    # For vertex i, triangles = sum over neighbors j,k of i that are adjacent to each other
    # This equals (1/2) * sum_j A[i,j] * (A @ A)[i,j]
    A2 = adj.astype(np.int32) @ adj.astype(np.int32)
    # triangles[i] = sum over j in neighbors of i of A2[i,j] / 2
    triangles = np.zeros(n, dtype=float)
    for i in range(n):
        if degrees[i] > 0:
            triangles[i] = 0.5 * np.sum(A2[i, adj[i]])
    
    # Compute 2-hop neighbor counts (number of vertices at distance exactly 2)
    # two_hop[i] = sum_j (A2[i,j] - A[i,j] - 1 if i!=j) ... actually:
    # vertices at distance 2 from i = sum_j A2[i,j] - degree[i] - 1 (exclude self and direct neighbors)
    two_hop = np.zeros(n, dtype=float)
    for i in range(n):
        if degrees[i] > 0:
            # A2[i,i] = degree[i] (number of 2-walks from i to i)
            # Total 2-walks from i to others = sum(A2[i,:]) - A2[i,i]
            # Subtract direct neighbors (they're at distance 1, not 2)
            two_hop[i] = np.sum(A2[i, :]) - A2[i, i] - degrees[i]
    
    # Deadline check after expensive computations
    if time.time() - start_time > 120:
        # Fallback: simple degree-based score
        return -degrees
    
    # Smooth (soft-max) transformation of neighbor degree signal:
    # Rewards vertices whose neighbors are "expensively blocked" in a smooth, bounded way.
    # Using log-sum-exp style: log(1 + exp(x)) gives a smooth max(0, x) surrogate.
    # Scale down to keep numerical stability.
    x = (mean_neighbor_degrees - degrees) / 2.0
    # softplus(x) = log(1 + exp(x)) for x > 0, ~x for large x, ~log(2) for very negative x
    softplus_x = np.where(x > 0, np.log1p(np.exp(np.minimum(x, 50))), np.log1p(np.exp(np.maximum(x, -50))))
    
    # Component 1: smooth structural benefit (softplus of the raw difference)
    component1 = softplus_x
    
    # Component 2: quadratic penalty for own degree (stronger penalty for high-degree vertices)
    # This captures that high-degree vertices block many others quadratically in expectation
    component2 = -0.1 * degrees**2 / (n + 1)
    
    # Component 3: small inverse-degree bonus for tie-breaking among similar vertices
    component3 = 0.1 / (1.0 + degrees)
    
    # Component 4: triangle penalty (vertices in many triangles have more local conflicts)
    # Normalize by max triangles to keep scale reasonable
    max_tri = np.max(triangles) if np.max(triangles) > 0 else 1.0
    component4 = -0.3 * triangles / max_tri
    
    # Component 5: 2-hop bonus (vertices with many 2-hop neighbors can "unblock" more when selected)
    # Normalized by max 2-hop count
    max_2hop = np.max(two_hop) if np.max(two_hop) > 0 else 1.0
    component5 = 0.2 * two_hop / max_2hop
    
    # Combine: structural benefit dominates, quadratic penalty shapes the tradeoff
    scores = component1 + component2 + component3 + component4 + component5
    
    # Check deadline
    if time.time() - start_time > 140:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
