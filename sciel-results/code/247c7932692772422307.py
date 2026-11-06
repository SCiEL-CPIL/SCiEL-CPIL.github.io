"""Rewritten program for Maximum Independent Set greedy priority."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    n = adjacency.shape[0]
    adj = adjacency.astype(np.float64)
    
    # Calculate degrees
    deg = adj.sum(axis=1)
    
    # Calculate sum of neighbor degrees
    neighbor_deg_sum = adj @ deg
    
    # Calculate mean neighbor degree (avoid division by zero)
    mean_neighbor_deg = np.zeros(n)
    mask = deg > 0
    mean_neighbor_deg[mask] = neighbor_deg_sum[mask] / deg[mask]
    
    # Calculate second-order neighborhood information
    # Sum of degrees of neighbors' neighbors (excluding self and direct neighbors)
    # A@A gives number of length-2 paths between vertices
    A2 = adj @ adj
    # For vertex i, (A2[i]) = sum over neighbors j of deg[j] = neighbor_deg_sum[i]
    # We want the "independent" contribution: neighbors of neighbors not adjacent to i
    # A2[i,j] counts paths i->k->j. If i and j are not adjacent, this counts common neighbors.
    # For a vertex i, the number of vertices at distance 2 is related to A2.
    
    # Let's think about what makes a good greedy choice for MIS:
    # 1. Low degree is good (fewer conflicts)
    # 2. Neighbors with high degree are bad (they block many other vertices)
    # 3. Neighbors with low degree are less bad (they block fewer other vertices)
    # 4. Vertices whose neighbors are close to each other (high clustering) are worse
    #    because picking them blocks a "dense" region
    
    # Clustering coefficient-like measure: for vertex i, the fraction of edges among its neighbors
    # That's expensive to compute exactly, but we can approximate.
    
    # Alternative: use the "effective degree" which accounts for how much picking i blocks
    # Picking vertex i blocks all its neighbors. The "cost" is related to how many
    # independent set candidates those neighbors would have blocked.
    
    # A better heuristic: score = -deg + alpha * (sum of 1/deg[neighbors]) 
    # because neighbors with low degree are "cheaper" to block
    
    # Let's compute sum of inverse degrees of neighbors
    inv_deg = np.zeros(n)
    inv_deg[deg > 0] = 1.0 / deg[deg > 0]
    
    # Sum of inverse degrees of neighbors
    sum_inv_neighbor_deg = adj @ inv_deg
    
    # Also consider: the number of "free" vertices that would be lost
    # When we pick vertex i, we lose all its neighbors from consideration.
    # But some of those neighbors might have been picked anyway if they had high priority.
    
    # Let's try a combination:
    # Primary: low degree
    # Secondary: neighbors with low degree (less harmful to block)
    # Tertiary: lower clustering among neighbors
    
    # For ER graphs, let's think about what the optimal greedy order is.
    # The greedy algorithm picks vertices in descending score order.
    # We want to maximize the number of vertices picked.
    
    # Key insight: if two vertices have the same degree, we prefer the one whose
    # neighbors have lower degrees, because those neighbors are less "valuable" 
    # (they would have lower priority anyway and are less likely to be picked).
    
    # Another insight: vertices in sparse regions are better to pick because
    # they don't block many candidates.
    
    # Let's use a weighted combination:
    # score = -deg + beta * mean_neighbor_deg * (-1) + gamma * sum_inv_neighbor_deg
    
    # Actually, let me think more carefully. The greedy decoder picks in descending
    # score order. So higher score = picked earlier.
    
    # If we pick a vertex with many high-degree neighbors, we block those high-degree
    # vertices, which is good because high-degree vertices block many others.
    # Wait, that's the opposite! If we pick a vertex whose neighbors are high-degree,
    # we're "sacrificing" high-degree vertices, which is good because they would have
    # blocked many other vertices.
    
    # So: prefer vertices whose neighbors have HIGH degrees (we block the "bad" vertices)
    # AND prefer vertices with LOW degree (we block few vertices)
    
    # This is a trade-off. Let's try:
    # score = -deg + alpha * mean_neighbor_deg
    
    # The current program uses -deg * 10 + mean_neighbor_deg. Let me tune the ratio.
    
    # For ER(200, p), typical degree is ~p*199. If p=0.1, deg~20.
    # mean_neighbor_deg is similar range.
    
    # Let's try different weights and see what makes sense.
    # The key question is: how much does blocking a high-degree neighbor help?
    
    # If I pick vertex i with degree d_i, I block d_i neighbors.
    # The "value" of blocking neighbor j is roughly deg[j] (the number of vertices it would block).
    # So the total "benefit" of picking i is sum of deg[j] for j in neighbors(i).
    # But I also lose the opportunity to pick any of those neighbors.
    
    # Net benefit ≈ sum(deg[j] for j in N(i)) - d_i * (expected value of a random vertex)
    
    # This is getting complex. Let me just try a few combinations.
    
    # Option 1: -deg + mean_neighbor_deg (current with weight 1)
    # Option 2: -deg * 2 + mean_neighbor_deg
    # Option 3: -deg + 2 * mean_neighbor_deg
    # Option 4: -deg + sum_inv_neighbor_deg
    
    # Let me also consider: the "independent set potential" of a vertex
    # A vertex with low degree and neighbors with low degree is in a "sparse" region
    # where we can pick many vertices.
    
    # Let's try: score = -(deg - mean_neighbor_deg)
    # This rewards vertices where neighbors have higher degree than the vertex itself.
    
    # Actually, for the greedy MIS, the best strategy is often:
    # 1. Pick vertices with lowest degree first (classic heuristic)
    # 2. Among same-degree vertices, prefer those with neighbors that have low degree
    
    # Let me try a simple refinement:
    # score = -deg + 0.1 * mean_neighbor_deg
    
    # Or even better, let's think about the "regret" of picking a vertex.
    # If I don't pick vertex i, I might be able to pick some of its neighbors.
    # The best case is if I can pick all neighbors (unlikely, they're adjacent to each other).
    # The worst case is if I can pick none of them.
    
    # For ER graphs, the neighbors of a vertex are mostly non-adjacent to each other
    # (low clustering). So if I don't pick i, I might pick ~deg[i]/2 neighbors on average.
    
    # This is getting too theoretical. Let me just try a few simple combinations.
    
    # Let's use: score = -deg + lambda * mean_neighbor_deg
    # and try to find a good lambda.
    
    # Based on the current score of 1.0075 with -deg*10 + mean_neighbor_deg,
    # the weight of 10 on degree seems to work. Let me try variations.
    
    # Let me also add a small bonus for vertices with many "low-degree" neighbors,
    # as blocking low-degree neighbors is less costly.
    
    # Final strategy:
    # score = -deg * w1 + mean_neighbor_deg * w2 + sum_inv_neighbor_deg * w3
    
    # Let's try w1=5, w2=1, w3=0.5
    scores = -deg * 5.0 + mean_neighbor_deg * 1.0 + sum_inv_neighbor_deg * 0.5
    
    # Check deadline
    if time.time() > deadline:
        return -deg
    
    return scores
# EVOLVE-BLOCK-END
