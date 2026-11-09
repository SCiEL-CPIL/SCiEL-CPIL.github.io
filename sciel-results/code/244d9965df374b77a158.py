"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    n = adjacency.shape[0]
    deadline = time.time() + 120.0  # Safety margin
    
    # Convert to adjacency lists for efficient traversal
    adj = adjacency.astype(bool)
    deg = adj.sum(axis=1)
    
    # Compute initial scores using a combination of local features
    # then refine with BFS-based territory estimation
    
    # Local features:
    # 1. Negative degree (prefer low-degree vertices)
    # 2. Mean neighbor degree (prefer vertices with low-degree neighbors)
    # 3. Number of common neighbors (prefer vertices with fewer shared neighbors)
    
    # Mean neighbor degree: for vertex i, mean of degrees of its neighbors
    # adj[i] is bool vector of neighbors
    # neighbor_deg_sum[i] = sum of deg[j] for j in neighbors of i
    neighbor_deg_sum = adj @ deg
    
    # Safe division: where deg[i] > 0
    mean_neighbor_deg = np.zeros(n)
    mask = deg > 0
    mean_neighbor_deg[mask] = neighbor_deg_sum[mask] / deg[mask]
    
    # Number of common neighbors: for each pair of neighbors of i
    # This is expensive; use a proxy: degree of degree (second-order)
    # Or use: number of triangles involving i
    # Triangles: (adj^2 @ adj) / 2 gives triangle count per vertex
    adj2 = adj @ adj
    triangle_count = (adj2 * adj).sum(axis=1) / 2.0
    
    # Initial score: weighted combination
    # Lower degree is better, lower mean neighbor degree is better, fewer triangles is better
    # Normalize each to [0, 1]
    
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    norm_deg = normalize(deg)
    norm_mnd = normalize(mean_neighbor_deg)
    norm_tri = normalize(triangle_count)
    
    # Initial scores: we want to select vertices with low deg, low mnd, low tri
    # So score = -(w1*norm_deg + w2*norm_mnd + w3*norm_tri)
    # But we'll use BFS to refine
    
    # BFS-based territory estimation:
    # For each vertex, simulate greedy selection starting with that vertex,
    # and count how many vertices end up in the independent set.
    # This is O(n^2) per vertex which is too slow for n=200 (200^3 = 8M ops, OK)
    # But we need to be careful about the deadline.
    
    # Actually, let's do a simpler approximation:
    # For each vertex i, compute the size of the "exclusive region" - 
    # nodes that are closer to i than to any other vertex with similar priority.
    
    # Alternative: use a multi-source BFS approach where each vertex starts with 
    # its initial score, and we propagate influence.
    
    # Simpler approach: for each vertex, compute the number of vertices that 
    # would be "claimed" by i in a greedy process where i is selected first.
    
    # For efficiency, we'll sample a subset of vertices for full simulation
    # and use interpolation for others.
    
    # Let's try: for each vertex, compute the size of the connected component 
    # in the subgraph induced by vertices with degree <= deg[i]
    # This is a proxy for "how many low-degree vertices i can coexist with"
    
    # Actually, let's do the BFS territory approach but optimized:
    # For vertex i, the "territory" is the set of vertices v such that:
    # - v is not adjacent to i
    # - v has no neighbor with higher priority than i (in the initial ordering)
    
    # Let's use the initial score to define an ordering, then for each vertex,
    # count how many non-adjacent vertices have all their neighbors with lower 
    # initial scores than i.
    
    # Sort vertices by initial score (descending = selected first)
    initial_score = -(0.5 * norm_deg + 0.3 * norm_mnd + 0.2 * norm_tri)
    
    # For each vertex i, territory[i] = number of vertices v (v != i, not adj to i)
    # such that for all neighbors u of v, initial_score[u] < initial_score[i]
    # This means v would not be blocked by any higher-priority vertex.
    
    # This is still O(n^2 * avg_deg) which for n=200 is manageable.
    
    territory = np.zeros(n)
    for i in range(n):
        # Count vertices v that are:
        # 1. Not i
        # 2. Not adjacent to i
        # 3. All neighbors of v have initial_score < initial_score[i]
        not_adj = ~adj[i]
        not_adj[i] = False  # exclude self
        
        count = 0
        for v in np.where(not_adj)[0]:
            # Check if all neighbors of v have lower score than i
            neighbors_v = np.where(adj[v])[0]
            if len(neighbors_v) == 0:
                count += 1
            else:
                if np.all(initial_score[neighbors_v] < initial_score[i]):
                    count += 1
        territory[i] = count
    
    # Combine: final score should favor vertices with high territory
    # But also keep the local features
    norm_territory = normalize(territory)
    
    final_score = 0.6 * norm_territory - 0.2 * norm_deg - 0.1 * norm_mnd - 0.1 * norm_tri
    
    # Check deadline
    if time.time() > deadline - 5.0:
        # Return best we have
        return final_score
    
    return final_score
# EVOLVE-BLOCK-END
