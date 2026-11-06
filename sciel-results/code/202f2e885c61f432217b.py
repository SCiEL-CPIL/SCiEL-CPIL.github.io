"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0
_SAFETY = 10.0  # seconds before the hard limit


def _greedy_decode(scores, adjacency):
    """Fixed greedy decoder: pick vertices in descending score, skip neighbours of picked."""
    n = len(scores)
    order = np.argsort(-scores, kind='stable')
    selected = np.zeros(n, dtype=bool)
    independent = np.zeros(n, dtype=bool)
    for v in order:
        if selected[v] or independent[v]:
            continue
        independent[v] = True
        selected[v] = True
        # Mark all neighbours of v as selected (blocked)
        neighbors = np.where(adjacency[v])[0]
        selected[neighbors] = True
    return independent


def priority(adjacency):
    """
    Assign a score to each vertex for the greedy Maximum Independent Set decoder.
    Uses a multi-criteria heuristic that balances:
      - low degree (fewer neighbours to block)
      - high 'independence potential' (neighbours with high degree are less valuable to block)
      - local structural features (clustering coefficient proxy, community structure)
    
    Returns: array of scores (higher = picked earlier)
    """
    start_time = time.time()
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    # Basic features
    deg = A.sum(axis=1)
    
    # Second-order features: for each vertex, the average degree of its neighbours
    # High avg-neighbor-degree means picking this vertex blocks high-degree nodes (good)
    # But also means the vertex is in a dense area
    if n > 0:
        deg_sum_neighbors = A @ deg  # sum of degrees of neighbours for each vertex
        avg_deg_neighbors = np.where(deg > 0, deg_sum_neighbors / deg, 0.0)
    else:
        avg_deg_neighbors = np.zeros(n)
    
    # Number of triangles each vertex participates in (approximate via A^2 diagonal)
    # A^2[i,j] = number of common neighbours; diagonal of A^2 = degree (not triangles)
    # Triangles through i: 0.5 * sum over neighbours j of (A[i,j] * (A @ A)[i,j])
    # More precisely: triangles_i = 0.5 * (A @ (A * A)) [i,i] ... let's use:
    # For each edge (i,j), common neighbors = (A @ A)[i,j]. Triangle count for i = 0.5 * sum_j A[i,j]*(A@A)[i,j]
    A2 = A @ A
    # triangles[i] = 0.5 * sum_j A[i,j] * A2[i,j]
    triangles = 0.5 * (A * A2).sum(axis=1)
    
    # Clustering coefficient proxy
    clustering = np.where(deg >= 2, triangles / (deg * (deg - 1) / 2.0), 0.0)
    
    # "Isolatedness": vertices with low degree and low clustering are easier to include
    # Score should favor: low degree, low clustering, and being a "bridge" (connects sparse parts)
    
    # Betweenness-like proxy: vertices whose removal disconnects more structure
    # Approximate by: degree * (1 - clustering) -- high degree but low clustering = bridge
    
    # Multi-criteria score:
    # We want to pick vertices that:
    # 1. Have low degree (block fewer vertices)
    # 2. Are not in dense clusters (clustering low)
    # 3. Have neighbours with high degree (so blocking them is "worth it" in the sense that
    #    high-degree neighbours would block many vertices if picked instead)
    
    # Normalize features to [0,1] range
    def _normalize(x):
        mn = x.min()
        mx = x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    norm_deg = _normalize(deg)
    norm_clustering = _normalize(clustering)
    norm_avg_deg_neighbors = _normalize(avg_deg_neighbors)
    
    # Score: we want low degree, low clustering, but high avg-neighbor-degree is a tradeoff
    # The key insight: picking a low-degree vertex in a sparse region is good.
    # Picking a high-degree vertex that has many low-degree neighbours is also good 
    # because it blocks many "easy" vertices that would otherwise be picked.
    
    # Let's think about it differently:
    # The greedy algorithm picks vertex v, then all its neighbours are blocked.
    # The "cost" of picking v is: we lose all neighbours of v from the independent set.
    # The "benefit" is: we gain v.
    # Net gain = 1 - (number of neighbours that would have been picked anyway)
    
    # A good heuristic: pick vertices where the "opportunity cost" is low.
    # Opportunity cost ≈ number of neighbours with low degree (they would likely be picked)
    
    # For each vertex v, count neighbours with degree <= deg[v]
    # These are the neighbours that would likely be picked in a degree-based ordering
    low_deg_neighbors = np.zeros(n)
    for i in range(n):
        if deg[i] > 0:
            neighbors = np.where(A[i] > 0)[0]
            low_deg_neighbors[i] = np.sum(deg[neighbors] <= deg[i])
    
    # Normalize
    norm_low_deg_neighbors = _normalize(low_deg_neighbors)
    
    # Final score: 
    # High score = good to pick early
    # We want: low degree, low clustering, few low-degree neighbours (less opportunity cost)
    # But also: if a vertex has high degree but all its neighbours are high-degree, it's still good
    
    # Let's use a weighted combination:
    # score = w1 * (1 - norm_deg) + w2 * (1 - norm_clustering) + w3 * (1 - norm_low_deg_neighbors)
    
    # Actually, let's reconsider. The seed uses -degree. Let's build on that.
    # The key improvement: account for the structure of the neighbourhood.
    
    # A vertex with low degree is good. But among low-degree vertices, 
    # prefer those whose neighbours are high-degree (so we're not "wasting" the pick 
    # on a vertex that would have blocked many low-degree vertices anyway).
    
    # Alternative: use the "core" structure. Vertices in the k-core with small k are easier to include.
    
    # Let's try a different approach: iterative refinement of scores
    # Start with -degree, then adjust based on local structure
    
    scores = -deg.copy()
    
    # Adjustment: for each vertex, if it has many neighbours with similar or lower degree,
    # it's "redundant" with them, so lower its score slightly.
    # If it has neighbours with much higher degree, it's a "leaf" in some sense, keep its score.
    
    # For each vertex i, compute the fraction of neighbours with degree <= deg[i]
    frac_low_deg_neighbors = np.where(deg > 0, low_deg_neighbors / deg, 0.0)
    
    # Vertices with high frac_low_deg_neighbors are in "dense" parts of the low-degree structure
    # They are more likely to conflict with each other, so we should pick fewer of them
    # Lower their score (make it more negative) to deprioritize
    
    # Actually, if they conflict with each other, we want to pick the best one among them.
    # The "best" one is the one that blocks the most high-degree vertices.
    
    # Let's compute: for each vertex, the maximum degree among its neighbours
    max_neighbor_deg = np.zeros(n)
    for i in range(n):
        if deg[i] > 0:
            neighbors = np.where(A[i] > 0)[0]
            max_neighbor_deg[i] = deg[neighbors].max()
    
    norm_max_neighbor_deg = _normalize(max_neighbor_deg)
    
    # Final score:
    # Base: -degree (lower degree is better)
    # Bonus: if the vertex has neighbours with high degree, it's more "valuable" to pick
    # Penalty: if the vertex is in a dense cluster (high clustering), it's less valuable
    
    # score = -deg + alpha * norm_max_neighbor_deg - beta * norm_clustering
    
    # Let's try:
    alpha = 1.0
    beta = 0.5
    
    scores = -deg + alpha * norm_max_neighbor_deg * deg.max() - beta * norm_clustering * deg.max()
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - _SAFETY:
        return scores
    
    # Try a second refinement: use the scores to simulate a partial greedy and adjust
    # This is expensive, so only do it if we have time
    
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - _SAFETY - 30:
        # Do a quick local refinement
        # For each vertex, consider swapping it with a neighbour
        # If swapping improves the independent set size, adjust scores
        
        # Simulate greedy with current scores
        independent = _greedy_decode(scores, A)
        base_size = independent.sum()
        
        # For each vertex not in the independent set, check if adding it (and removing its neighbours)
        # would improve the set
        # This is O(n^2) which is fine for n=200
        
        adjusted_scores = scores.copy()
        for i in range(n):
            if independent[i]:
                continue
            # Try adding i to the independent set
            # Remove all neighbours of i from the set
            neighbors = np.where(A[i] > 0)[0]
            new_size = base_size + 1 - np.sum(independent[neighbors])
            if new_size > base_size:
                # Adding i is beneficial; increase its score
                adjusted_scores[i] += 10.0
            else:
                # Adding i is not beneficial; decrease its score
                adjusted_scores[i] -= 5.0
        
        elapsed = time.time() - start_time
        if elapsed < _DEADLINE - _SAFETY:
            scores = adjusted_scores
    
    return scores

# EVOLVE-BLOCK-END
