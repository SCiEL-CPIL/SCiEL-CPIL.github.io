"""Guided local search penalty rule for TSP using edge centrality (2-opt blocking) penalties."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy:
    - Edge Centrality Penalty: Instead of just penalizing historical usage, we measure
      how "central" or "blocking" each edge in the current tour is. 
    - An edge is considered central if removing it via a 2-opt move results in a tour 
      that is still a local optimum or close to it, meaning the edge is part of a 
      robust basin. Alternatively, we look at how many potential 2-opt moves are 
      "blocked" or made less attractive because of the edge's presence.
    - Specifically, for each edge (u, v) in the current tour, we count how many 
      2-opt moves involving (u, v) would result in a tour length that is NOT 
      significantly worse than the current tour. If an edge is part of many "near-optimal" 
      swaps, it is a stable part of the basin and should be penalized heavily to 
      force escape.
    - However, a simpler and more robust proxy for "basin stability" is the number of 
      2-opt moves that *would* improve the tour if this edge were removed or broken. 
      Actually, standard GLS penalizes edges in the current local optimum. 
      The key insight for a structural change is to penalize edges based on their 
      *structural importance* in the current tour's connectivity.
      
    Let's refine the "Centrality" idea:
    1. Identify all edges in the current local_opt_tour.
    2. For each such edge, calculate a "centrality score". 
       A high centrality score means the edge is heavily "supported" by the rest of the tour structure, 
       making it hard to break without large disruption. 
       We can approximate this by looking at the degrees of the nodes or the local geometry.
       
    Alternative Structural Approach: **Subgraph Density Penalty**
    - The current tour is a Hamiltonian cycle. 
    - Edges that are part of dense clusters of short edges are likely to be part of the "core" 
      of the optimal solution. 
    - We penalize edges based on the sum of distances to their k-nearest neighbors (excluding tour neighbors). 
    - If an edge connects two nodes that are both well-connected to other nearby nodes, 
      it might be a "long-range" bridge or a "short-range" core edge. 
      
    Let's try a different structural metric: **Edge Flexibility**
    - For each edge (u, v) in the tour, consider the 2-opt moves that break this edge. 
    - If there are few 2-opt moves that break (u, v) and result in a tour length close to the 
      current optimum, the edge is "rigid" and part of a tight basin. 
    - If there are many such moves, the edge is "flexible" and easy to escape. 
    - We want to penalize the "rigid" edges to break the basin.
    
    Implementation Plan:
    1. Get the current tour length L.
    2. For each edge (u, v) in the tour, we want to know how "hard" it is to remove.
    3. A proxy for "hard to remove" is the penalty incurred by the *best* 2-opt move that 
       removes (u, v). If the best move that removes (u, v) is very bad (large increase), 
       then (u, v) is a "critical" edge holding the tour together in this basin. 
       We should penalize it strongly to force the search to find a way to break it.
       Conversely, if there is a 2-opt move that removes (u, v) with a small increase, 
       the edge is not critical, and we can penalize it less.
       
    Wait, if the edge is critical (hard to remove), penalizing it makes it even harder to remove 
    in the perturbed matrix, which might trap us further. 
    Guided Local Search typically penalizes edges in the current solution to *discourage* 
    their reuse. The goal is to make the current solution *worse* in the perturbed space so 
    that the 2-opt on the perturbed space finds a *different* local optimum.
    
    So, if an edge is "critical" to the current basin, it is likely a "good" edge in the true 
    space. Penalizing it will force the search to avoid it, potentially leading to a different 
    basin. This is the standard GLS logic.
    
    The structural change here is using **2-opt Regret** to scale the penalty.
    - Standard GLS: Penalty = alpha * count.
    - Regret GLS: Penalty = alpha * count * (1 + regret_factor).
    - Where regret_factor is high for edges that are "expensive to replace".
    
    Let's calculate the "regret" for each edge in the current tour.
    Regret(u, v) = min_{2-opt move m removing (u,v)} (Length(Tour_m) - Length(Tour_current))
    If no such move exists (impossible for a cycle), or if the min increase is very large, 
    the edge is highly specialized.
    
    To keep it efficient (N=100, 20 rounds), we can approximate this.
    For each edge (u, v) in the tour, we look at the 4 edges that could replace it in a 2-opt 
    move? No, a 2-opt move removes 2 edges and adds 2.
    To remove edge (u, v), we must pick another edge (x, y) and reverse the path between them.
    This is O(N^2) per edge, which is O(N^3) total. For N=100, 10^6 operations per call. 
    20 calls * 3 instances * 10^6 = 6 * 10^7 operations. This is feasible in Python/NumPy 
    if vectorized, but Python loops might be slow.
    
    Let's use a simpler structural proxy: **Local Density**
    For each edge (u, v), calculate the number of nodes w (not u or v) such that 
    dist(u, w) + dist(w, v) < dist(u, v) + epsilon.
    This measures how "shortcut" the edge is. If there are many such w, the edge is a 
    "long" edge that could be replaced by a path through w. If there are few, it is a 
    "tight" edge.
    
    Actually, let's stick to the **Regret** idea but approximate it.
    The cost of a 2-opt move removing edges (u, v) and (x, y) is:
    delta = dist(u, x) + dist(v, y) - dist(u, v) - dist(x, y)  (if reversing u-x path)
    or delta = dist(u, y) + dist(v, x) - dist(u, v) - dist(x, y) (if reversing u-y path)
    
    We want the minimum delta for all x, y such that the move is valid.
    This is computationally heavy.
    
    Alternative: **Edge Uniqueness**
    For each edge (u, v) in the tour, check if it is among the top-k shortest edges incident 
    to u or v. If it is a very short edge relative to its neighbors, it is likely part of the 
    global optimum structure. Penalize it more.
    
    Let's try this:
    1. For each node u in the tour, find the k-shortest edges incident to u (k=5).
    2. If the tour edge (u, v) is in this top-k, it is "structurally important".
    3. Assign a higher penalty multiplier to these edges.
    
    This is O(N * N log N) or O(N^2) which is fast.
    
    Let's combine this with the usage count.
    Penalty(u, v) = base_penalty * usage_count(u, v) * importance_factor(u, v)
    
    importance_factor = 2.0 if (u, v) is in top-3 shortest edges for u OR v, else 1.0.
    
    This is a structural change: it uses local geometric structure (neighborhood density) 
    to modulate the penalty, rather than just global history.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()

    D = edge_distance.copy()

    # Calculate average edge length for scaling
    tour_len = 0.0
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[u, v]
    avg_edge_len = tour_len / n

    # Base penalty based on usage
    max_count = np.max(edge_n_used)
    if max_count == 0:
        max_count = 1.0
    normalized_counts = edge_n_used / max_count
    base_penalty = 1.0 * avg_edge_len * (normalized_counts ** 2)

    # Structural Importance: 
    # For each node, identify the k shortest incident edges.
    # If a tour edge is among these, it is "critical".
    k = 3
    importance_matrix = np.zeros((n, n), dtype=float)
    
    # We only need to check edges in the current tour
    # To be efficient, we can just compute the importance for all edges, 
    # but we only apply it to tour edges? 
    # Actually, GLS usually penalizes all edges or just tour edges.
    # The current program penalizes all edges based on usage.
    # Let's keep that, but modulate the penalty for tour edges based on importance.
    
    # Compute k-shortest neighbors for each node
    # D is symmetric.
    # For each node i, sort D[i] and take indices of smallest k (excluding i itself)
    # This is O(N^2 log N) which is fine for N=100.
    
    # Vectorized approach for k-shortest is tricky, loop is fine for N=100.
    for i in range(n):
        # Get distances from i to all others
        dists = D[i].copy()
        dists[i] = np.inf
        # Get k smallest indices
        # argsort is O(N log N)
        idx_sorted = np.argsort(dists)
        k_nearest = idx_sorted[:k]
        for j in k_nearest:
            importance_matrix[i, j] = 1.0
            importance_matrix[j, i] = 1.0

    # Now, modify the penalty matrix
    # For edges in the current tour, if they are "important" (in k-nearest of either endpoint),
    # multiply their penalty by a factor (e.g., 2.0).
    # This encourages breaking "critical" edges to escape the basin.
    
    penalty_matrix = base_penalty.copy()
    
    # Identify tour edges
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if importance_matrix[u, v] > 0:
            # This edge is structurally critical
            # Boost the penalty
            penalty_matrix[u, v] *= 2.0
            penalty_matrix[v, u] *= 2.0

    D += penalty_matrix

    # Ensure symmetry
    D = (D + D.T) / 2.0
    
    # Ensure finiteness
    if not np.all(np.isfinite(D)):
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
