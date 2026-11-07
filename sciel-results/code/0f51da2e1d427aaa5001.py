# EVOLVE-BLOCK-START
import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edges_to_set(edges):
    s = set()
    for u, v in edges:
        s.add((min(u, v), max(u, v)))
    return s

def _find_biclique_from_anchor(adj, remaining, anchor):
    """Find a biclique anchored at `anchor` using only remaining edges."""
    n = P['n']
    # Right: neighbors of anchor that have remaining edges to anchor
    right = set()
    for v in adj[anchor]:
        if (min(anchor, v), max(anchor, v)) in remaining:
            right.add(v)
    if not right:
        return None, None
    
    # Left: start with anchor, try to add vertices connected to all of right
    left = {anchor}
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if v in left:
                continue
            # v must connect to all of right (in original graph)
            if not all(v in adj[w] for w in right):
                continue
            # v must have at least one remaining edge to right
            has_rem = False
            for w in right:
                if (min(v, w), max(v, w)) in remaining:
                    has_rem = True
                    break
            if has_rem:
                left.add(v)
                changed = True
    
    # Shrink right: must connect to all of left
    new_right = set()
    for w in right:
        if all(w in adj[u] for u in left):
            new_right.add(w)
    right = new_right
    
    # Final shrink left: must connect to all of right
    new_left = set()
    for u in left:
        if all(u in adj[v] for v in right):
            new_left.add(u)
    left = new_left
    
    if not left or not right:
        return None, None
    
    return list(left), list(right)

def _greedy_max_biclique_partition(adj, remaining, deadline):
    """Repeatedly find the largest biclique in remaining edges."""
    n = P['n']
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Count degree in remaining
        deg = {}
        for u, v in remaining:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        
        # Try top anchors by remaining degree
        sorted_v = sorted(deg.keys(), key=lambda x: -deg[x])
        max_anchors = min(len(sorted_v), 30)
        
        best_left, best_right, best_edges = None, None, 0
        
        for anchor in sorted_v[:max_anchors]:
            left, right = _find_biclique_from_anchor(adj, remaining, anchor)
            if left is None:
                continue
            # Count how many remaining edges this biclique covers
            count = 0
            for u in left:
                for v in right:
                    if (min(u, v), max(u, v)) in remaining:
                        count += 1
            if count > best_edges:
                best_edges = count
                best_left = left
                best_right = right
        
        if best_left is None or best_edges == 0:
            # Fallback: take one edge
            e = next(iter(remaining))
            u, v = e
            bicliques.append([[u], [v]])
            remaining.discard(e)
            continue
        
        # Remove covered edges
        for u in best_left:
            for v in best_right:
                e = (min(u, v), max(u, v))
                remaining.discard(e)
        
        bicliques.append([list(best_left), list(best_right)])
    
    # Handle leftover edges
    for e in list(remaining):
        u, v = e
        bicliques.append([[u], [v]])
        remaining.discard(e)
    
    return bicliques

def _validate(bicliques, edges_set):
    """Check if biclique partition is valid."""
    n = P['n']
    adj = _build_adj()
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                e = (min(u, v), max(u, v))
                if e not in edges_set:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == edges_set

def _try_merge(bicliques, adj, edges_set):
    """Try to merge pairs of bicliques if valid."""
    n = P['n']
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(bicliques):
            j = i + 1
            while j < len(bicliques):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try all 4 merge configurations
                configs = [
                    (set(L1) | set(L2), set(R1)),  # merge lefts, keep R1
                    (set(L1) | set(L2), set(R2)),  # merge lefts, keep R2
                    (set(L1), set(R1) | set(R2)),  # keep L1, merge rights
                    (set(L2), set(R1) | set(R2)),  # keep L2, merge rights
                    (set(L1) | set(L2), set(R1) | set(R2)),  # full merge
                ]
                
                merged = False
                for new_L, new_R in configs:
                    if not new_L or not new_R:
                        continue
                    # Check all cross-edges exist in graph
                    valid = True
                    for u in new_L:
                        for v in new_R:
                            if u == v:
                                valid = False
                                break
                            if v not in adj[u]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    # Check that the union of edges from bicliques i and j 
                    # equals the cross edges of the new biclique
                    # i.e., no edge in new_L x new_R is covered by other bicliques
                    new_edges = set()
                    for u in new_L:
                        for v in new_R:
                            new_edges.add((min(u, v), max(u, v)))
                    
                    # Edges from bicliques i and j
                    old_edges = set()
                    for u in L1:
                        for v in R1:
                            old_edges.add((min(u, v), max(u, v)))
                    for u in L2:
                        for v in R2:
                            old_edges.add((min(u, v), max(u, v)))
                    
                    # All new edges must be in old_edges (no new edges introduced)
                    if new_edges <= old_edges:
                        # Valid merge
                        bicliques[i] = [list(new_L), list(new_R)]
                        del bicliques[j]
                        changed = True
                        merged = True
                        break
                
                if merged:
                    break
                j += 1
            if changed:
                break
            i += 1
    return bicliques

def _local_search(bicliques, adj, edges_set, deadline):
    """Try to reduce number of bicliques via local moves."""
    n = P['n']
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing a vertex from a biclique and redistributing
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            L, R = bicliques[i]
            if len(L) <= 1 and len(R) <= 1:
                continue
            
            # Try removing each vertex from L
            for vi, v in enumerate(L):
                new_L = [x for x in L if x != v]
                if not new_L:
                    continue
                # Edges incident to v in this biclique
                v_edges = set()
                for w in R:
                    v_edges.add((min(v, w), max(v, w)))
                
                # Check if these edges can be absorbed by other bicliques
                # or if we can form a new smaller biclique
                # For now, just check if removing v and keeping [new_L, R] is valid
                # and the v_edges can be covered elsewhere
                # This is complex; skip for now and try merge instead
                pass
            
            # Try removing each vertex from R
            for vi, v in enumerate(R):
                new_R = [x for x in R if x != v]
                if not new_R:
                    continue
                pass
        
        # Try merging
        new_bicliques = _try_merge(bicliques, adj, edges_set)
        if len(new_bicliques) < len(bicliques):
            bicliques = new_bicliques
            improved = True
    
    return bicliques

def _spectral_partition_edges(adj, remaining_edges, k_clusters, deadline):
    """
    Use spectral clustering on the line graph to partition edges into k_clusters.
    Returns a list of sets of edges.
    """
    if not remaining_edges:
        return []
    
    edges_list = list(remaining_edges)
    m = len(edges_list)
    if m <= k_clusters:
        return [set([e]) for e in edges_list]
    
    # Build Line Graph Adjacency
    # Node i in line graph corresponds to edges_list[i]
    # Edge i-j if edges_list[i] and edges_list[j] are compatible
    # Compatible: share a vertex OR form K2,2
    
    # Preprocess edges for fast lookup
    edge_to_idx = {e: i for i, e in enumerate(edges_list)}
    
    # Compatibility matrix
    # We can use a sparse representation or just a list of neighbors
    line_adj = [set() for _ in range(m)]
    
    for i in range(m):
        u1, v1 = edges_list[i]
        for j in range(i + 1, m):
            u2, v2 = edges_list[j]
            
            # Check compatibility
            compatible = False
            
            # Case 1: Share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                compatible = True
            else:
                # Case 2: Disjoint, check if K2,2
                # Edges (u1,v1) and (u2,v2).
                # Need (u1,v2) and (u2,v1) to be in graph.
                if (min(u1, v2), max(u1, v2)) in adj[u1] and (min(u2, v1), max(u2, v1)) in adj[u2]:
                    # Note: adj[u1] contains neighbors. (u1,v2) is edge if v2 in adj[u1].
                    if v2 in adj[u1] and v1 in adj[u2]:
                        compatible = True
            
            if compatible:
                line_adj[i].add(j)
                line_adj[j].add(i)
                
            if time.time() > deadline:
                return [] # Timeout, abort spectral
    
    # Build Laplacian
    # L = D - A
    degrees = [len(nei) for nei in line_adj]
    # Normalized Laplacian L_sym = I - D^-1/2 A D^-1/2
    # Or just use standard Laplacian for simplicity if m is small
    
    # Construct A
    A = np.zeros((m, m))
    for i in range(m):
        for j in line_adj[i]:
            A[i, j] = 1.0
    
    D = np.diag(degrees)
    L = D - A
    
    # Eigen decomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return []
    
    # Take k smallest eigenvectors
    k = min(k_clusters, m)
    U = eigenvectors[:, :k]
    
    # K-Means clustering
    # Simple K-Means implementation
    # Initialize centroids randomly
    centroids = U[np.random.choice(m, k, replace=False)]
    
    for _ in range(10):
        # Assign points to nearest centroid
        # Calculate distances
        # dist[i, j] = || U[i] - centroids[j] ||^2
        # = ||U[i]||^2 - 2 U[i] . centroids[j] + ||centroids[j]||^2
        U_sq = np.sum(U**2, axis=1)
        C_sq = np.sum(centroids**2, axis=1)
        D = U_sq[:, np.newaxis] - 2 * U.dot(centroids.T) + C_sq[np.newaxis, :]
        
        labels = np.argmin(D, axis=1)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if np.sum(mask) > 0:
                new_centroids[j] = U[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j] # Keep old if empty
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Group edges
    clusters = [set() for _ in range(k)]
    for i, label in enumerate(labels):
        clusters[label].add(edges_list[i])
    
    # Filter empty clusters
    clusters = [c for c in clusters if c]
    
    return clusters

def _solve_cluster_biclique(adj, cluster_edges, deadline):
    """
    Given a set of edges that are likely in one biclique, find the maximal biclique covering them.
    If they don't fit in one, fallback to greedy on the cluster.
    """
    if not cluster_edges:
        return []
    
    # Try to find a biclique that covers all these edges.
    # This is hard. Instead, just run greedy on this subset?
    # Or, pick the highest degree node in the cluster and expand.
    
    # Let's just use the greedy logic but restricted to these edges?
    # Actually, if the spectral clustering is good, these edges *should* form a biclique.
    # We can try to find L and R such that L x R covers cluster_edges.
    
    # Heuristic:
    # Pick an edge (u,v) in cluster. Assume u in L, v in R.
    # Expand L: all u' such that u' connected to all R.
    # Expand R: all v' such that v' connected to all L.
    
    # Start with the first edge
    u0, v0 = next(iter(cluster_edges))
    
    L = {u0}
    R = {v0}
    
    changed = True
    while changed:
        changed = False
        # Expand L
        new_L = set(L)
        for u in range(P['n']):
            if u in L: continue
            if all(u in adj[v] for v in R):
                # Check if u has any edge to R in cluster_edges
                has_edge = False
                for v in R:
                    if (min(u,v), max(u,v)) in cluster_edges:
                        has_edge = True
                        break
                if has_edge:
                    new_L.add(u)
                    changed = True
        L = new_L
        
        # Expand R
        new_R = set(R)
        for v in range(P['n']):
            if v in R: continue
            if all(v in adj[u] for u in L):
                # Check if v has any edge to L in cluster_edges
                has_edge = False
                for u in L:
                    if (min(u,v), max(u,v)) in cluster_edges:
                        has_edge = True
                        break
                if has_edge:
                    new_R.add(v)
                    changed = True
        R = new_R
    
    # Check if this biclique covers all cluster_edges
    covered = set()
    for u in L:
        for v in R:
            covered.add((min(u,v), max(u,v)))
            
    if cluster_edges <= covered:
        # Valid biclique for this cluster
        return [[list(L), list(R)]]
    else:
        # Fallback: Greedy on the cluster
        remaining = set(cluster_edges)
        return _greedy_max_biclique_partition(adj, remaining, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    adj = _build_adj()
    edges_set = _edges_to_set(P['edges'])
    n = P['n']
    
    best = []
    
    # Strategy 1: Standard Greedy
    remaining1 = set(edges_set)
    result1 = _greedy_max_biclique_partition(adj, remaining1, deadline)
    if _validate(result1, edges_set):
        best = result1
    
    # Strategy 2: Spectral Partitioning
    if time.time() < deadline - 10:
        # Try different numbers of clusters
        for k in [5, 10, 15, 20]:
            if time.time() > deadline - 5:
                break
            
            clusters = _spectral_partition_edges(adj, edges_set, k, deadline)
            if not clusters:
                continue
            
            bicliques = []
            for cluster in clusters:
                cluster_bicliques = _solve_cluster_biclique(adj, cluster, deadline)
                bicliques.extend(cluster_bicliques)
            
            if _validate(bicliques, edges_set) and len(bicliques) < len(best):
                best = bicliques
    
    # Local Search on best
    if best and time.time() < deadline - 5:
        improved = _local_search(list(best), adj, edges_set, deadline - 5)
        if _validate(improved, edges_set) and len(improved) <= len(best):
            best = improved
    
    # Final fallback
    if not best or not _validate(best, edges_set):
        best = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                best.append([neighbors, [u]])
    
    # Convert to proper types
    result = []
    for left, right in best:
        result.append([list(left), list(right)])
    
    return result

# EVOLVE-BLOCK-END
