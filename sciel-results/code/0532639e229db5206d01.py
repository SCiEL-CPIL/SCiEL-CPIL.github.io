"""Clique partition via complement-graph spectral clustering + local refinement."""
# EVOLVE-BLOCK-START
import time
import random as _random
import math
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 115.0


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _build_complement_adj(edges, n):
    """
    Build complement adjacency bitmasks.
    comp_adj[u] has bit v set if (u,v) is NOT an edge in G.
    """
    comp_adj = [0] * n
    # Start with all possible neighbors
    for u in range(n):
        comp_adj[u] = ((1 << n) - 1) & ~(1 << u)
    # Remove actual edges
    for u, v in edges:
        comp_adj[u] &= ~(1 << v)
        comp_adj[v] &= ~(1 << u)
    return comp_adj


def _spectral_clustering(comp_adj, n, k, rng):
    """
    Use spectral clustering on the complement graph to find k independent sets
    (which are cliques in the original graph).
    
    The complement graph H has edges where G has no edges.
    An independent set in H is a clique in G.
    We want to partition V into k independent sets of H.
    
    Spectral approach:
    1. Build Laplacian of H
    2. Find k eigenvectors
    3. Cluster vertices by their positions in eigenvector space
    4. Each cluster should be an independent set in H (clique in G)
    
    Returns list of clusters (lists of vertices).
    """
    # Build adjacency matrix of complement graph
    A = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        mask = comp_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            A[u, v] = 1.0
            A[v, u] = 1.0
            mask ^= lsb
    
    # Degree matrix
    deg = A.sum(axis=1)
    D = np.diag(deg)
    
    # Laplacian L = D - A
    L = D - A
    
    # Add small regularization for numerical stability
    L = L + 1e-10 * np.eye(n)
    
    # Find k smallest eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        # Fallback: random partition
        clusters = [[] for _ in range(k)]
        for v in range(n):
            clusters[rng.randint(0, k - 1)].append(v)
        return clusters
    
    # Take k smallest eigenvectors
    U = eigvecs[:, :k]
    
    # Normalize rows
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = U / norms
    
    # K-means clustering
    # Initialize centers randomly
    indices = rng.sample(range(n), min(k, n))
    centers = U[indices].copy()
    if len(centers) < k:
        # Pad with random points
        while len(centers) < k:
            centers = np.vstack([centers, U[rng.randint(0, n - 1)]])
    
    assignments = np.zeros(n, dtype=int)
    
    for _ in range(50):  # K-means iterations
        # Assign each point to nearest center
        dists = np.sum((U[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        new_assignments = np.argmin(dists, axis=1)
        
        if np.all(new_assignments == assignments):
            break
        assignments = new_assignments
        
        # Update centers
        for c in range(k):
            mask = assignments == c
            if np.sum(mask) > 0:
                centers[c] = U[mask].mean(axis=0)
            else:
                centers[c] = U[rng.randint(0, n - 1)]
    
    # Build clusters
    clusters = [[] for _ in range(k)]
    for v in range(n):
        clusters[assignments[v]].append(v)
    
    return clusters


def _refine_clusters(clusters, comp_adj, n, rng):
    """
    Refine clusters to ensure each is an independent set in the complement graph
    (i.e., a clique in the original graph).
    
    For each cluster, remove vertices that have complement edges to other vertices
    in the cluster. Move them to other clusters or create new ones.
    """
    refined = []
    for cluster in clusters:
        if not cluster:
            continue
        
        # Check if cluster is an independent set in complement graph
        valid_cluster = []
        for v in cluster:
            # Check if v has no complement edges to any vertex in valid_cluster
            compatible = True
            for w in valid_cluster:
                if comp_adj[v] & (1 << w):
                    compatible = False
                    break
            if compatible:
                valid_cluster.append(v)
        
        if valid_cluster:
            refined.append(valid_cluster)
        
        # Vertices that couldn't be kept need to be placed elsewhere
        # For now, leave them out; they'll be handled by the greedy phase
    
    return refined


def _greedy_complete_partition(cliques, comp_adj, edges, n, rng):
    """
    Given a partial partition (list of cliques), complete it by greedily
    covering remaining edges.
    
    Returns complete list of cliques.
    """
    # Build set of edges already covered
    covered_edges = set()
    for clique in cliques:
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                if u > v:
                    u, v = v, u
                covered_edges.add((u, v))
    
    # Build remaining edges
    remaining = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) not in covered_edges:
            remaining.add((u, v))
    
    if not remaining:
        return cliques
    
    # Build adjacency for remaining edges
    rem_adj = [0] * n
    for u, v in remaining:
        rem_adj[u] |= (1 << v)
        rem_adj[v] |= (1 << u)
    
    result = list(cliques)
    
    while remaining:
        if time.time() > _DEADLINE - 3:
            # Fallback: add remaining edges as 2-cliques
            for u, v in remaining:
                result.append([u, v])
            break
        
        # Find edge with fewest common neighbors in remaining graph
        best_u, best_v = -1, -1
        best_common = float('inf')
        
        # Sample for efficiency
        edge_list = list(remaining)
        if len(edge_list) > 200:
            sample = rng.sample(edge_list, 200)
        else:
            sample = edge_list
        
        for u, v in sample:
            common = rem_adj[u] & rem_adj[v]
            count = bin(common).count('1')
            if count < best_common:
                best_common = count
                best_u, best_v = u, v
        
        if best_u == -1:
            break
        
        # Build maximal clique from (best_u, best_v) in remaining graph
        C = [best_u, best_v]
        cand_mask = rem_adj[best_u] & rem_adj[best_v]
        cand_mask &= ~(1 << best_u)
        cand_mask &= ~(1 << best_v)
        
        while cand_mask:
            # Find vertex with max degree in cand_mask
            best_c = -1
            max_deg = -1
            temp = cand_mask
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                d = bin(rem_adj[c] & cand_mask).count('1')
                d += rng.random() * 0.01
                if d > max_deg:
                    max_deg = d
                    best_c = c
                temp ^= lsb
            
            if best_c == -1:
                break
            
            C.append(best_c)
            cand_mask &= rem_adj[best_c]
            cand_mask &= ~(1 << best_c)
        
        if len(C) >= 2:
            result.append(C)
            # Remove edges within C from remaining
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, w = C[i], C[j]
                    if u > w:
                        u, w = w, u
                    remaining.discard((u, w))
                    rem_adj[u] &= ~(1 << w)
                    rem_adj[w] &= ~(1 << u)
        else:
            result.append([best_u, best_v])
            remaining.discard((best_u, best_v))
            rem_adj[best_u] &= ~(1 << best_v)
            rem_adj[best_v] &= ~(1 << best_u)
    
    return result


def _validate_partition(cliques, edges):
    """Validate that cliques form a valid partition of edges."""
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    seen_edges = set()
    for c in cliques:
        k = len(c)
        if k < 2:
            return False
        for i in range(k):
            for j in range(i + 1, k):
                u, v = c[i], c[j]
                if u > v:
                    u, v = v, u
                e = (u, v)
                if e not in edge_set or e in seen_edges:
                    return False
                seen_edges.add(e)
    return len(seen_edges) == len(edges)


def _solve_instance(edges, n, time_budget):
    """
    Solve a single instance using spectral clustering + greedy completion.
    """
    if not edges:
        return []
    
    comp_adj = _build_complement_adj(edges, n)
    
    best_cliques = None
    best_count = float('inf')
    
    # Try different numbers of initial clusters
    for k in [5, 8, 12, 16, 20, 30]:
        if time.time() > _DEADLINE - 5:
            break
        
        rng = _random.Random(10000 + k * 13)
        
        # Spectral clustering to get initial independent sets in complement
        clusters = _spectral_clustering(comp_adj, n, k, rng)
        
        # Refine to ensure valid cliques
        valid_cliques = _refine_clusters(clusters, comp_adj, n, rng)
        
        # Complete the partition
        full_partition = _greedy_complete_partition(valid_cliques, comp_adj, edges, n, rng)
        
        if len(full_partition) < best_count:
            best_count = len(full_partition)
            best_cliques = full_partition
    
    # If spectral approach didn't work well, fall back to pure greedy
    if best_cliques is None or best_count > len(edges):
        rng = _random.Random(42)
        # Pure greedy fallback
        remaining = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            remaining.add((u, v))
        
        rem_adj = [0] * n
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        while remaining:
            if time.time() > _DEADLINE - 3:
                for u, v in remaining:
                    cliques.append([u, v])
                break
            
            best_u, best_v = -1, -1
            best_common = float('inf')
            
            edge_list = list(remaining)
            if len(edge_list) > 200:
                sample = rng.sample(edge_list, 200)
            else:
                sample = edge_list
            
            for u, v in sample:
                common = rem_adj[u] & rem_adj[v]
                count = bin(common).count('1')
                if count < best_common:
                    best_common = count
                    best_u, best_v = u, v
            
            if best_u == -1:
                break
            
            C = [best_u, best_v]
            cand_mask = rem_adj[best_u] & rem_adj[best_v]
            cand_mask &= ~(1 << best_u)
            cand_mask &= ~(1 << best_v)
            
            while cand_mask:
                best_c = -1
                max_deg = -1
                temp = cand_mask
                while temp:
                    lsb = temp & -temp
                    c = lsb.bit_length() - 1
                    d = bin(rem_adj[c] & cand_mask).count('1')
                    d += rng.random() * 0.01
                    if d > max_deg:
                        max_deg = d
                        best_c = c
                    temp ^= lsb
                
                if best_c == -1:
                    break
                
                C.append(best_c)
                cand_mask &= rem_adj[best_c]
                cand_mask &= ~(1 << best_c)
            
            if len(C) >= 2:
                cliques.append(C)
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, w = C[i], C[j]
                        if u > w:
                            u, w = w, u
                        remaining.discard((u, w))
                        rem_adj[u] &= ~(1 << w)
                        rem_adj[w] &= ~(1 << u)
            else:
                cliques.append([best_u, best_v])
                remaining.discard((best_u, best_v))
                rem_adj[best_u] &= ~(1 << best_v)
                rem_adj[best_v] &= ~(1 << best_u)
        
        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques
    
    return best_cliques if best_cliques is not None else [[u, v] for u, v in edges]


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)
    
    # Allocate time: 60% to first, 40% to second
    time1 = 0.6 * 110.0
    time2 = 0.4 * 110.0
    
    # Solve smaller instance first
    res1 = _solve_instance(edges1, 32, time1)
    
    # Solve larger instance with remaining time
    res2 = _solve_instance(edges2, 64, time2)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
