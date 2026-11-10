import time
import random

_T0 = time.time()
_DEADLINE = _T0 + 110.0  # Safety margin: 10s before 120s limit

def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

def _build_adjacency(edges, n):
    """Build adjacency list as bitmasks for fast operations."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_clique(vertices, adj):
    """Check if a set of vertices forms a clique using bitmasks."""
    if len(vertices) < 2:
        return False
    mask = 0
    for v in vertices:
        mask |= (1 << v)
    for v in vertices:
        if (mask & ~(1 << v)) & ~adj[v]:
            return False
    return True

def _fast_greedy_partition(edges, n, adj, rng, time_limit):
    """
    Fast greedy clique partition using bitmasks for remaining edges.
    """
    if not edges:
        return []
    
    # Map edges to indices for bitmask representation
    # N=64, max edges ~2000. Python int is fine.
    edge_index = {}
    edge_list = []
    for i, (u, v) in enumerate(edges):
        u, v = min(u, v), max(u, v)
        edge_index[(u, v)] = i
        edge_list.append((u, v))
        
    num_edges = len(edges)
    remaining_mask = (1 << num_edges) - 1
    cliques = []
    
    # Degree in remaining graph
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        
    # Adjacency in remaining graph (bitmasks of neighbors)
    rem_adj = [0] * n
    for i, (u, v) in enumerate(edges):
        if (remaining_mask >> i) & 1:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
            
    while remaining_mask:
        if time.time() > time_limit:
            # Fallback: emit remaining edges as 2-cliques
            # We need to extract edges from mask
            m = remaining_mask
            while m:
                i = (m & -m).bit_length() - 1
                u, v = edge_list[i]
                cliques.append([u, v])
                m &= m - 1
            return cliques
            
        # Find active vertex with highest degree
        # deg[v] is the number of remaining edges incident to v
        active_vertices = [v for v in range(n) if deg[v] > 0]
        if not active_vertices:
            break
            
        # Pick a start vertex: top degree, with random tie-breaking
        max_deg = max(deg[v] for v in active_vertices)
        top_verts = [v for v in active_vertices if deg[v] == max_deg]
        start_v = rng.choice(top_verts)
        
        # Build clique
        C = [start_v]
        # Candidates: neighbors of start_v in remaining graph
        candidates_mask = rem_adj[start_v]
        
        # Greedy expansion
        while candidates_mask:
            # Pick candidate with max degree among candidates
            # candidates_mask is a bitmask of vertices
            # We need to find the vertex in candidates_mask with max deg
            # Iterate through bits
            best_u = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if deg[u] > best_deg:
                    best_deg = deg[u]
                    best_u = u
                m &= m - 1
                
            if best_u == -1:
                break
                
            C.append(best_u)
            
            # Update candidates: intersect with neighbors of best_u
            candidates_mask &= rem_adj[best_u]
            
        # Add clique
        if len(C) >= 2:
            cliques.append(C)
            # Remove edges within C from remaining_mask
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    u, v = min(u, v), max(u, v)
                    idx = edge_index.get((u, v))
                    if idx is not None:
                        remaining_mask &= ~(1 << idx)
                        deg[u] -= 1
                        deg[v] -= 1
                        # Update rem_adj? 
                        # rem_adj is only used to find candidates.
                        # Since we are removing edges, rem_adj should be updated to reflect this.
                        # However, updating rem_adj for every edge removal is O(1) per edge.
                        # rem_adj[u] &= ~(1 << v)
                        # rem_adj[v] &= ~(1 << u)
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
        else:
            # Fallback: take one edge
            # Find any edge in remaining_mask
            m = remaining_mask
            i = (m & -m).bit_length() - 1
            u, v = edge_list[i]
            cliques.append([u, v])
            remaining_mask &= ~(1 << i)
            deg[u] -= 1
            deg[v] -= 1
            rem_adj[u] &= ~(1 << v)
            rem_adj[v] &= ~(1 << u)

    return cliques

def _local_improve_fast(cliques, adj, n, time_limit):
    """
    Fast local improvement: merge cliques if union is a clique.
    """
    changed = True
    while changed and time.time() < time_limit:
        changed = False
        new_cliques = []
        used = [False] * len(cliques)
        
        for i in range(len(cliques)):
            if used[i]:
                continue
            Ci = cliques[i]
            merged = False
            for j in range(i + 1, len(cliques)):
                if used[j]:
                    continue
                Cj = cliques[j]
                # Check if Ci U Cj is a clique
                # Since Ci and Cj are cliques, we just need cross edges
                is_mergeable = True
                # Optimization: if any vertex in Cj is not connected to any in Ci, fail
                for u in Cj:
                    # Check if u is connected to all v in Ci
                    mask_ci = 0
                    for v in Ci:
                        mask_ci |= (1 << v)
                    if (mask_ci & ~adj[u]):
                        is_mergeable = False
                        break
                if is_mergeable:
                    new_cliques.append(Ci + Cj)
                    used[i] = True
                    used[j] = True
                    changed = True
                    merged = True
                    break
            if not merged:
                new_cliques.append(Ci)
        cliques = new_cliques
    return cliques

def _validate_partition(cliques, edges, adj):
    """
    Check if the partition is valid.
    """
    if not cliques:
        return len(edges) == 0
        
    seen_edges = set()
    for C in cliques:
        if len(C) < 2:
            return False
        if not _is_clique(C, adj):
            return False
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                u, v = C[i], C[j]
                e = (min(u, v), max(u, v))
                if e in seen_edges:
                    return False
                seen_edges.add(e)
                
    if len(seen_edges) != len(edges):
        return False
        
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    if seen_edges != edge_set:
        return False
        
    return True

def _solve_instance(edges, n, time_limit):
    if not edges:
        return []
        
    adj = _build_adjacency(edges, n)
    
    best_cliques = None
    best_count = float('inf')
    
    rng_seed = 42
    restart = 0
    
    while time.time() < time_limit:
        rng = random.Random(rng_seed + restart)
        
        cliques = _fast_greedy_partition(edges, n, adj, rng, time_limit)
        
        if not cliques:
            continue
            
        cliques = _local_improve_fast(cliques, adj, n, time_limit)
        
        if _validate_partition(cliques, edges, adj):
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques
                
        restart += 1
        
    if best_cliques is None:
        # Fallback
        rng = random.Random(0)
        best_cliques = _fast_greedy_partition(edges, n, adj, rng, time_limit)
        
    return best_cliques

def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    n1, p1, seed1 = 32, 0.6, 601
    edges1 = _random_graph(n1, p1, seed1)
    
    n2, p2, seed2 = 64, 0.55, 602
    edges2 = _random_graph(n2, p2, seed2)
    
    # Allocate time budget
    # Total limit 110s.
    # Give 30s to Graph 1, 80s to Graph 2.
    
    time_limit_1 = _T0 + 30.0
    res1 = _solve_instance(edges1, n1, time_limit_1)
    
    time_limit_2 = _T0 + 110.0
    res2 = _solve_instance(edges2, n2, time_limit_2)
    
    return [res1, res2]
