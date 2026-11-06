"""Optimized graph construction for minimizing max triangle degree sum."""
import time
import random

def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _deg_from_adj(adj, n):
    return [bin(adj[u]).count('1') for u in range(n)]

def _all_triangles(adj, n):
    tris = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                continue
            common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
            c = common
            while c:
                low = c & (-c)
                c &= c - 1
                w = low.bit_length() - 1
                tris.append((u, v, w))
    return tris

def _max_tri_sum(adj, deg, tris):
    if not tris:
        return 0
    mx = 0
    for u, v, w in tris:
        s = deg[u] + deg[v] + deg[w]
        if s > mx:
            mx = s
    return mx

def _edges_from_adj(adj, n):
    es = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                es.append([u, v])
    return es

def _add_edge(adj, deg, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    deg[u] += 1
    deg[v] += 1

def _remove_edge(adj, deg, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    deg[u] -= 1
    deg[v] -= 1

def _initial_graphs(n=16, target=65):
    """Generate several diverse initial graphs."""
    graphs = []
    
    # Graph 1: K_{8,8} + one intra edge
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    edges.append([0, 1])
    graphs.append(edges)
    
    # Graph 2: K_{8,8} + one intra edge, different vertex
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    edges.append([3, 5])
    graphs.append(edges)
    
    # Graph 3: Near-regular graph with balanced degrees
    # Try to create a graph where degrees are as equal as possible
    # 65 edges * 2 = 130 degree sum, 130/16 = 8.125
    # So 8 vertices have degree 8, 8 have degree 9 (or similar)
    # Start with a random near-regular graph
    for seed in [42, 123, 789]:
        rng = random.Random(seed)
        edges_set = set()
        # Generate random graph with 65 edges
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        for u, v in all_pairs[:65]:
            edges_set.add((u, v))
        edges = [[u, v] for u, v in edges_set]
        graphs.append(edges)
    
    # Graph 4: Structured - two clusters with some cross edges
    # 8 vertices in each cluster, 60 cross edges + 5 intra edges
    edges = []
    # 60 cross edges (out of 64 possible)
    cross_pairs = [(u, v) for u in range(8) for v in range(8, 16)]
    rng = random.Random(999)
    rng.shuffle(cross_pairs)
    for u, v in cross_pairs[:60]:
        edges.append([u, v])
    # 5 intra edges
    intra1 = [(0, 1), (2, 3), (4, 5), (6, 7)]
    intra2 = [(8, 9), (10, 11), (12, 13), (14, 15)]
    for u, v in intra1[:2]:
        edges.append([u, v])
    for u, v in intra2[:3]:
        edges.append([u, v])
    graphs.append(edges)
    
    # Graph 5: More structured
    edges = []
    # Complete bipartite minus some edges, plus intra edges
    cross_pairs = [(u, v) for u in range(8) for v in range(8, 16)]
    rng = random.Random(555)
    rng.shuffle(cross_pairs)
    for u, v in cross_pairs[:62]:
        edges.append([u, v])
    # 3 intra edges
    edges.append([0, 2])
    edges.append([1, 3])
    edges.append([8, 10])
    graphs.append(edges)
    
    return graphs

def _local_search(adj, deg, n, target, deadline, max_iter=50000):
    """Local search: try edge swaps to reduce max triangle degree sum."""
    tris = _all_triangles(adj, n)
    best_max = _max_tri_sum(adj, deg, tris)
    best_adj = list(adj)
    best_deg = list(deg)
    
    start = time.time()
    no_improve_count = 0
    max_no_improve = 2000
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
        if no_improve_count > max_no_improve:
            break
        
        if not tris:
            break
        
        # Find worst triangle
        worst_tri = None
        worst_sum = -1
        for u, v, w in tris:
            s = deg[u] + deg[v] + deg[w]
            if s > worst_sum:
                worst_sum = s
                worst_tri = (u, v, w)
        
        if worst_tri is None:
            break
        
        u, v, w = worst_tri
        tri_edges = [(u, v), (u, w), (v, w)]
        random.shuffle(tri_edges)
        
        improved = False
        for rem_u, rem_v in tri_edges:
            if not (adj[rem_u] & (1 << rem_v)):
                continue
            
            _remove_edge(adj, deg, rem_u, rem_v)
            
            # Find good non-edges to add
            non_edges = []
            for a in range(n):
                for b in range(a + 1, n):
                    if not (adj[a] & (1 << b)):
                        # Estimate impact
                        common = adj[a] & adj[b]
                        est = 0
                        c = common
                        while c:
                            low = c & (-c)
                            c &= c - 1
                            ci = low.bit_length() - 1
                            s = deg[a] + deg[b] + deg[ci]
                            if s > est:
                                est = s
                        non_edges.append((est, a, b))
            
            if not non_edges:
                _add_edge(adj, deg, rem_u, rem_v)
                continue
            
            non_edges.sort()
            
            for est, add_u, add_v in non_edges[:15]:
                if time.time() > deadline - 3:
                    break
                _add_edge(adj, deg, add_u, add_v)
                new_tris = _all_triangles(adj, n)
                new_max = _max_tri_sum(adj, deg, new_tris)
                
                if new_max < best_max:
                    best_max = new_max
                    best_adj = list(adj)
                    best_deg = list(deg)
                    tris = new_tris
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    _remove_edge(adj, deg, add_u, add_v)
            
            if improved:
                break
            
            _add_edge(adj, deg, rem_u, rem_v)
        
        if not improved:
            no_improve_count += 1
            # Random perturbation
            if random.random() < 0.5 and tris:
                u, v, w = random.choice(tris)
                rem_u, rem_v = random.choice([(u, v), (u, w), (v, w)])
                if adj[rem_u] & (1 << rem_v):
                    _remove_edge(adj, deg, rem_u, rem_v)
                    non_edges = []
                    for a in range(n):
                        for b in range(a + 1, n):
                            if not (adj[a] & (1 << b)):
                                non_edges.append((a, b))
                    if non_edges:
                        add_u, add_v = random.choice(non_edges)
                        _add_edge(adj, deg, add_u, add_v)
                        tris = _all_triangles(adj, n)
                        new_max = _max_tri_sum(adj, deg, tris)
                        if new_max < best_max:
                            best_max = new_max
                            best_adj = list(adj)
                            best_deg = list(deg)
                            no_improve_count = 0
    
    return best_adj, best_deg, best_max

def construct():
    n = 16
    target = 65
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    initial_graphs = _initial_graphs(n, target)
    
    best_adj = None
    best_deg = None
    best_score = float('inf')
    
    for i, edges in enumerate(initial_graphs):
        if time.time() > deadline - 20:
            break
        
        adj = _build_adj(n, edges)
        deg = _deg_from_adj(adj, n)
        
        # Verify edge count
        edge_count = sum(deg) // 2
        if edge_count != target:
            continue
        
        adj, deg, score = _local_search(adj, deg, n, target, deadline)
        
        if score < best_score:
            best_score = score
            best_adj = list(adj)
            best_deg = list(deg)
    
    # If we didn't find anything, use the first graph
    if best_adj is None:
        adj = _build_adj(n, initial_graphs[0])
        best_adj = adj
        best_deg = _deg_from_adj(adj, n)
    
    edges = _edges_from_adj(best_adj, n)
    
    # Final verification
    assert len(edges) == target, f"Expected {target} edges, got {len(edges)}"
    assert all(0 <= u < n and 0 <= v < n and u != v for u, v in edges)
    
    return {'n': n, 'edges': edges}
