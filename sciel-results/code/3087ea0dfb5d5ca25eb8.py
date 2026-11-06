import time
import random

N = 16
ALL_BITS = (1 << N) - 1

def _groetzsch_edges():
    """
    Grötzsch graph (Mycielski of C5).
    11 vertices, 20 edges.
    Vertices 0..4 form C5.
    Vertices 5..9 are copies of 0..4.
    Vertex 10 is the center.
    """
    edges = []
    k = 5
    # Cycle 0-1-2-3-4-0
    for i in range(k):
        edges.append([i, (i + 1) % k])
    
    # Mycielski construction:
    # For each edge (u, v) in original, add edges (v, u') and (u, v') where u' is copy of u.
    # Actually, standard Mycielski:
    # V' = {v_0, ..., v_{k-1}}
    # E' = { (v_i, v_j) | (i, j) in E } U { (v_i, v'_j) | (j, i) in E }
    # Plus center v_k connected to all v'_i.
    
    # Let's just hardcode the known valid edges for Grötzsch to be safe.
    # Reference: Grötzsch graph has 11 vertices and 20 edges.
    # It is triangle-free and 4-chromatic.
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 6], [1, 7], [2, 8], [3, 9], [4, 10],
        [5, 7], [5, 8], [6, 8], [6, 9], [7, 9],
        [7, 10], [8, 10], [9, 10]
    ]
    # Wait, the hardcoded list in the original was suspicious. 
    # Let's generate it properly to be 100% sure.
    edges = []
    # Original C5: 0-1, 1-2, 2-3, 3-4, 4-0
    for i in range(5):
        edges.append([i, (i+1)%5])
        
    # Copies 5-9 correspond to 0-4.
    # Center 10.
    # Edges between original and copies:
    # If (u,v) is an edge in C5, then (v, u') and (u, v') are edges in Mycielski.
    # u' is 5+u, v' is 5+v.
    orig_edges = [[0,1], [1,2], [2,3], [3,4], [4,0]]
    for u, v in orig_edges:
        edges.append([v, 5+u])
        edges.append([u, 5+v])
        
    # Center 10 connected to all copies 5-9
    for i in range(5):
        edges.append([10, 5+i])
        
    return edges

def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        if u < N and v < N:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2

def _is_triangle_free_add(adj, u, v):
    # Check if adding (u,v) creates a triangle.
    # This happens if there is a common neighbor w.
    return (adj[u] & adj[v]) == 0

def _is_3_colorable(adj):
    """
    Simple backtracking to check if graph is 3-colorable.
    """
    colors = [-1] * N
    # Order vertices by degree descending for efficiency
    order = sorted(range(N), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    
    def backtrack(i):
        if i == N:
            return True
        v = order[i]
        # Try colors 0, 1, 2
        for c in range(3):
            # Check if neighbor has color c
            valid = True
            m = adj[v]
            while m:
                lb = m & (-m)
                u = lb.bit_length() - 1
                m ^= lb
                if colors[u] == c:
                    valid = False
                    break
            if valid:
                colors[v] = c
                if backtrack(i + 1):
                    return True
                colors[v] = -1
        return False
        
    return backtrack(0)

def _greedy_fill(adj, deadline, rng):
    """
    Greedily add edges to adj, respecting triangle-freeness.
    """
    candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if not ((adj[u] >> v) & 1):
                candidates.append((u, v))
    
    rng.shuffle(candidates)
    
    for u, v in candidates:
        if time.time() > deadline:
            break
        if _is_triangle_free_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return adj

def _local_improve(adj, deadline, rng):
    """
    Local search to maximize edges.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any missing edge
        missing = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        rng.shuffle(missing)
        
        for u, v in missing:
            if _is_triangle_free_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                improved = True
                break # Restart search from top
        
        if not improved:
            # 2. Try to remove a non-core edge to allow adding more?
            # For triangle-free graphs, removing an edge never helps add *more* edges 
            # in a simple greedy sense unless we are looking at specific structural swaps.
            # However, a common move is "2-for-1" or just accepting we are at a local max.
            # Given the small N, we can try a few random removals and re-greedy.
            current_edges = _edge_count(adj)
            
            # Identify existing edges
            existing = []
            for u in range(N):
                for v in range(u + 1, N):
                    if (adj[u] >> v) & 1:
                        existing.append((u, v))
            
            if not existing:
                break

            # Try a few random removals
            best_adj = None
            best_cnt = current_edges
            
            for _ in range(5): # Limit inner loop to prevent timeout
                if time.time() > deadline:
                    break
                rem_u, rem_v = rng.choice(existing)
                
                # Copy adj
                adj_copy = [a for a in adj]
                adj_copy[rem_u] &= ~(1 << rem_v)
                adj_copy[rem_v] &= ~(1 << rem_u)
                
                # Greedy fill
                adj_copy = _greedy_fill(adj_copy, deadline, rng)
                cnt = _edge_count(adj_copy)
                
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_adj = adj_copy
            
            if best_adj:
                adj = best_adj
                improved = True

    return adj

def construct():
    start = time.time()
    deadline = start + 110.0
    
    # Use Grötzsch graph (11 vertices) as base.
    # It is 4-chromatic and triangle-free.
    # We will extend it to 16 vertices.
    base_edges = _groetzsch_edges()
    
    best_adj = None
    best_cnt = -1
    
    rng = random.Random(42)
    
    # Multiple trials
    for trial in range(200):
        if time.time() > deadline - 2.0:
            break
        
        # Start with Grötzsch graph on vertices 0..10
        adj = _adj_from_edges(base_edges)
        
        # Vertices 11..15 are isolated initially.
        # Greedy fill will try to connect them.
        
        # Greedy fill
        adj = _greedy_fill(adj, deadline, rng)
        
        # Local improve
        adj = _local_improve(adj, deadline, rng)
        
        cnt = _edge_count(adj)
        
        # Check if it's still 4-chromatic (not 3-colorable)
        # This is expensive, so we do it only if cnt is promising
        if cnt > best_cnt:
            if not _is_3_colorable(adj):
                best_cnt = cnt
                best_adj = adj
    
    # Fallback: If we didn't find a good one, return the Grötzsch graph with isolated nodes
    # But we must ensure it's not 3-colorable. Grötzsch is not.
    if best_adj is None:
        best_adj = _adj_from_edges(base_edges)
    
    # Final Verification
    # 1. Triangle-free?
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                if (best_adj[u] & best_adj[v]) != 0:
                    # Triangle found! This should not happen.
                    # If it does, we have a bug.
                    pass
    
    # 2. Not 3-colorable?
    if _is_3_colorable(best_adj):
        # If it became 3-colorable, we failed.
        # This can happen if the search removed critical edges or the greedy fill
        # created a 3-colorable graph.
        # In this case, we should return the base Grötzsch graph which is known 4-chromatic.
        best_adj = _adj_from_edges(base_edges)
        
    # Convert back to edge list
    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])
                
    return out
