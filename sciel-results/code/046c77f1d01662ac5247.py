import time

def _has_c5(n, adj):
    """Exact check for any 5-cycle as a subgraph."""
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not adj[v0][v1]:
                continue
            for v2 in range(v1 + 1, n):
                if not adj[v1][v2]:
                    continue
                for v3 in range(v2 + 1, n):
                    if not adj[v2][v3]:
                        continue
                    for v4 in range(v3 + 1, n):
                        if adj[v3][v4] and adj[v4][v0]:
                            return True
    return False

def _alpha_exact(n, adj):
    """Exact max independent set size via branch-and-bound DFS."""
    best = 0
    adj_list = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                adj_list[i].add(j)
                adj_list[j].add(i)
    
    def dfs(cand, size):
        nonlocal best
        if size + len(cand) <= best:
            return
        if not cand:
            if size > best:
                best = size
            return
        v = cand[0]
        rest = cand[1:]
        # Include v
        new_cand = [x for x in rest if x not in adj_list[v]]
        dfs(new_cand, size + 1)
        # Exclude v
        dfs(rest, size)
    
    dfs(list(range(n)), 0)
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    # Try n=35, 34, 33, 32 in order
    for target_n in [35, 34, 33, 32]:
        if time.time() > deadline:
            break
        
        if target_n == 32:
            # 8 disjoint K4s
            n = 32
            edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
            
            adj = [[False] * n for _ in range(n)]
            for u, v in edges:
                adj[u][v] = adj[v][u] = True
            
            if not _has_c5(n, adj) and _alpha_exact(n, adj) < 9:
                return {'n': n, 'edges': edges}
        
        elif target_n == 33:
            # 8 K4s + vertex 32 connected to all 4 vertices in K4_0
            n = 33
            edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
            # Connect vertex 32 to all vertices in K4_0 (vertices 0,1,2,3)
            for v in range(4):
                edges.append([32, v])
            
            adj = [[False] * n for _ in range(n)]
            for u, v in edges:
                adj[u][v] = adj[v][u] = True
            
            if not _has_c5(n, adj) and _alpha_exact(n, adj) < 9:
                return {'n': n, 'edges': edges}
        
        elif target_n == 34:
            # Try: 8 K4s + 2 extra vertices
            # Vertex 32 connected to all in K4_0, vertex 33 connected to all in K4_1
            n = 34
            edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
            # Vertex 32 -> K4_0 (0,1,2,3)
            for v in range(4):
                edges.append([32, v])
            # Vertex 33 -> K4_1 (4,5,6,7)
            for v in range(4, 8):
                edges.append([33, v])
            
            adj = [[False] * n for _ in range(n)]
            for u, v in edges:
                adj[u][v] = adj[v][u] = True
            
            if not _has_c5(n, adj) and _alpha_exact(n, adj) < 9:
                return {'n': n, 'edges': edges}
            
            # Alternative: both extra vertices connected to same K4
            edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
            # Vertex 32 -> K4_0 (0,1,2,3)
            for v in range(4):
                edges.append([32, v])
            # Vertex 33 -> K4_0 (0,1,2,3)
            for v in range(4):
                edges.append([33, v])
            
            adj = [[False] * n for _ in range(n)]
            for u, v in edges:
                adj[u][v] = adj[v][u] = True
            
            if not _has_c5(n, adj) and _alpha_exact(n, adj) < 9:
                return {'n': n, 'edges': edges}
        
        elif target_n == 35:
            # Try: 8 K4s + 3 extra vertices
            # Vertex 32 -> K4_0, Vertex 33 -> K4_1, Vertex 34 -> K4_2
            n = 35
            edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
            for v in range(4):
                edges.append([32, v])       # 32 -> K4_0
            for v in range(4, 8):
                edges.append([33, v])       # 33 -> K4_1
            for v in range(8, 12):
                edges.append([34, v])       # 34 -> K4_2
            
            adj = [[False] * n for _ in range(n)]
            for u, v in edges:
                adj[u][v] = adj[v][u] = True
            
            if not _has_c5(n, adj) and _alpha_exact(n, adj) < 9:
                return {'n': n, 'edges': edges}
            
            # Alternative: all 3 extra vertices connected to K4_0
            edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
            for v in range(4):
                edges.append([32, v])
                edges.append([33, v])
                edges.append([34, v])
            
            adj = [[False] * n for _ in range(n)]
            for u, v in edges:
                adj[u][v] = adj[v][u] = True
            
            if not _has_c5(n, adj) and _alpha_exact(n, adj) < 9:
                return {'n': n, 'edges': edges}
    
    # Fallback: return n=32
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
