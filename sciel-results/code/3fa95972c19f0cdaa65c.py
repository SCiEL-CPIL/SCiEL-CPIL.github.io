import time
from itertools import combinations

def _is_q3(vertices, edges_set):
    """Fast check if 8 vertices induce a Q3."""
    if len(vertices) != 8:
        return False
    # Count edges
    ec = 0
    for i in range(8):
        for j in range(i+1, 8):
            u, v = vertices[i], vertices[j]
            if (u,v) in edges_set:
                ec += 1
    if ec != 12:
        return False
    # Check 3-regular
    for v in vertices:
        deg = 0
        for u in vertices:
            if u != v and (v,u) in edges_set:
                deg += 1
        if deg != 3:
            return False
    # Check bipartite 4+4
    color = {}
    def bfs_check():
        for start in vertices:
            if start in color:
                continue
            color[start] = 0
            stack = [start]
            while stack:
                x = stack.pop()
                for y in vertices:
                    if y != x and (x,y) in edges_set:
                        if y in color:
                            if color[y] == color[x]:
                                return False
                        else:
                            color[y] = 1 - color[x]
                            stack.append(y)
        return True
    if not bfs_check():
        return False
    c0 = sum(1 for v in vertices if color[v] == 0)
    return c0 == 4

def _add_edge_creates_q3_fast(u, v, adj, edges_set):
    """Check if adding (u,v) creates a Q3. Q3 must contain edge (u,v)."""
    # In Q3, u and v are adjacent. Each has 2 other neighbors in Q3.
    # u's other Q3-neighbors: choose 2 from adj[u] (excluding v)
    # v's other Q3-neighbors: choose 2 from adj[v] (excluding u)
    # The 8 vertices are {u, v} + U_other + V_other where |U_other|=2, |V_other|=2... 
    # No wait, Q3 has 8 vertices. u and v are 2. We need 6 more.
    # In Q3, the vertex opposite to u (distance 3) and opposite to v (distance 3) are specific.
    # Actually, let's think of Q3 structure: vertices are 3-bit strings.
    # If u=000, v=001 (adjacent), then Q3 vertices are all 8 3-bit strings.
    # u's neighbors: 001, 010, 100. v's neighbors: 000, 011, 101.
    # So u has 2 other neighbors (010, 100), v has 2 other neighbors (011, 101).
    # The remaining 2 vertices are 110 and 111.
    # 
    # General approach: enumerate all 8-sets containing u and v. C(14,6)=3003.
    # For each, check if it's Q3. This is the straightforward approach.
    # To speed up: precompute which 8-sets could be Q3.
    
    # Faster: the 6 other vertices must include:
    # - 2 neighbors of u (from adj[u] \ {v})
    # - 2 neighbors of v (from adj[v] \ {u})
    # - 2 more vertices
    # But the "2 more" are not necessarily non-neighbors.
    
    # Let's just do the C(14,6) enumeration but with early termination.
    other = [x for x in range(16) if x != u and x != v]
    for combo in combinations(other, 6):
        vertices = (u, v) + combo
        # Quick edge count check
        ec = 1  # edge (u,v) being added
        for i in range(8):
            for j in range(i+1, 8):
                a, b = vertices[i], vertices[j]
                if a == u and b == v:
                    continue
                if a == v and b == u:
                    continue
                if (a,b) in edges_set:
                    ec += 1
        if ec != 12:
            continue
        # Check 3-regular
        ok = True
        for v2 in vertices:
            deg = 0
            for u2 in vertices:
                if u2 == v2:
                    continue
                if u2 == u and v2 == v:
                    deg += 1
                    continue
                if u2 == v and v2 == u:
                    deg += 1
                    continue
                if (v2,u2) in edges_set:
                    deg += 1
            if deg != 3:
                ok = False
                break
        if not ok:
            continue
        # Check bipartite
        color = {}
        def bfs():
            for start in vertices:
                if start in color:
                    continue
                color[start] = 0
                stack = [start]
                while stack:
                    x = stack.pop()
                    for y in vertices:
                        if y == x:
                            continue
                        is_edge = (x==u and y==v) or (x==v and y==u) or ((x,y) in edges_set)
                        if is_edge:
                            if y in color:
                                if color[y] == color[x]:
                                    return False
                            else:
                                color[y] = 1 - color[x]
                                stack.append(y)
            return True
        if not bfs():
            continue
        c0 = sum(1 for x in vertices if color[x] == 0)
        if c0 == 4:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    edges = [[a, b] for a in range(3) for b in range(3, 16)]
    adj = [set() for _ in range(16)]
    edges_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edges_set.add((u, v))
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Candidate edges: within {0,1,2} and within {3..15}
    candidates = []
    for i in range(16):
        for j in range(i+1, 16):
            if (i, j) not in edges_set:
                candidates.append((i, j))
    
    # Try adding edges greedily
    for u, v in candidates:
        if time.time() > deadline:
            break
        if not _add_edge_creates_q3_fast(u, v, adj, edges_set):
            edges.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
            edges_set.add((u, v))
            best_edges = list(edges)
            best_count = len(edges)
    
    return best_edges

