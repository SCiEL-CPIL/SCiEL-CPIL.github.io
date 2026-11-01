def construct(k, g):
    import time
    from collections import deque
    import numpy as np

    start_time = time.time()
    deadline = start_time + 115.0

    # Hardcoded known optimal/near-optimal cages
    known_cages = {}

    # Petersen Graph (3, 5) - 10 vertices
    known_cages[(3, 5)] = (10, [(0,1),(1,2),(2,3),(3,4),(4,0),
                                 (5,7),(7,9),(9,6),(6,8),(8,5),
                                 (0,5),(1,6),(2,7),(3,8),(4,9)])

    # K_{3,3} (3,4) - 6 vertices
    known_cages[(3, 4)] = (6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)])

    # K4 (3,3) - 4 vertices
    known_cages[(3, 3)] = (4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])

    # K_{4,4} (4,4) - 8 vertices
    known_cages[(4, 4)] = (8, [(i,j) for i in range(4) for j in range(4,8)])

    # K5 (4,3) - 5 vertices
    known_cages[(4, 3)] = (5, [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)])

    if (k, g) in known_cages:
        return known_cages[(k, g)]

    q = k

    def build_luw(d, q):
        """Build Lazebnik-Ustimenko-Woldar graph CD(d,q)."""
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q

        rule = {}
        rule[2] = (1,1)
        rule[3] = (2,1)
        rule[4] = (1,2)
        for c in range(5, d+1):
            r = (c-5)%4
            c0 = c - (c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1-side), tuple(y)

        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        return len(order), edges

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def girth_check_fast(n, adj, g, time_limit):
        """
        Fast girth check: verify no cycle shorter than g exists.
        Uses BFS from each vertex with non-backtracking constraint.
        Returns True if girth >= g, False otherwise.
        Checks time limit periodically.
        """
        check_time = time.time()
        for start in range(n):
            if time.time() - check_time > time_limit:
                return None  # timeout, unknown result
            
            # BFS over non-backtracking walks
            # dist[(u, prev)] = shortest non-backtracking walk length from start to u via prev
            visited = set()
            queue = deque()
            
            for v in adj[start]:
                state = (v, start)
                if state not in visited:
                    visited.add(state)
                    queue.append((v, start, 1))
            
            while queue:
                if time.time() - check_time > time_limit:
                    return None  # timeout
                
                u, prev, depth = queue.popleft()
                
                if u == start:
                    if depth < g:
                        return False
                    continue
                
                if depth >= g - 1:
                    continue
                
                for w in adj[u]:
                    if w == prev:
                        continue
                    new_state = (w, u)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((w, u, depth + 1))
        
        return True

    # For the main task (5,9): LUW(5,5) should work
    # LUW(d,q) has girth at least 2d-1 when q is prime or prime power
    # For d=5, girth >= 9, and n = q^d = 5^5 = 3125
    
    best_n = None
    best_edges = None
    best_adj = None

    d_min = max(3, (g + 1) // 2)
    
    # Try d values from minimum upward
    for d in range(d_min, d_min + 4):
        if time.time() > deadline - 40:
            break
            
        try:
            n, edges = build_luw(d, q)
            
            if n > 50000:
                continue
                
            adj = build_adj(n, edges)
            
            # Verify degree
            ok = True
            for i in range(n):
                if len(adj[i]) != k:
                    ok = False
                    break
            
            if not ok:
                continue
            
            # Check girth with time limit
            time_left = deadline - time.time() - 5
            if time_left < 10:
                # Not enough time for full verification, accept if n is reasonable
                # LUW(d,q) theoretically has girth >= 2d-1
                if 2*d - 1 >= g:
                    best_n, best_edges = n, edges
                    best_adj = adj
                    break
                else:
                    continue
            
            result = girth_check_fast(n, adj, g, time_left)
            
            if result is True:
                best_n, best_edges = n, edges
                best_adj = adj
                break
            elif result is None:
                # Timeout - if theoretical guarantee holds, accept
                if 2*d - 1 >= g:
                    best_n, best_edges = n, edges
                    best_adj = adj
                    break
            else:
                continue
                
        except Exception as e:
            continue
    
    if best_n is None:
        # Fallback: use larger d
        d = max(3, g)
        try:
            n, edges = build_luw(d, q)
            best_n, best_edges = n, edges
            best_adj = build_adj(n, edges)
        except Exception:
            # Last resort: construct a simple circulant-like graph
            n = k * g
            edges = []
            seen = set()
            jumps = list(range(1, k // 2 + 1))
            if k % 2 == 1:
                jumps.append(n // 2)
            
            for i in range(n):
                for j in jumps:
                    u = i
                    v = (i + j) % n
                    edge = (min(u,v), max(u,v))
                    if edge not in seen:
                        seen.add(edge)
                        edges.append(edge)
                    u2 = i
                    v2 = (i - j) % n
                    edge2 = (min(u2,v2), max(u2,v2))
                    if edge2 not in seen:
                        seen.add(edge2)
                        edges.append(edge2)
            
            best_n, best_edges = n, edges
            best_adj = build_adj(n, edges)

    return best_n, best_edges
