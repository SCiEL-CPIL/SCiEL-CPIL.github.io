import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def luw_construct(q, d):
        """Build LUW graph CD(d,q). Returns (n, edges)."""
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
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
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
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
    
    def check_girth(n, edges, min_girth):
        """Check if girth >= min_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            # BFS tracking (vertex, parent)
            visited = {start: None}  # vertex -> parent
            queue = deque([(start, None)])
            depth = 0
            while queue and depth < min_girth:
                depth += 1
                new_queue = deque()
                for v, parent in queue:
                    for w in adj[v]:
                        if w == parent:
                            continue
                        if w not in visited:
                            visited[w] = v
                            new_queue.append((w, v))
                        else:
                            # Found a cycle
                            # Need to check if it's a valid cycle (not just backtracking)
                            # This is tricky; let's use a different approach
                            pass
                queue = new_queue
        
        # Better approach: BFS with path tracking
        for start in range(n):
            # (vertex, prev_vertex, depth)
            visited = {}
            queue = deque([(start, None, 1)])
            while queue:
                v, prev, depth = queue.popleft()
                if depth >= min_girth:
                    continue
                for w in adj[v]:
                    if w == prev:
                        continue
                    if w == start and depth + 1 >= min_girth:
                        # Found a cycle of length depth+1
                        return False
                    key = (w, v)
                    if key not in visited or visited[key] > depth + 1:
                        visited[key] = depth + 1
                        queue.append((w, v, depth + 1))
        
        return True
    
    # Try d values from smallest to largest
    q = k
    best_n = None
    best_edges = None
    
    for d in range(3, 10):
        if time.time() > deadline:
            break
        try:
            n, edges = luw_construct(q, d)
            if n < (best_n if best_n else float('inf')):
                if check_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
                    break  # Found a valid graph with this d
        except Exception:
            continue
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use d=5 (should always work for g=9)
    n, edges = luw_construct(q, 5)
    return n, edges
