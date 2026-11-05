import random
import time
from collections import deque

def luw_construct(k, g):
    # Logic from seed
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Get fallback
    n_fallback, edges_fallback = luw_construct(k, g)
    
    # If k or g are such that L-U-W is small, maybe it's good?
    # But for 3,16 it's huge.
    
    # Try to find a better graph
    # Moore bound for k=3, g=16 is 766.
    # Known best is 936.
    # We will try to construct a graph with N=936.
    
    target_n = 936
    
    # If k != 3 or g != 16, this specific target might be wrong.
    # The task says "evaluator passes this task's degree k and girth g".
    # It implies k=3, g=16.
    # But to be safe, if k!=3 or g!=16, just return fallback.
    if k != 3 or g != 16:
        return n_fallback, edges_fallback
        
    # Random search for N=936
    # We will try a few restarts.
    
    def try_construct(N, k, g, deadline):
        # N vertices, k-regular, girth >= g
        # Returns edges list or None
        
        # State
        adj = [set() for _ in range(N)]
        deg = [0] * N
        edges = []
        
        # For backtracking, we store the edges added.
        # We use a simple recursive backtracking or iterative with stack.
        # Given the depth (1500 edges), recursion limit might be an issue.
        # Use iterative DFS or just restarts.
        # Restarts are simpler.
        
        # We will do a greedy random construction with limited backtracking.
        # Actually, just greedy with random restarts is often enough for these sparse graphs.
        
        # To check cycle:
        def has_short_path(u, v, max_depth):
            # BFS from u to v
            if u == v: return True
            visited = {u: 0}
            q = deque([u])
            while q:
                curr = q.popleft()
                dist = visited[curr]
                if dist >= max_depth: continue
                for nei in adj[curr]:
                    if nei == v:
                        return True
                    if nei not in visited:
                        visited[nei] = dist + 1
                        q.append(nei)
            return False
        
        # Greedy loop
        remaining_edges = (N * k) // 2
        retries = 0
        max_retries = 100
        
        while remaining_edges > 0:
            if time.time() > deadline:
                return None
            
            # Pick random u, v with deg < k
            candidates_u = [i for i in range(N) if deg[i] < k]
            if not candidates_u: break
            
            u = random.choice(candidates_u)
            candidates_v = [i for i in range(N) if deg[i] < k and i != u and i not in adj[u]]
            if not candidates_v:
                # Dead end
                return None
            
            v = random.choice(candidates_v)
            
            # Check if adding (u,v) creates a cycle < g
            # This happens if there is a path of length 1 to g-2 between u and v
            # Path length 1 is checked by i not in adj[u]
            # So we check path length 2 to g-2
            # BFS from u, avoid v at start? No, just check if v is reachable in < g-1 steps.
            # If v is reachable in L steps, cycle is L+1.
            # We want L+1 >= g => L >= g-1.
            # So we must ensure v is NOT reachable in < g-1 steps.
            # So check if v is reachable in 1 to g-2 steps.
            
            if has_short_path(u, v, g - 2):
                retries += 1
                if retries > max_retries:
                    return None
                continue
            
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edges.append((u, v))
            remaining_edges -= 1
            retries = 0
            
        if remaining_edges == 0:
            return edges
        return None

    # Try to construct
    for attempt in range(20):
        if time.time() > deadline: break
        result = try_construct(target_n, k, g, deadline)
        if result:
            return target_n, result
            
    # If failed, try slightly larger N?
    # Or just return fallback.
    # Given 936 is the cage, it might be hard to find with random greedy.
    # But it's worth a try.
    
    return n_fallback, edges_fallback
