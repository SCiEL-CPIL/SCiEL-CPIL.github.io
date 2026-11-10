import random
import time
from collections import deque

def _luw_construct(k, g):
    """Fallback LUW construction"""
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

def _has_short_cycle(adj, n, max_len=9):
    """Check if graph has any cycle of length <= max_len.
    Uses BFS from each vertex up to depth max_len//2.
    """
    # For efficiency, we can just check if any vertex has a neighbor within distance < max_len
    # that is not its parent in the BFS tree, forming a cycle.
    # But simpler: just check if the graph contains any cycle of length L for L in 3..9.
    # Given n is small (~200), we can afford O(n * degree^depth) or similar.
    
    # Actually, a simpler check: 
    # If the graph is 4-regular, we can check for triangles and squares easily.
    # For longer cycles, we can use a randomized walk or full BFS.
    
    # Let's do BFS from each vertex up to depth 4.
    # If we encounter a vertex already visited in the current BFS path (cycle),
    # check the length.
    
    for start in range(n):
        visited = [-1] * n
        parent = [-1] * n
        depth = [0] * n
        q = deque([start])
        visited[start] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if visited[v] == -1:
                    visited[v] = 1
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    q.append(v)
                else:
                    # Cycle found
                    # Length is depth[u] + depth[v] + 1
                    # But this might not be the shortest cycle.
                    # For girth check, we just need to know if ANY short cycle exists.
                    # If we find a cycle of length < 10, return True.
                    cycle_len = depth[u] + depth[v] + 1
                    if cycle_len < 10:
                        return True
    return False

def _try_random_4regular(n, time_limit):
    """Try to find a 4-regular graph with girth >= 10 on n vertices."""
    start_time = time.time()
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random 4-regular graph
        # Use configuration model
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
        random.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        valid = True
        for i in range(0, n * 4, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            # Check for multi-edge
            if v in adj[u]:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
        
        if not valid:
            continue
        
        # Check girth
        if not _has_short_cycle(adj, n):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
            
        # If short cycle found, try to fix it (optional, might be too slow)
        # For now, just restart
    return None

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
        
    deadline = time.time() + 110 # Leave 10s margin
    
    # Try n from 162 upwards
    # Step by 2 to ensure even n
    for n in range(162, 400, 2):
        if time.time() > deadline:
            break
            
        # Allocate some time for this n
        # If we have lots of time, give more.
        time_alloc = 5.0
        res = _try_random_4regular(n, time_alloc)
        if res:
            return res
            
    # Fallback
    return _luw_construct(k, g)
