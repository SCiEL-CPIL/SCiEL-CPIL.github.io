import time
import itertools
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if k == 4 and g == 10:
        # Attempt to find a valid 320-vertex graph
        n, edges = _find_4_10_cage(deadline)
        if n is not None:
            return n, edges
        
        # Fallback to LUW (will be large, but valid)
        return _luw_construct(k, g)

    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
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

def _check_girth(n, edges, required_girth):
    """
    Check if the graph has girth at least required_girth.
    Uses BFS from each vertex to find the shortest cycle.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Sort neighbors for determinism? Not needed for check.
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    # But this finds the shortest cycle passing through u-v edge?
                    # Actually, BFS from start finds shortest path to all.
                    # A cycle is formed if there is an edge (u,v) where 
                    # dist[u] + 1 != dist[v] and dist[v] + 1 != dist[u]
                    # The length of the cycle is dist[u] + dist[v] + 1?
                    # No, that's for non-backtracking.
                    
                    # Simpler: If we find an edge (u,v) such that dist[u] + 1 > dist[v] 
                    # and u is not parent of v, then we have a cycle.
                    # The length is dist[u] + dist[v] + 1.
                    # Wait, this logic is for finding *a* cycle.
                    # To find the *shortest* cycle, we need to be careful.
                    
                    # Actually, the standard BFS girth check is:
                    # For each edge (u,v), if dist[u] + 1 > dist[v] and parent[u] != v,
                    # then cycle length is dist[u] + dist[v] + 1.
                    # We take the minimum over all such edges.
                    
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < required_girth:
                        return False
        # If no cycle found from start, continue to next start?
        # No, if the graph is connected, one BFS might not find all cycles?
        # Actually, BFS from a single node finds the shortest cycle *containing that node*.
        # To find the global shortest cycle, we must check from all nodes?
        # Or just check all edges?
        # Yes, checking all edges with dist from a single source is not enough.
        # We need to run BFS from each node? That's O(n * E).
        # For n=320, E=640, this is fine.
        
    return True

def _find_4_10_cage(deadline):
    n = 320
    # Search for circulant graph C_320(S)
    # S is a set of 4 steps.
    
    # Heuristic: Try steps that are "far apart".
    # We'll use a randomized search with a time limit.
    
    rng = random.Random(42)
    
    # Pre-calculate all_steps for a given S
    def _girth_ok(n, S, required_girth=10):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for s in S:
            for i in range(n):
                j = (i + s) % n
                adj[i].append(j)
        
        # Check girth
        # To speed up, we can check for cycles of length 4, 6, 8 explicitly first.
        # If any exist, return False.
        
        # Cycle of length 4: i -> i+s1 -> i+s1+s2 -> i+s1+s2+s3 -> i
        # s1+s2+s3+s4 = 0
        # We can just use the BFS method, it's robust.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True

    # Try random sets S
    while time.time() < deadline - 5:
        S = rng.sample(range(1, n // 2 + 1), 4)
        # Ensure no 2-cycles: 2s != 0 mod n
        if any((2 * s) % n == 0 for s in S):
            continue
        # Ensure distinct steps
        if len(set(S)) < 4:
            continue
            
        if _girth_ok(n, S, 10):
            edges = []
            for i in range(n):
                for s in S:
                    j = (i + s) % n
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    # If no circulant found, return None
    return None, None
