"""Construct small cages: 5-regular girth-9 via Cyclic Search + LW fallback."""
import time
import random
import numpy as np
from collections import deque

_START_TIME = time.time()
_TIMEOUT = 145  # wall-clock budget (seconds)

def _deadline_exceeded():
    return (time.time() - _START_TIME) > _TIMEOUT

def _verify_girth_bfs(edges, n, g):
    """Check no cycle shorter than g using BFS from every vertex (or just one if Cayley).
    For general graph, we should check all. But for cyclic, one is enough.
    We will use this for the final verification of the returned graph."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        prev = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2: # Heuristic pruning, but we need exact
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    queue.append(v)
                elif v != prev[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _check_simple_and_regular(edges, n, k):
    if len(edges) != n * k // 2:
        return False
    seen = set()
    deg = [0] * n
    for u, v in edges:
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    return True

def _cyclic_girth_check(n, a, b, g):
    """Check girth of cyclic graph with S = {n/2, a, -a, b, -b}."""
    S = [n // 2, a, (-a) % n, b, (-b) % n]
    # Remove duplicates if any (e.g. if a == b or a == 0)
    S = list(set(S))
    if len(S) != 5:
        return False
    
    # BFS from 0 to find shortest cycle
    # State: (current, previous) to avoid backtracking?
    # Actually, for girth, we can just do standard BFS and check for edges connecting nodes at dist d1, d2.
    # But we need to ensure the cycle is simple.
    # A standard BFS from 0 finds the shortest path to each node.
    # If there is an edge (u, v) such that dist[u] + dist[v] + 1 < g, is it a cycle?
    # Yes, if the paths from 0 to u and 0 to v are disjoint except at 0.
    # In a tree (BFS tree), they are disjoint.
    # So if we find such an edge, we have a cycle of length dist[u] + dist[v] + 1.
    
    dist = [-1] * n
    dist[0] = 0
    queue = deque([0])
    
    while queue:
        u = queue.popleft()
        if dist[u] >= g:
            continue
        for s in S:
            v = (u + s) % n
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
            else:
                # Check if this edge creates a short cycle
                # The cycle length is dist[u] + dist[v] + 1
                # But we must ensure u and v are not "backtracking" in a way that creates a 2-cycle (impossible in simple graph)
                # Actually, this logic finds *a* cycle.
                # Is it the shortest?
                # The shortest cycle through 0 is found this way?
                # Not necessarily. The shortest cycle might not pass through 0.
                # However, for a Cayley graph, the girth is the length of the shortest word summing to 0.
                # This is equivalent to the shortest cycle passing through 0.
                # So checking cycles through 0 is sufficient for Cayley graphs.
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < g:
                    return False
    return True

def construct_cyclic(k, g):
    """Try to find a cyclic k-regular graph of girth g."""
    if k != 5 or g != 9:
        return None
    
    # n must be even for k=5 (odd degree) in cyclic construction
    # S = {n/2, a, -a, b, -b}
    
    # Range of n to search
    # Moore bound 428. Best known 1116.
    # Let's search 1000 to 1500.
    for n in range(1000, 1501, 2):
        if _deadline_exceeded():
            return None
        
        # Try random pairs (a, b)
        # a, b in [1, n/2 - 1]
        # Ensure a != b, a != -b, etc.
        
        for _ in range(200): # 200 restarts per n
            if _deadline_exceeded():
                return None
            
            a = random.randint(1, n // 2 - 1)
            b = random.randint(1, n // 2 - 1)
            
            # Quick checks for duplicates in S
            # S = {n/2, a, n-a, b, n-b}
            # Duplicates if a==b, a==n-b (a+b=n), b==n-a (same), a==n/2 (handled by range), etc.
            if a == b:
                continue
            if (a + b) % n == 0:
                continue
            if a == n // 2 or b == n // 2:
                continue
            
            if _cyclic_girth_check(n, a, b, g):
                # Construct edges
                S = [n // 2, a, (-a) % n, b, (-b) % n]
                edges = []
                for i in range(n):
                    for s in S:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
    return None

def construct_lw(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d,q) construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    """Main entry point: construct k-regular graph with girth >= g, minimize n."""
    t0 = time.time()
    
    # Try cyclic search first
    result = construct_cyclic(k, g)
    if result:
        n, edges = result
        # Verify
        if _check_simple_and_regular(edges, n, k) and _verify_girth_bfs(edges, n, g):
            return n, edges
    
    # Fallback to LW
    n, edges = construct_lw(k, g)
    
    return n, edges
