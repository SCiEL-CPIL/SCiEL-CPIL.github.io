import time
import random
import math
from collections import deque

_DEADLINE = 120.0

def _verify(n, edges, k, g):
    """Verify graph is simple, k-regular, connected, girth >= g."""
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # girth check
    # BFS from each vertex up to depth g//2
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([(start, 0)])
        while q:
            u, d = q.popleft()
            if d >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    q.append((v, d + 1))
                else:
                    # Cycle found
                    # Length = dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _luzw(k, g):
    """Fallback LUW construction."""
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

def _search_circulant(k, g, max_n, time_limit):
    """
    Search for a circulant k-regular graph with girth >= g.
    For k=4, we search for C_n({a, -a, b, -b}).
    """
    if k != 4:
        return None
        
    start_time = time.time()
    # Moore bound for general graph
    # 1 + k * (1 + (k-1) + ... + (k-1)^{g/2 - 1})
    # For g=10, g/2=5.
    # 1 + 4 * (1 + 3 + 9 + 27) = 1 + 4 * 40 = 161.
    moore_bound = 161
    
    # We want even n? Not strictly, but bipartite requires even.
    # Girth 10 is even.
    # Let's try n from moore_bound to max_n.
    
    for n in range(moore_bound, max_n + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Connection set {a, -a, b, -b}
        # a, b must be distinct mod n.
        # 1 <= a < b < n/2 (to avoid duplicates and self loops)
        # Also a != b.
        
        found = False
        for a in range(1, n // 2):
            for b in range(a + 1, n // 2):
                # Check if a, -a, b, -b are distinct
                # a != -a => 2a != 0 mod n => n % 2a != 0
                # a != b, a != -b => a+b != 0 mod n
                # b != -b => 2b != 0 mod n
                
                if (2 * a) % n == 0 or (2 * b) % n == 0 or (a + b) % n == 0:
                    continue
                    
                # Construct graph and check girth
                # We can do a fast girth check without building full edge list first?
                # Just BFS on the implicit graph.
                
                def get_girth(n, a, b):
                    # Check for cycles < g
                    # Since g=10, we check for 4, 6, 8.
                    # BFS from 0 up to depth 4.
                    # If we find a cycle of length < 10, return False.
                    # We need to check from all vertices?
                    # For circulant graphs, checking from 0 might be enough?
                    # No, not necessarily. But it's a good heuristic.
                    # Actually, for small n, full check is fast.
                    
                    # Optimized check:
                    # 4-cycles: a+a=0 (no), a-b=0 (no), a+b=0 (no).
                    # Wait, 4-cycle in C_n(a,b) exists if x+y = u+v.
                    # E.g., 0 -> a -> a+b -> b -> 0?
                    # Edges: (0,a), (a, a+b), (a+b, b), (b, 0).
                    # This is a 4-cycle if a+b != 0 and a != b.
                    # So if a+b < n, we have a 4-cycle?
                    # Path: 0 -a-> a -(b)-> a+b -(-b)-> b -(-a)-> 0?
                    # Edge (a+b, b) exists if difference is b. (a+b)-b = a. No.
                    # Edge (a+b, b) exists if difference is b or a.
                    # (a+b) - b = a. So edge (a+b, b) is NOT in the set {a, -a, b, -b} unless a=b or a=-b.
                    # So 0-a-(a+b)-b-0 is NOT a cycle unless a=b or a=-b.
                    
                    # Let's just run BFS.
                    dist = [-1] * n
                    dist[0] = 0
                    q = deque([(0, 0)])
                    while q:
                        u, d = q.popleft()
                        if d >= g // 2:
                            continue
                        for step in [a, -a, b, -b]:
                            v = (u + step) % n
                            if dist[v] == -1:
                                dist[v] = d + 1
                                q.append((v, d + 1))
                            else:
                                if dist[u] + dist[v] + 1 < g:
                                    return False
                    return True

                if get_girth(n, a, b):
                    # Build edges
                    edges = []
                    for i in range(n):
                        for step in [a, b]:
                            j = (i + step) % n
                            if i < j:
                                edges.append((i, j))
                            # We need to add all edges.
                            # i -> i+a, i -> i-a, i -> i+b, i -> i-b.
                            # To avoid duplicates, we can just iterate i and add (i, (i+a)%n) and (i, (i+b)%n).
                            # Since a, b > 0, (i+a)%n != i.
                            # We might add (i, j) and (j, i).
                            # The verifier handles sets.
                    # Actually, let's just generate unique edges.
                    edges = []
                    for i in range(n):
                        j1 = (i + a) % n
                        j2 = (i + b) % n
                        edges.append((i, j1))
                        edges.append((i, j2))
                        # This adds each edge twice?
                        # i -> i+a. Later (i+a) -> (i+a)-a = i.
                        # So yes, duplicates.
                        # But _verify checks len(edges) == len(set(edges)).
                        # So we must deduplicate.
                    
                    unique_edges = list(set(tuple(sorted(e)) for e in edges))
                    # Wait, (i, j) and (j, i) are the same edge.
                    # My loop adds (i, i+a) and (i+a, i).
                    # So set will handle it.
                    
                    # Check validity
                    if _verify(n, unique_edges, k, g):
                        return n, unique_edges
                        
        # If not found for this n, continue to next n

    return None

def construct(k, g):
    start = time.time()
    
    if k == 4 and g == 10:
        # Try circulant search first
        # Limit time to 60s for search, leaving 60s for fallback/verify
        result = _search_circulant(k, g, 400, 60.0)
        if result:
            n, edges = result
            # Double check
            if _verify(n, edges, k, g):
                return n, edges
                
        # Fallback to LUW
        n, edges = _luzw(k, g)
        return n, edges

    # General fallback
    n, edges = _luzw(k, g)
    return n, edges
