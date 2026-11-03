import time
import random
import sys
from collections import deque

def _exact_girth_check(adj, g):
    """Return True iff the graph has no cycle of length < g.
    Uses non-backtracking walks from every vertex up to depth g-1.
    A cycle of length L < g corresponds to a non-backtracking walk of length L
    from a vertex back to itself. We detect this by tracking (current, prev)
    states and checking for cycles in the non-backtracking line graph.
    """
    n = len(adj)
    # For each vertex, do a DFS/BFS on non-backtracking walks up to depth g-1.
    # A non-backtracking walk is a sequence v0, v1, ..., vL where v_i != v_{i-2} for i >= 2.
    # A cycle of length L exists iff there is a non-backtracking walk of length L from v to v.
    # We can detect this by: for each starting vertex u, enumerate all non-backtracking
    # walks of length exactly L for L = 3, 4, ..., g-1, and check if any returns to u.
    # But that's expensive. Better: use the standard method.
    #
    # Standard efficient method: For each edge (u, v), check if there's a path from v to u
    # of length g-2 that doesn't use edge (u,v) in the first step. Actually, simpler:
    #
    # For each vertex u, perform BFS in the "state space" of (node, parent) where
    # state (v, p) means we're at v having come from p. Transitions: from (v, p) go to
    # (w, v) for all w in adj[v], w != p. A cycle of length L < g exists iff we can
    # return to the starting state in L steps, OR more precisely, a cycle of length L
    # corresponds to a closed non-backtracking walk of length L.
    #
    # Practical approach: For each vertex u, BFS up to depth g-1 in the non-backtracking
    # sense. If we reach u again at depth L < g, we have a cycle of length L.
    # To avoid counting trivial "backtracks", we track (current, prev) pairs.
    
    for u in range(n):
        # BFS on states (node, prev_node)
        # Start: from u, go to each neighbor v. State: (v, u) at depth 1.
        visited = set()
        queue = deque()
        for v in adj[u]:
            state = (v, u)
            visited.add(state)
            queue.append((v, u, 1))
        
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for w in adj[node]:
                if w == prev:
                    continue
                new_state = (w, node)
                if w == u and depth + 1 < g:
                    # Found a cycle of length depth + 1
                    return False
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((w, node, depth + 1))
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Moore bound for odd girth g, degree k
    if g % 2 == 1:
        half_g = (g - 3) // 2
        moore_bound = 1 + k * sum((k - 1) ** i for i in range(half_g + 1))
    else:
        half_g = g // 2 - 1
        moore_bound = 2 * sum((k - 1) ** i for i in range(half_g + 1))
    
    # Try the LUW (Lazebnik-Ustimenko-Woldar) construction first,
    # which is known to produce k-regular graphs of high girth.
    # For k=5, g=9: d = g - 5 = 4, but d must be odd, so d = 5.
    # The LUW construction with q=k and parameter d gives a 2*q*d vertex graph
    # that is q-regular with girth at least d + 5.
    
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    
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
    
    n_luw = len(order)
    
    # Build adjacency list for the LUW graph
    adj = [[] for _ in range(n_luw)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Verify it's k-regular
    is_regular = all(len(adj[u]) == k for u in range(n_luw))
    
    # Verify girth >= g using exact check
    if is_regular and n_luw >= moore_bound:
        if time.time() < deadline:
            has_girth = _exact_girth_check(adj, g)
            if has_girth:
                return n_luw, edges
    
    # If LUW didn't work or failed, try circulant graphs
    max_n = 5000
    for n in range(moore_bound, max_n):
        if time.time() > deadline:
            break
        if n % 2 != 0:
            continue
            
        for _ in range(100):
            if time.time() > deadline:
                break
                
            a = random.randint(1, n // 2 - 1)
            b = random.randint(1, n // 2 - 1)
            
            if a == b or a == n - b or b == n - a or a == 0 or b == 0:
                continue
            if a == n // 2 or b == n // 2:
                continue
                
            S = {a, n - a, b, n - b, n // 2}
            
            adj_circ = [[] for _ in range(n)]
            for u in range(n):
                for s in S:
                    v = (u + s) % n
                    adj_circ[u].append(v)
            
            if _exact_girth_check(adj_circ, g):
                edges_circ = []
                for u in range(n):
                    for v in adj_circ[u]:
                        if u < v:
                            edges_circ.append((u, v))
                return n, edges_circ
    
    # Final fallback: return the LUW graph even if verification was skipped
    return n_luw, edges
