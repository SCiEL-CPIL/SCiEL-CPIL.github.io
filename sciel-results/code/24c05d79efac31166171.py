"""Combined program: Attempts LPS construction for k=4, g=10 to beat the LUW baseline.
Falls back to LUW if LPS fails validity checks or for other parameters."""
# EVOLVE-BLOCK-START
import time
import sys
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # For the specific case k=4, g=10, try LPS constructions with small primes
    if k == 4 and g == 10:
        # Try q=7 first (n=336)
        n, edges = _try_lps(7)
        if n is not None and _check_girth(n, edges, g, deadline):
            return n, edges
        
        # Try q=11 (n=1320)
        n, edges = _try_lps(11)
        if n is not None and _check_girth(n, edges, g, deadline):
            return n, edges
            
        # Try q=13 (n=2184) - just in case
        n, edges = _try_lps(13)
        if n is not None and _check_girth(n, edges, g, deadline):
            return n, edges

    # Fallback: LUW construction for general k, g
    return _luw_construct(k, g)

def _try_lps(q):
    """Construct the LPS graph R_q. Returns (n, edges) or (None, None) on failure."""
    # Generate all matrices in SL(2, q)
    # Matrix is (a, b, c, d) with ad - bc = 1 mod q
    matrices = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # ad - bc = 1 => ad = 1 + bc => d = (1 + bc) * a^{-1} if a != 0
                # If a == 0, -bc = 1 => bc = -1 => c != 0, b != 0
                if a == 0:
                    # -bc = 1 mod q => bc = q-1
                    # b cannot be 0. c = (q-1) * b^{-1}
                    for b in range(1, q):
                        # Find inverse of b mod q
                        # Since q is prime, use pow(b, q-2, q)
                        inv_b = pow(b, q-2, q)
                        c = (q-1) * inv_b % q
                        d = 0
                        matrices.append((a, b, c, d))
                else:
                    inv_a = pow(a, q-2, q)
                    for b in range(q):
                        for c in range(q):
                            # ad = 1 + bc => d = (1 + bc) * inv_a
                            d = (1 + b * c) * inv_a % q
                            matrices.append((a, b, c, d))
    
    n = len(matrices)
    if n == 0:
        return None, None
        
    # Map matrix tuple to index
    mat_to_idx = {m: i for i, m in enumerate(matrices)}
    
    # Define generators
    # alpha = [[1, 1], [0, 1]]
    # beta  = [[1, 0], [1, 1]]
    # gamma = [[0, 1], [-1, 0]] -> (0, 1, q-1, 0)
    # delta = [[1, 0], [0, -1]] -> (1, 0, 0, q-1)
    
    gen_alpha = (1, 1, 0, 1)
    gen_beta  = (1, 0, 1, 1)
    gen_gamma = (0, 1, q-1, 0)
    gen_delta = (1, 0, 0, q-1)
    
    generators = [gen_alpha, gen_beta, gen_gamma, gen_delta]
    
    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return (
            (a1*a2 + b1*c2) % q,
            (a1*b2 + b1*d2) % q,
            (c1*a2 + d1*c2) % q,
            (c1*b2 + d1*d2) % q
        )
        
    edges = []
    seen_edges = set()
    
    for i, m in enumerate(matrices):
        for gen in generators:
            m2 = mat_mul(m, gen)
            if m2 in mat_to_idx:
                j = mat_to_idx[m2]
                if i != j: # No loops
                    edge = (i, j) if i < j else (j, i)
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        edges.append(edge)
                        
    return n, edges

def _check_girth(n, edges, g, deadline):
    """Check if graph has girth at least g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
            
    max_depth = (g - 1) // 2
    
    # Re-implement check with dist array for correctness
    for start_v in range(n):
        if time.time() > deadline:
            return False
        dist = [-1] * n
        parent = [-1] * n
        dist[start_v] = 0
        queue = deque([start_v])
        
        while queue:
            v = queue.popleft()
            if dist[v] == max_depth:
                continue
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                else:
                    # Found a visited node w
                    if w != parent[v]:
                        # Cycle length is dist[v] + dist[w] + 1
                        # This is a valid cycle if the paths are disjoint except for start?
                        # Not necessarily disjoint. But it is a cycle in the graph.
                        # If dist[v] + dist[w] + 1 < g, then girth < g.
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < g:
                            return False
                            
    return True

def _luw_construct(k, g):
    """LUW construction fallback."""
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    if not d_candidates:
        d = max(3, g - 5)
        if d % 2 == 0:
            d += 1
    else:
        d = min(d_candidates)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]
    
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
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
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
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
