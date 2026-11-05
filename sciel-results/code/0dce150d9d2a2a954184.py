"""Deterministic greedy tree-folding search for the (4,10)-cage."""
import sys
import time
from collections import deque

# --- Finite-field helpers for GF(4) (char-2) and GF(k) (prime) ---
def _gf4_ops():
    # GF(4) = {0,1,2,3} with x^2 = x + 1; use representation: 0=0,1=1,2=alpha,3=alpha+1
    # addition is XOR; multiplication table below.
    add = lambda a, b: a ^ b
    table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
    ]
    mul = lambda a, b: table[a][b]
    return add, mul

def _gfp_ops(p):
    add = lambda a, b: (a + b) % p
    mul = lambda a, b: (a * b) % p
    return add, mul

# --- LUW / Lazebnik–Ustimenko–Woldar construction (fallback) ---
def _luw_construct(k, g):
    """Build the LUW graph CD(d, q) with q = k, choosing d so girth >= g.
    Returns (n, edges)."""
    q = k
    # For odd girth g, the LUW graph has girth at least d + 5 when d is odd.
    # We want d + 5 >= g  =>  d >= g - 5.  Choose smallest odd d >= g - 5.
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    if q == 4:
        add, mul = _gf4_ops()
        sub = add  # in char 2, sub == add
    else:
        add, mul = _gfp_ops(q)
        sub = lambda a, b: (a - b) % q

    # Rule for LUW: for c = 2..d, (a_c, b_c)
    # For c = 2: (1,1); c=3: (2,1); c=4: (1,2)
    # For c >= 5: periodic pattern with period 4
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Build graph via BFS from the zero point on side 0
    # A vertex is (side, x) where side in {0,1}, x is a tuple of length d+1
    # x[0] is unused (None), x[1..d] are field elements
    # Point ~ Line iff for all c in 2..d: line[c] - point[c] = line[a_c] * point[b_c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p, find line l: l[c] = p[c] + l[a]*p[b]  -- but l[a] is unknown
                    # Actually: l[c] - p[c] = l[a] * p[b]
                    # We need to solve for l given p. The LUW construction defines:
                    # Given point p, line l is determined by l[1] = first, and for c=2..d:
                    # l[c] = p[c] + l[a] * p[b]
                    # But l[a] may refer to an already-determined coordinate.
                    # We build l incrementally: l[1] is set, then for c=2..d compute l[c].
                    # But l[a] might not be computed yet if a >= c. Let's check the rule.
                    # For the standard LUW, a_c < c always holds.
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l, find point p: l[c] - p[c] = l[a] * p[b]
                    # p[c] = l[c] - l[a]*p[b]
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
    return len(order), edges

# --- New Approach: Cayley Graph Search ---
def _cayley_search(k, g, time_limit):
    """Search for a k-regular Cayley graph with girth >= g."""
    start_time = time.time()
    # Moore bound for k=4, g=10 is 243.
    # We search for groups of size n >= 243.
    # We try abelian groups Z_n, Z_n x Z_m, etc.
    
    for n in range(243, 400):
        if time.time() - start_time > time_limit:
            break
            
        # Try Z_n
        # Generators: 1, n-1, a, n-a
        for a in range(2, n//2):
            if a == 1 or a == n-1: continue
            if a == n-a: continue
            gens = [1, n-1, a, n-a]
            if len(set(gens)) != 4: continue
            
            edges = []
            for i in range(n):
                for g_val in gens:
                    j = (i + g_val) % n
                    if i < j:
                        edges.append((i, j))
            
            if _validate_graph(n, edges, k, g):
                return n, edges
                
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    start_time = time.time()
    
    # Try Cayley search
    result = _cayley_search(k, g, time_limit=5.0)
    
    if result is not None:
        n, edges = result
        if _validate_graph(n, edges, k, g):
            return n, edges
    
    # Fall back to LUW
    return _luw_construct(k, g)

def _validate_graph(n, edges, k, g):
    """Validate that the graph is k-regular, simple, and has girth >= g."""
    # Check simple: no loops, no repeated edges
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        e = (min(u, v), max(u, v))
        if e in edge_set:
            return False
        edge_set.add(e)
    
    # Check degree
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    if any(d != k for d in degree):
        return False
    
    # Check girth >= g using BFS from each vertex
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for src in range(n):
        # BFS from src
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif abs(dist[v] - dist[u]) <= 1 and dist[v] != dist[u] + 1:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
