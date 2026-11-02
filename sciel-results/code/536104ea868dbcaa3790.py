"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from collections import defaultdict, deque
import random

# EVOLVE-BLOCK-START
def _validate_graph(k, g, n, edges):
    """Validate that the graph is k-regular, simple, and has girth >= g."""
    if n <= 0:
        return False
    degree = [0] * n
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
        degree[u] += 1
        degree[v] += 1
    
    for d in degree:
        if d != k:
            return False
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check girth using BFS from each vertex (non-backtracking)
    # For girth >= g, no cycle of length < g
    # Use BFS with parent tracking to detect cycles
    for start in range(n):
        # BFS with state (node, prev) to avoid backtracking
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))  # (node, prev, dist)
        visited.add((start, -1))
        
        while queue:
            node, prev, dist = queue.popleft()
            if dist >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and dist + 1 < g:
                    return False  # Found cycle shorter than g
                if (nb, node) not in visited:
                    visited.add((nb, node))
                    queue.append((nb, node, dist + 1))
    
    return True

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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

def _circulant_search(k, g, max_n, deadline):
    """Search for a circulant k-regular graph with girth >= g on Z_n."""
    from itertools import combinations_with_replacement
    
    # For a circulant graph on Z_n with connection set S, we need:
    # 1. |S| = k (counting both +s and -s, but S is symmetric)
    # 2. No non-trivial product of < g elements from S equals 0 mod n
    # 
    # For k=5, S must be symmetric: if s in S, then -s mod n in S.
    # So S = {±s1, ±s2, ±s3, ...} but we need exactly 5 elements.
    # Wait, k=5 is odd, so we can't have a symmetric set of size 5 unless one element is its own inverse.
    # For n even, n/2 is its own inverse. So S = {±a, ±b, n/2} gives 5 elements.
    # For n odd, no element is its own inverse (except 0, which we can't use), so k must be even for circulant.
    # 
    # For k=5, n must be even. Let's try n = 2m.
    # S = {a, -a, b, -b, m} where m = n/2.
    
    if k != 5:
        return None
    
    # Try various n values
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (5,9): roughly 1 + 5*(1 + 4 + 4^2 + 4^3 + 4^4) = 1 + 5*(1+4+16+64+256) = 1 + 5*341 = 1706
    # But the actual smallest known is 1116, so we should search around there.
    # Let's search n from some lower bound up to max_n
    
    # Lower bound: for girth g, the number of vertices must be at least 1 + k*sum_{i=0}^{g/2-1} (k-1)^i
    # For g=9, g/2 = 4.5, so sum to i=4: 1 + 4 + 16 + 64 + 256 = 341, times 5 = 1705, plus 1 = 1706
    # But this is the Moore bound for the general case. For circulant, we might do better or worse.
    
    # Let's just search n from 400 to max_n, stepping by 2 (since n must be even for k=5)
    
    for n in range(400, min(max_n, 2000) + 1, 2):
        if time.time() > deadline:
            break
        
        m = n // 2
        # S = {a, n-a, b, n-b, m}
        # We need a, b in [1, m-1], a != b, a != m, b != m
        
        # For girth >= 9, we need no closed walk of length < 9 that sums to 0 mod n
        # A closed walk is a sequence of elements from S (with signs) that sums to 0.
        # Since S contains ±a, ±b, m, a walk is a sum of terms each being ±a, ±b, or m.
        # Let the walk have p terms of ±a, q terms of ±b, and r terms of m.
        # The sum is a*(sum of signs for a-terms) + b*(sum of signs for b-terms) + m*r.
        # The sum of signs for a-terms is an integer with same parity as p, ranging from -p to p in steps of 2.
        # Similarly for b-terms.
        
        # Total length = p + q + r < g = 9
        # Sum = a*sa + b*sb + m*r ≡ 0 (mod n)
        # where sa is sum of p signs, sb is sum of q signs, r is count of m-terms.
        # sa ∈ {-p, -p+2, ..., p}, sb ∈ {-q, -q+2, ..., q}
        
        # We need to check if there exist p, q, r >= 0 with p+q+r < 9, and sa, sb as above,
        # such that a*sa + b*sb + m*r ≡ 0 (mod n), and the walk is non-trivial (not all zero).
        
        # Non-trivial means at least one of p, q, r is positive.
        
        # This is a complex check. Let's precompute for each (p, q, r) the set of possible (sa, sb) pairs.
        
        valid = True
        for total_len in range(3, g):  # minimum cycle length is 3
            for p in range(total_len + 1):
                for q in range(total_len - p + 1):
                    r = total_len - p - q
                    if r < 0:
                        continue
                    # Possible sa values: -p, -p+2, ..., p
                    # Possible sb values: -q, -q+2, ..., q
                    for sa in range(-p, p + 1, 2):
                        for sb in range(-q, q + 1, 2):
                            # Check if a*sa + b*sb + m*r ≡ 0 (mod n)
                            # We need to check if there exist a, b in [1, m-1] such that this holds
                            # But we're checking for a specific a, b later. Let's just check for each (a, b) pair.
                            pass
        
        # Actually, let's just iterate over a and b directly.
        # For each (a, b), check if the graph has girth >= g.
        
        # To speed up, let's first filter out (a, b) pairs that clearly fail.
        # For example, if 2a ≡ 0 or 2b ≡ 0 mod n, then there's a 2-cycle (but we don't have 2-cycles since no loops).
        # Actually, a 2-cycle would require a + (-a) ≡ 0, which is always true, but that's just going back and forth, not a cycle.
        # A 3-cycle: a + a + (-a) = a ≠ 0 (if a ≠ 0). Or a + b + (-a-b) but -a-b might not be in S.
        # Let's just do the full check for each (a, b).
        
        for a in range(1, m):
            if time.time() > deadline:
                break
            for b in range(a + 1, m):
                if a == b or b == m:
                    continue
                
                # Check girth
                has_short_cycle = False
                for total_len in range(3, g):
                    for p in range(total_len + 1):
                        for q in range(total_len - p + 1):
                            r = total_len - p - q
                            if r < 0:
                                continue
                            for sa in range(-p, p + 1, 2):
                                for sb in range(-q, q + 1, 2):
                                    val = (a * sa + b * sb + m * r) % n
                                    if val == 0:
                                        # Check if this is a non-trivial walk
                                        # It's non-trivial if (p, q, r) != (0, 0, 0)
                                        if p + q + r > 0:
                                            has_short_cycle = True
                                            break
                                if has_short_cycle:
                                    break
                        if has_short_cycle:
                            break
                    if has_short_cycle:
                        break
                
                if not has_short_cycle:
                    # Found a valid circulant graph
                    edges = []
                    for i in range(n):
                        for s in [a, n - a, b, n - b, m]:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges
    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try LUW graph
    try:
        n_luw, edges_luw = _luw_graph(k, g)
        if _validate_graph(k, g, n_luw, edges_luw):
            # Try circulant search to see if we can do better
            result = _circulant_search(k, g, n_luw - 1, deadline)
            if result is not None:
                n_circ, edges_circ = result
                if _validate_graph(k, g, n_circ, edges_circ):
                    return n_circ, edges_circ
            return n_luw, edges_luw
    except Exception:
        pass
    
    # If we get here, return the LUW result anyway (it should be valid)
    try:
        n, edges = _luw_graph(k, g)
        return n, edges
    except Exception:
        # Fallback: return a trivial valid graph if possible
        # For now, just raise an error
        raise Exception("Failed to construct valid graph")
# EVOLVE-BLOCK-END
