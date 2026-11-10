# EVOLVE-BLOCK-START
import time
import math
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Primary strategy: Search for 3-regular Cayley graphs on Z_n x Z_2.
    Fallback: Lazebnik-Ustimenko-Woldar (LUW) construction.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin against 120s limit

    if k != 3:
        # Fallback for other degrees
        return _lnw_construct(k, g)

    # Moore bound for k=3, girth g
    # n >= 2 * (1 + (k-1)^(g/2 - 1))
    if g % 2 == 0:
        moore_bound = 2 * (1 + 2 ** (g // 2 - 1))
    else:
        moore_bound = 2 * (1 + 2 ** ((g - 1) // 2))
        
    # For Cayley graphs on Z_n x Z_2, total vertices N = 2n.
    # So 2n >= moore_bound => n >= moore_bound // 2
    min_n = max(3, moore_bound // 2)
    
    best_n = float('inf')
    best_edges = None

    # Search for Cayley graphs
    # We iterate n from min_n upwards.
    # For each n, we try possible generators.
    # Generators: u=(a,1) with 2a=0 mod n, v=(b,0), w=(-b,0).
    
    # Limit the search space to keep within time
    # If n gets too large, BFS becomes expensive, but we break early if girth < g.
    # However, we should prioritize smaller n.
    
    max_n_to_try = min_n + 5000 # Try a reasonable range
    
    for n in range(min_n, max_n_to_try):
        if time.time() > deadline - 5:
            break
            
        # Determine valid 'a' values for involution u=(a,1)
        # 2a = 0 mod n
        if n % 2 == 0:
            a_candidates = [0, n // 2]
        else:
            a_candidates = [0]
            
        for a in a_candidates:
            # v=(b,0), w=(-b,0)
            # b must be such that v != e, w != e, v != w, u != v, u != w
            # v != e => b != 0
            # w != e => -b != 0 => b != 0
            # v != w => b != -b => 2b != 0 mod n
            # u != v => (a,1) != (b,0) => always true (different 2nd component)
            # u != w => (a,1) != (-b,0) => always true
            
            # So we just need b != 0 and 2b != 0 mod n.
            
            for b in range(1, n // 2 + 1):
                if (2 * b) % n == 0:
                    continue
                    
                # Check if we have time for BFS
                if time.time() > deadline - 2:
                    break
                    
                # Run BFS on the Cayley graph to check girth
                # Group size is 2n
                girth = _check_cayley_girth(n, a, b, g)
                
                if girth >= g:
                    # Found a valid graph!
                    edges = _build_cayley_edges(n, a, b)
                    N = 2 * n
                    return N, edges

    # If Cayley search fails or times out, fallback to LUW
    return _lnw_construct(3, g)

def _check_cayley_girth(n, a, b, target_girth):
    """
    Check the girth of the Cayley graph on Z_n x Z_2 with generators 
    u=(a,1), v=(b,0), w=(-b,0).
    Returns the girth.
    """
    N = 2 * n
    dist = [-1] * N
    parent = [-1] * N
    
    # Start at identity (0,0) -> index 0
    dist[0] = 0
    queue = deque([0])
    
    min_cycle = N + 1
    
    # Precompute neighbor deltas? 
    # Neighbors of (i, j):
    # 1. (i+a mod n, 1-j)
    # 2. (i+b mod n, j)
    # 3. (i-b mod n, j)
    
    while queue:
        u_idx = queue.popleft()
        d_u = dist[u_idx]
        
        if d_u >= target_girth:
            continue
            
        # Decode u_idx to (i, j)
        i = u_idx // 2
        j = u_idx % 2
        
        # Generate neighbors
        # 1. u = (a, 1)
        ni1 = (i + a) % n
        nj1 = 1 - j
        v1 = 2 * ni1 + nj1
        
        # 2. v = (b, 0)
        ni2 = (i + b) % n
        nj2 = j
        v2 = 2 * ni2 + nj2
        
        # 3. w = (-b, 0)
        ni3 = (i - b) % n
        nj3 = j
        v3 = 2 * ni3 + nj3
        
        for v in (v1, v2, v3):
            if dist[v] == -1:
                dist[v] = d_u + 1
                parent[v] = u_idx
                queue.append(v)
            else:
                # If v is not the parent, we found a cycle
                if v != parent[u_idx]:
                    cycle_len = d_u + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle < target_girth:
                            return min_cycle # Early exit: girth is too small
                
    return min_cycle if min_cycle <= N else N + 1

def _build_cayley_edges(n, a, b):
    """
    Build the edge list for the Cayley graph on Z_n x Z_2.
    """
    N = 2 * n
    edges = []
    seen = set()
    
    for i in range(n):
        for j in range(2):
            u_idx = 2 * i + j
            
            # Neighbor 1: (i+a mod n, 1-j)
            ni1 = (i + a) % n
            nj1 = 1 - j
            v1 = 2 * ni1 + nj1
            
            # Neighbor 2: (i+b mod n, j)
            ni2 = (i + b) % n
            nj2 = j
            v2 = 2 * ni2 + nj2
            
            # Neighbor 3: (i-b mod n, j)
            ni3 = (i - b) % n
            nj3 = j
            v3 = 2 * ni3 + nj3
            
            for v in (v1, v2, v3):
                if u_idx < v:
                    edges.append((u_idx, v))
                    seen.add((u_idx, v))
                elif u_idx > v:
                    # Already added if we iterate all nodes
                    pass
                    
    return edges

def _lnw_construct(k, g):
    """
    Fallback: Lazebnik-Ustimenko-Woldar construction.
    """
    # Heuristic for d
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)

def _lnw_construct_params(k, d):
    """
    Generate the Lazebnik-Ustimenko-Woldar graph L_{k,d}.
    """
    q = k
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
