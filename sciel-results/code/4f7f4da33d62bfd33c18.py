import random
import time
import numpy as np
from array import array

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        seen[u].add(v)
        seen[v].add(u)
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    if u2 in adj[u1] or v2 in adj[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    return True

def _girth_fast(adj, g):
    """
    Check if graph has girth >= g using non-backtracking walks.
    Returns True if girth >= g, False otherwise.
    Uses integer keys for visited states for speed.
    """
    n = len(adj)
    if n == 0:
        return True
    
    # Build adjacency as tuples for faster iteration
    adj_tup = [tuple(v) for v in adj]
    
    for start in range(n):
        # visited: set of (prev, curr) encoded as prev * n + curr
        visited = set()
        
        # Initialize: walks of length 1 from start
        queue = []
        for nb in adj_tup[start]:
            state = start * n + nb
            visited.add(state)
            queue.append((start, nb))
        
        # BFS up to depth g-2
        for depth in range(1, g - 1):
            new_queue = []
            for (prev, curr) in queue:
                for nxt in adj_tup[curr]:
                    if nxt == prev:
                        continue
                    # Check for cycle: nxt == start and total length < g
                    if nxt == start and depth + 1 < g:
                        return False
                    state = curr * n + nxt
                    if state not in visited:
                        visited.add(state)
                        new_queue.append((curr, nxt))
            if not new_queue:
                break
            queue = new_queue
    
    return True

def _voltage_lift_petersen(m, voltages):
    """
    Lift the Petersen graph (5-regular, 10 vertices) using voltage assignments
    in Z_m to produce a 10m-vertex 5-regular graph.
    
    Petersen graph vertices: 0-4 (outer pentagon), 5-9 (inner pentagram)
    Edges (15 total, indexed 0-14):
      0: (0,1)  1: (1,2)  2: (2,3)  3: (3,4)  4: (4,0)  -- outer cycle
      5: (5,7)  6: (7,9)  7: (9,6)  8: (6,8)  9: (8,5)  -- inner cycle
      10: (0,5) 11: (1,6) 12: (2,7) 13: (3,8) 14: (4,9) -- spokes
    """
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # 0-4: outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # 5-9: inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # 10-14: spokes
    ]
    
    n = 10 * m
    adj = [[] for _ in range(n)]
    
    for idx, (u, v) in enumerate(petersen_edges):
        volt = voltages[idx] % m
        for t in range(m):
            nu = u * m + t
            nv = v * m + (t + volt) % m
            if nv in adj[nu]:
                return None  # Multi-edge: not simple
            adj[nu].append(nv)
            adj[nv].append(nu)
    
    # Verify regularity
    for i in range(n):
        if len(adj[i]) != 5:
            return None
    
    return adj

def construct(k, g):
    deadline = time.time() + 145.0
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth_fast(adj, g):
            best = adj
            best_n = n
            return True
        return False

    # Moore bound for k-regular graph with girth g:
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
    
    # For k=5, g=9: moore = 106
    # Smallest known: 1116 vertices
    # Try sizes around 1100-1200 first, then expand
    
    # === Voltage-based search on Petersen graph ===
    # The Petersen graph is 5-regular with 10 vertices.
    # Lifting with voltages in Z_m gives a 5m-vertex 5-regular graph.
    # We need n = 10*m, so m must make n even (always true since 10m is even).
    # Try m values that give n near 1116: m = 112 gives n=1120, m=111 gives n=1110
    # For girth >= 9, we need to search voltage assignments.
    
    if k == 5:
        # Petersen graph has 15 edges, each needs a voltage in Z_m
        # The Petersen graph girth is 5, so we need voltages that "unwind" short cycles.
        # Try systematic voltage assignments for various m values.
        
        # m values to try (n = 10*m)
        m_candidates = []
        for m_try in range(100, 150):
            m_candidates.append(m_try)
        # Sort by closeness to 111.6
        m_candidates.sort(key=lambda x: abs(x - 111.6))
        
        for m in m_candidates:
            if time.time() > deadline - 20:
                break
            
            n = 10 * m
            
            # Try structured voltage assignments
            # Strategy: use voltages that are multiples of small generators to avoid short cycles
            # The Petersen graph has 5-cycles; we want the voltage sum around each 5-cycle to be non-zero mod m
            # and ideally large enough to prevent short lifted cycles.
            
            max_voltage_attempts = 80
            for va in range(max_voltage_attempts):
                if time.time() > deadline - 10:
                    break
                
                # Mix of random and structured voltages
                if va % 4 == 0:
                    # Pure random
                    voltages = [_randrange(rng, m) for _ in range(15)]
                elif va % 4 == 1:
                    # Structured: use a base value with random perturbations
                    base = _randrange(rng, m)
                    voltages = [(base + _randrange(rng, m)) % m for _ in range(15)]
                elif va % 4 == 2:
                    # Use distinct generators for different edge groups
                    g1 = _randrange(rng, m) or 1
                    g2 = _randrange(rng, m) or 2
                    g3 = _randrange(rng, m) or 3
                    voltages = []
                    for i in range(15):
                        if i < 5:
                            voltages.append((i * g1) % m)
                        elif i < 10:
                            voltages.append((i * g2) % m)
                        else:
                            voltages.append((i * g3) % m)
                else:
                    # Random with some zero structure
                    voltages = [_randrange(rng, m) if _randrange(rng, 3) != 0 else 0 for _ in range(15)]
                
                adj = _voltage_lift_petersen(m, voltages)
                if adj is None:
                    continue
                
                if consider(n, adj):
                    return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
                
                # Try some 2-switches
                for _ in range(3000):
                    if time.time() > deadline - 10:
                        break
                    if _swap_regular(adj, n, k, rng):
                        if consider(n, adj):
                            return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

    # === Random pairing + 2-switch search ===
    # Strategy: try sizes in order of likelihood of success
    # Start near known optimum, then expand outward
    sizes_to_try = []
    # Add sizes from 1100 to 1300 in steps of 2
    for n_try in range(1100, 1301, 2):
        if n_try % 2 == 0:  # n*k must be even, k=5 so n must be even
            sizes_to_try.append(n_try)
    # Add smaller sizes
    for n_try in range(1000, 1100, 10):
        if n_try % 2 == 0:
            sizes_to_try.append(n_try)
    # Add larger sizes
    for n_try in range(1302, 1501, 10):
        if n_try % 2 == 0:
            sizes_to_try.append(n_try)
    
    # Sort by priority: near 1116 first
    sizes_to_try.sort(key=lambda x: abs(x - 1116))
    
    # Deduplicate while preserving order
    seen_sizes = set()
    unique_sizes = []
    for s in sizes_to_try:
        if s not in seen_sizes:
            seen_sizes.add(s)
            unique_sizes.append(s)
    sizes_to_try = unique_sizes
    
    for n in sizes_to_try:
        if time.time() > deadline - 5:
            break
        
        adj = None
        # Try multiple random attempts for this size
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts and time.time() < deadline - 10:
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
            attempts += 1

        if adj is not None:
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

            # Try 2-switches to improve girth
            # Use more switches for promising sizes
            max_switches = 10000 if 1100 <= n <= 1200 else 5000
            for _ in range(max_switches):
                if time.time() > deadline - 5:
                    break
                if _swap_regular(adj, n, k, rng):
                    if consider(n, adj):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # Fallback: LUW (same as seed, but we likely won't reach here)
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
        if time.time() > deadline - 5:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
