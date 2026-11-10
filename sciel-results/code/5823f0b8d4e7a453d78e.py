"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (L-U-W) construction CD(d, q).
    For k=5, g=9, we use q=5, d=4, yielding 6250 vertices.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # L-U-W parameters
    # The graph CD(d, q) is q-regular and has girth at least d+5.
    # We need degree k, so q must be k (assuming k is a prime power, which 5 is).
    # We need girth >= g, so d+5 >= g => d >= g-5.
    # For odd g, d is typically chosen as g-5 if (g-5) is odd? 
    # The theorem usually states for odd d, girth >= d+5.
    # If g=9, d=4 is even. 
    # Let's check the parity constraint in the original code.
    # Original code: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9, g-5=4 (even), so d = 9-4=5.
    # If d=5, girth >= 5+5=10.
    # n = 2 * q^(d+1) = 2 * 5^6 = 31250.
    # This is worse!
    
    # Let's re-read the L-U-W theorem conditions.
    # "girth is at least d+5 for odd d".
    # If d is even, the girth bound might be different or the construction might not apply directly in the same way.
    # However, the original code used d=5 for g=9.
    # If we can use d=4, we get n=6250.
    # Does CD(4, 5) have girth >= 9?
    # If the theorem requires odd d, we cannot use d=4.
    # Let's assume the original code's logic was correct for validity.
    # If d=5, n=31250.
    # If d=4, n=6250.
    # I will try d=4 and verify girth. If it fails, I fall back to d=5.
    
    q = k
    # Try to use the smallest d such that girth condition is met.
    # If d is odd, girth >= d+5.
    # If we set d=4 (even), we hope girth is still high enough.
    # Let's calculate both and pick the one that works and is smallest.
    
    candidates = []
    for d in [4, 5, 6]:
        if d + 5 >= g:
            n_vertices = 2 * (q ** (d + 1))
            candidates.append((n_vertices, d))
    
    # Sort by n_vertices
    candidates.sort(key=lambda x: x[0])
    
    best_n = None
    best_edges = None
    
    for n_vertices, d in candidates:
        if time.time() > deadline:
            break
        
        try:
            n, edges = _luw_construct(q, d)
            # Verify girth and regularity
            if _verify(k, g, n, edges):
                best_n = n
                best_edges = edges
                break
        except Exception:
            continue
            
    if best_n is None:
        # Fallback to a known valid construction, e.g., d=5
        n, edges = _luw_construct(q, 5)
        return n, edges
        
    return best_n, best_edges

def _luw_construct(q, d):
    """Constructs the L-U-W graph CD(d, q)."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Define the rule for coefficients
    # The rule maps c to (a, b) such that l[c] - p[c] = l[a] * p[b]
    # For c=2, a=1, b=1 is standard.
    rule = {}
    # We need to define rule for c=2..d
    # Standard L-U-W uses a specific sequence.
    # The original code had:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _verify(k, g, n, edges):
    """Verifies regularity and girth."""
    # Check regularity
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    if any(d != k for d in degree):
        return False
        
    # Check girth using BFS from each vertex (expensive, but necessary for certainty)
    # Optimization: If n is large, this is slow. 
    # For n=6250, BFS from each vertex is 6250 * (5^4) ~ 2M ops. Manageable.
    # Actually, girth check can be done by finding the shortest cycle.
    # A simpler check: For each edge (u,v), remove it and find shortest path u-v. 
    # If min dist + 1 < g, fail.
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # For each edge, check if there is a cycle shorter than g
    # We can just do a BFS from u to v avoiding the direct edge?
    # Or simpler: BFS from each vertex up to depth (g-1)/2.
    # If we find a neighbor in the visited set that is not the parent, we have a cycle.
    # Cycle length = depth(u) + depth(v) + 1.
    
    max_depth = (g - 1) // 2
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        for u in queue:
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
# EVOLVE-BLOCK-END
