import time
import math
import random
from collections import deque

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _deadline_exceeded(limit=140):
    return _elapsed() > limit

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using BFS from each vertex's neighbor pairs.
    Returns True if girth >= g, False otherwise."""
    # For each vertex u, for each pair of neighbors (v1, v2),
    # check if there's a path from v1 to v2 of length < g-2 (excluding u)
    # If such path exists, cycle length = path_length + 2 < g
    
    max_path_len = g - 3  # We need path length <= g-3 for cycle < g
    
    for u in range(n):
        neighbors = adj[u]
        if len(neighbors) < 2:
            continue
        for i in range(len(neighbors)):
            v1 = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                v2 = neighbors[j]
                if v1 == v2:
                    continue
                # BFS from v1 to v2, excluding u, limited to depth max_path_len
                if v1 == v2:
                    return False  # shouldn't happen in simple graph
                
                visited = [False] * n
                visited[u] = True  # exclude u
                visited[v1] = True
                queue = deque([(v1, 0)])
                found = False
                
                while queue:
                    curr, dist = queue.popleft()
                    if dist >= max_path_len:
                        continue
                    for nxt in adj[curr]:
                        if visited[nxt]:
                            continue
                        visited[nxt] = True
                        if nxt == v2:
                            found = True
                            break
                        queue.append((nxt, dist + 1))
                    if found:
                        break
                if found:
                    return False
    return True

def construct_lw(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def construct_abs_heisenberg(k, g):
    # Try Heisenberg group H_q construction for k=5
    if k != 5:
        return None
    
    # n = q^3 for the standard Heisenberg group construction
    # Moore bound for k=5, g=9 is ~426, so q^3 should be > 426
    # Try various q values, including odd q for different generator sets
    
    # For even q: generators (±1,0,0), (0,±1,0), (0,0,q/2)
    # For odd q: we need different generators to get degree 5
    
    # Let's try a range of q values
    # q^3 vertices, need girth >= 9
    # Start with promising values
    candidates = []
    for q in range(8, 30):
        candidates.append(q)
    
    # Sort by n = q^3 to try smaller first (but we need n >= Moore bound)
    # Actually, we want the smallest n that satisfies girth >= 9
    # Moore bound is 426, so q >= 8 (8^3=512)
    
    best_result = None
    
    for q in candidates:
        if _deadline_exceeded(130):
            break
        
        n = q ** 3
        if n < 400:  # Below Moore bound, skip
            continue
            
        # Build the Heisenberg group graph
        # Elements: (x, y, z) with x, y, z in Z_q
        # Multiplication: (x1,y1,z1) * (x2,y2,z2) = (x1+x2, y1+y2, z1+z2+x1*y2)
        # Identity: (0,0,0)
        # Inverse: (x,y,z)^(-1) = (-x, -y, -z - x*y)
        
        adj = [[] for _ in range(n)]
        
        def idx(x, y, z):
            return (x * q + y) * q + z
        
        def get_coords(u):
            z = u % q
            y = (u // q) % q
            x = u // (q * q)
            return x, y, z
        
        # For degree 5, we need 5 generators (with their inverses giving the other direction)
        # For even q: use (1,0,0), (q-1,0,0), (0,1,0), (0,q-1,0), (0,0,q//2)
        # Note: (0,0,q//2) is its own inverse when q is even
        
        if q % 2 == 0:
            gens = [
                (1, 0, 0),
                (q-1, 0, 0),
                (0, 1, 0),
                (0, q-1, 0),
                (0, 0, q // 2)
            ]
        else:
            # For odd q, we need a different set of 5 generators
            # Try: (1,0,0), (q-1,0,0), (0,1,0), (0,q-1,0), (0,0,1)
            # But (0,0,1) has order q, and its inverse is (0,0,q-1)
            # This gives degree 6, not 5. We need exactly 5 neighbors.
            # Alternative: use a different group or different generators
            # For now, skip odd q or try a different approach
            # Actually, let's try using (0,0,1) and accept that we need to be careful
            # The degree will be 6 for odd q with these generators, which is wrong
            # Let's skip odd q for now
            continue
        
        # Build adjacency
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    u = idx(x, y, z)
                    for gx, gy, gz in gens:
                        vx = (x + gx) % q
                        vy = (y + gy) % q
                        vz = (z + gz + x * gy) % q
                        v = idx(vx, vy, vz)
                        if u < v:
                            adj[u].append(v)
                            adj[v].append(u)
        
        # Verify the graph is 5-regular
        is_regular = all(len(adj[u]) == 5 for u in range(n))
        if not is_regular:
            continue
        
        # Verify girth >= g
        if _verify_girth_fast(adj, n, g):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
            if _deadline_exceeded(130):
                break
    
    return best_result

def construct(k, g):
    global _start_time
    _start_time = time.time()
    
    # Try Heisenberg for k=5
    if k == 5:
        result = construct_abs_heisenberg(k, g)
        if result is not None:
            return result
            
    # Fallback to LW
    return construct_lw(k, g)
