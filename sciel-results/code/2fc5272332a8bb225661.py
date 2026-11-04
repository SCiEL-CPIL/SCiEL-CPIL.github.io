import time
import math
import random

def _deadline_check(start_time, limit=140.0):
    """Return True if we've exceeded the deadline."""
    return (time.time() - start_time) > limit

def _verify_girth(n, edges, k, g):
    """Verify girth >= g using non-backtracking BFS from each vertex."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Non-backtracking walk verification
    # A cycle of length < g exists iff there's a non-backtracking walk of length < g that returns to start
    # For girth >= g, no non-backtracking walk of length 1..g-1 should return to its start
    
    # BFS from each vertex using non-backtracking walks
    # State: (current_vertex, previous_vertex)
    # We track the shortest non-backtracking walk from each vertex back to itself
    
    # For efficiency, do BFS from vertex 0 first (symmetry), then spot-check
    # Actually, for correctness we need all vertices, but let's be efficient
    
    from collections import deque
    
    for start in range(n):
        # BFS over non-backtracking walks
        # visited: dict from (current, prev) to distance
        # We want to find if any walk of length < g returns to start
        # i.e., is there (start, prev) reachable in < g steps where the walk started at start?
        
        # Actually, a non-backtracking walk from v0 to v1 to ... to vl where vi != v(i-2)
        # A cycle of length l exists if there's a non-backtracking walk of length l from v0 back to v0
        # where v0 != v2 (non-backtracking at the closure)
        
        # BFS: state = (current, prev), start states = (start, neighbor) for each neighbor
        # We track distance from original start
        # If we reach (start, prev) with distance < g and prev != the first vertex, we have a cycle
        
        dist = {}
        queue = deque()
        for nb in adj[start]:
            state = (nb, start)
            dist[state] = 1
            queue.append(state)
        
        while queue:
            cur, prev = queue.popleft()
            d = dist[(cur, prev)]
            if d >= g:
                continue
            # Check if we've returned to start
            if cur == start and d < g and prev != start:
                # This means we have a non-backtracking walk of length d from start back to start
                # where the last step is from prev to start, and prev != start (non-backtracking)
                # But we also need prev != the vertex before start in the walk... 
                # Actually non-backtracking means cur != the vertex before prev
                # The walk is: start -> ... -> prev -> start
                # Non-backtracking at the last step: start != the vertex before prev
                # This is automatically satisfied since we're doing non-backtracking BFS
                # But we need d >= 3 for a valid cycle (simple graph)
                if d >= 3:
                    return False
            for nb in adj[cur]:
                if nb == prev:
                    continue  # non-backtracking
                new_state = (nb, cur)
                if new_state not in dist:
                    dist[new_state] = d + 1
                    queue.append(new_state)
    
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

def construct_abs(k, g):
    # Alon-Babai-Suzuki construction on Heisenberg group H_q
    if g % 2 == 0:
        mb = 2 * sum((k-1)**i for i in range(g//2))
    else:
        mb = 1 + k * sum((k-1)**i for i in range((g-1)//2))
        
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    q = 2
    while True:
        if is_prime(q):
            if q**3 >= mb:
                break
        q += 1
        
    if k == 4:
        gens = [
            (1, 0, 0),
            (q-1, 0, 0),
            (0, 1, 0),
            (0, q-1, 1)
        ]
    elif k == 6:
        gens = [
            (1, 0, 0),
            (q-1, 0, 0),
            (0, 1, 0),
            (0, q-1, 1),
            (0, 0, 1),
            (0, 0, q-1)
        ]
    else:
        return construct_lw(k, g)

    n = q**3
    edges = []
    def idx(x, y, z):
        return (x * q + y) * q + z
        
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
                        edges.append((u, v))
    return n, edges

def construct_circulant(k, g, max_q=None):
    """Try circulant graphs for odd k. For k=5: gens {1,-1,2,-2,q/2}"""
    if k % 2 != 0:
        if g % 2 == 0:
            mb = 2 * sum((k-1)**i for i in range(g//2))
        else:
            mb = 1 + k * sum((k-1)**i for i in range((g-1)//2))
        
        if max_q is None:
            max_q = mb + 2000
        
        # For k=5: generators {1, q-1, 2, q-2, q/2}
        # For general odd k: {1, q-1, 2, q-2, ..., k//2, q-k//2, q/2}
        # k = 2*(k//2) + 1, so we have k//2 pairs + 1 self-inverse = k generators
        
        start_q = mb if mb % 2 == 0 else mb + 1
        best_n = None
        best_edges = None
        
        for q in range(start_q, max_q + 1, 2):
            if q < 2 * (k // 2) + 4:  # need q large enough for distinct generators
                continue
            
            # Build generator list
            gens = []
            for i in range(1, k // 2 + 1):
                gens.append(i)
                gens.append(q - i)
            gens.append(q // 2)
            
            # Check all generators are distinct and non-zero
            if len(set(gens)) != k or 0 in gens:
                continue
            
            # Build adjacency and check girth
            # For circulant, adjacency is symmetric
            # Check girth using BFS from vertex 0
            # Since it's a Cayley graph, if girth from 0 is >= g, 
            # we still need to check all vertices for full verification
            # But for circulant graphs, the girth is the same from every vertex
            # (vertex-transitive), so checking from 0 suffices!
            
            # BFS from 0 to find shortest cycle
            # Non-backtracking BFS
            from collections import deque
            
            # State: (current, prev)
            # Start: for each generator g, state (g, 0) at distance 1
            dist = {}
            queue = deque()
            for gen in gens:
                state = (gen % q, 0)
                dist[state] = 1
                queue.append(state)
            
            found_short_cycle = False
            while queue:
                cur, prev = queue.popleft()
                d = dist[(cur, prev)]
                if d >= g:
                    continue
                # Check if returned to 0
                if cur == 0 and d >= 3:
                    # Non-backtracking: prev != 0 is guaranteed since we skip prev
                    # But we need the walk to be non-backtracking at closure
                    # The walk ends with ... -> prev -> 0, and prev != 0 (since we don't go back to prev)
                    # Also need that the step before prev wasn't 0 (handled by non-backtracking)
                    found_short_cycle = True
                    break
                for gen in gens:
                    nb = (cur + gen) % q
                    if nb == prev:
                        continue
                    new_state = (nb, cur)
                    if new_state not in dist:
                        dist[new_state] = d + 1
                        queue.append(new_state)
            
            if not found_short_cycle:
                # Valid! Build edges
                n = q
                edges = []
                for u in range(n):
                    for gen in gens:
                        v = (u + gen) % q
                        if u < v:
                            edges.append((u, v))
                return n, edges
        
    return None

def construct(k, g):
    start_time = time.time()
    
    results = []
    
    # Try LW construction
    try:
        n_lw, edges_lw = construct_lw(k, g)
        results.append((n_lw, edges_lw, 'lw'))
    except Exception:
        pass
    
    # Try ABS construction
    try:
        n_abs, edges_abs = construct_abs(k, g)
        if n_abs is not None:
            results.append((n_abs, edges_abs, 'abs'))
    except Exception:
        pass
    
    # Try circulant for odd k
    if k % 2 != 0:
        try:
            circ = construct_circulant(k, g)
            if circ is not None:
                n_circ, edges_circ = circ
                results.append((n_circ, edges_circ, 'circ'))
        except Exception:
            pass
    
    if not results:
        # Fallback: return a trivial valid graph if possible
        # This shouldn't happen for the given parameters
        raise RuntimeError("No valid construction found")
    
    # Pick the smallest
    results.sort(key=lambda x: x[0])
    best_n, best_edges, best_name = results[0]
    
    # Verify the best result
    if _verify_girth(best_n, best_edges, k, g):
        return best_n, best_edges
    else:
        # Find next valid
        for n, edges, name in results[1:]:
            if _verify_girth(n, edges, k, g):
                return n, edges
        # If none valid, raise
        raise RuntimeError("No valid construction passed verification")

