import random
import time
import math
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Moore bound for n(3,16) is 512
    moore = 512
    
    # Try to build a 3-regular graph with girth >= g using configuration model
    # with girth constraint via BFS during construction
    best_result = None
    
    # Start from small n and increase
    # n must be even for 3-regular
    for n in range(moore, 1000, 2):
        if time.time() > deadline - 30.0:
            break
        
        # Try multiple random configurations
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            
            result = _try_build_girth_constrained(n, g, start_time, deadline)
            if result is not None:
                best_result = result
                return best_result
    
    if best_result is not None:
        return best_result
    
    # Fallback: use the voltage lift approach on Petersen
    return _petersen_lift(k, g, start_time, deadline)

def _try_build_girth_constrained(n, g, start_time, deadline):
    """
    Try to build a 3-regular graph on n vertices with girth >= g
    using the configuration model with dynamic girth checking.
    """
    # Create stubs: each vertex has 3 stubs
    stubs = []
    for v in range(n):
        for s in range(3):
            stubs.append((v, s))
    
    random.shuffle(stubs)
    
    # Build adjacency list incrementally
    adj = [[] for _ in range(n)]
    edge_set = set()
    
    # We'll pair stubs randomly, but check girth constraint
    # For girth checking, we maintain for each vertex the shortest cycle length
    # Actually, let's use a different approach: build the graph edge by edge,
    # and after each edge, check if it creates a short cycle
    
    # Pair stubs
    pairs = []
    for i in range(0, len(stubs), 2):
        u, su = stubs[i]
        v, sv = stubs[i + 1]
        if u == v:
            # Self-loop, invalid
            return None
        # Check for parallel edges
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
        pairs.append((u, v))
    
    # Now check girth: use BFS from each vertex to find shortest cycle
    # This is expensive, so let's do a quick check first
    # For 3-regular graph, girth >= g means no cycles of length < g
    
    # Quick girth check using BFS
    for start in range(n):
        if time.time() > deadline - 3.0:
            return None
        if not _check_girth_from(start, adj, g):
            return None
    
    # Verify it's 3-regular
    for v in range(n):
        if len(adj[v]) != 3:
            return None
    
    # Return the graph
    edges = list(edge_set)
    return (n, edges)

def _check_girth_from(start, adj, g):
    """
    Check if there's a cycle of length < g starting from 'start'.
    Returns True if girth is >= g (no short cycle found from this start).
    """
    # BFS with parent tracking to detect cycles
    # dist[v] = shortest distance from start
    # parent[v] = parent in BFS tree
    dist = [-1] * len(adj)
    parent = [-1] * len(adj)
    dist[start] = 0
    
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
            elif v != parent[u]:
                # Found a cycle: u - v is a back edge
                # Cycle length = dist[u] + dist[v] + 1
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < g:
                    return False
    
    return True

def _petersen_lift(k, g, start_time, deadline):
    """
    Voltage lift on Petersen graph as fallback.
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Generate all closed non-backtracking walks of length 3 to g-1
    walks = []
    for start_v in range(n_base):
        queue = deque()
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
        
        while queue:
            curr_v, prev_e, length, path = queue.popleft()
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
    
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    walks = list(unique_walks)
    
    walk_coeffs = []
    for w in walks:
        coeffs = [0] * len(edges)
        for e_id, sign in w:
            coeffs[e_id] += sign
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        walk_coeffs.append(nonzero)
    
    # Try prime orders starting from 52
    best_m = None
    best_volts = None
    
    for m in range(52, 300):
        if time.time() > deadline - 10.0:
            break
        if not _is_prime(m):
            continue
        
        max_tries = 500
        for trial in range(max_tries):
            if time.time() > deadline - 5.0:
                break
            
            volts = [random.randint(1, m - 1) for _ in range(len(edges))]
            
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
                if g_val == 1:
                    break
            if g_val != 1:
                continue
            
            valid = True
            for nonzero in walk_coeffs:
                s = 0
                for idx, c in nonzero:
                    s += c * volts[idx]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                best_m = m
                best_volts = volts
                break
        
        if best_m is not None:
            break
    
    if best_m is None:
        return _luw_fallback(k, g)
    
    candidate_n = 10 * best_m
    lift_edges = []
    for i, (u, v) in enumerate(edges):
        vol = best_volts[i]
        for t in range(best_m):
            u_lift = u * best_m + t
            v_lift = v * best_m + (t + vol) % best_m
            lift_edges.append((u_lift, v_lift))
    
    return candidate_n, lift_edges

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _luw_fallback(k, g):
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
