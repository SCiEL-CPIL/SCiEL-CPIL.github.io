import time
import random
import heapq
from collections import deque

def _gf16_table():
    """Build GF(16) addition/multiplication tables using primitive polynomial x^4+x+1."""
    add = [[0]*16 for _ in range(16)]
    mul = [[0]*16 for _ in range(16)]
    # Represent elements as 4-bit integers; addition is XOR
    for a in range(16):
        for b in range(16):
            add[a][b] = a ^ b
    # Multiplication via polynomial arithmetic mod x^4+x+1
    def gf16_mul(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 0x10:  # x^4 term
                a ^= 0x11  # reduce by x^4 + x + 1
        return result
    for a in range(16):
        for b in range(16):
            mul[a][b] = gf16_mul(a, b)
    return add, mul

def _gq44_construct():
    """
    Construct a 5-regular graph from GQ(4,4) incidence structure.
    
    GQ(4,4) has 200 points, 200 lines, each point on 9 lines, each line has 9 points.
    The incidence graph is 9-regular bipartite with girth 8.
    We remove a 4-regular 1-factorization to get a 5-regular subgraph.
    """
    add, mul = _gf16_table()
    
    # Points: (x, y) where x, y in GF(16)
    points = [(x, y) for x in range(16) for y in range(16)]
    # Lines: (a, b) where a, b in GF(16)
    lines = [(a, b) for a in range(16) for b in range(16)]
    
    point_idx = {p: i for i, p in enumerate(points)}
    line_idx = {l: i for i, l in enumerate(lines)}
    
    # Incidence: point (x,y) is on line (a,b) iff y = a*x + b
    # Build 9-regular bipartite adjacency
    n_points = 200
    n_lines = 200
    n_total = n_points + n_lines
    
    # For each point, find its 9 incident lines
    point_to_lines = [[] for _ in range(n_points)]
    for x in range(16):
        for y in range(16):
            pi = point_idx[(x, y)]
            for a in range(16):
                b = add[y, mul[a, x]]  # b = y - a*x = y + a*x in GF(16)
                li = line_idx[(a, b)]
                point_to_lines[pi].append(li)
    
    # Verify: each point should have exactly 9 lines? No, 16 lines.
    # Wait: for fixed (x,y), as a ranges over GF(16), b = y + a*x gives 16 distinct lines.
    # But GQ(4,4) should have 9 lines per point. Let me reconsider.
    
    # Actually GQ(s,t) has (s+1)(st+1) points, each on t+1 lines.
    # GQ(4,4): 5*17=85 points? No, that's not right either.
    # GQ(s,t): (s+1)(st+1) points, (t+1)(st+1) lines.
    # GQ(4,4): 5*17=85 points, 5*17=85 lines. Each point on t+1=5 lines.
    # Each line has s+1=5 points.
    # Total incidences: 85*5 = 425. Bipartite graph: 170 vertices, 425 edges, 5-regular.
    # Girth of GQ(s,t) incidence graph is 4 (bipartite, has 4-cycles).
    
    # Hmm, this doesn't give girth 8. Let me reconsider.
    
    # For GQ(4,4), the incidence graph has girth 4 (since it's a generalized quadrangle,
    # any two points are on at most one line, but adjacent points share a line).
    # Actually in a GQ, two points are either collinear (on a common line) or not.
    # The incidence graph: points and lines as vertices, edge if incident.
    # A 4-cycle would be: P1-L1-P2-L2-P1, meaning P1,P2 both on L1 and L2, 
    # which violates the GQ axiom (two points on at most one line).
    # So no 4-cycles! The girth is at least 6.
    # A 6-cycle: P1-L1-P2-L2-P3-L3-P1. This requires P1,P2 on L1; P2,P3 on L2; P3,P1 on L3.
    # In GQ(4,4), this can happen. The girth of GQ(4,4) incidence graph is 6.
    
    # So this approach gives girth 6, not 8 or 9. Not useful for g=9.
    
    # Let me try a different approach: use the LUW construction but optimize it.
    return None

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _compute_girth_bounded(adj, n, max_girth):
    """Compute the girth of the graph, bounded by max_girth. Returns min cycle length or max_girth if none found."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_girth - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[u] + 1 > dist[v]:
                    cycle_len = dist[u] + 1 + dist[v]
                    if cycle_len < max_girth:
                        return cycle_len
    return max_girth

def _verify_girth_nonbacktracking(adj, n, g):
    """Verify girth >= g using non-backtracking walks from every vertex."""
    for start in range(n):
        # BFS on non-backtracking walks: state is (current, previous)
        visited = set()
        visited.add((start, -1))
        queue = deque([(start, -1, 0)])
        while queue:
            u, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                if (v, u) in visited:
                    # Found a cycle of length depth + 1 + (distance from start to v in non-backtracking sense)
                    # Actually, this indicates a cycle. The cycle length is at most depth + 1 + something.
                    # For simplicity, if we revisit (v, u), there's a cycle of length <= depth + 1 + dist
                    # This is complex; let's use a simpler check
                    if depth + 1 < g:
                        return False
                    continue
                if depth + 1 < g:
                    visited.add((v, u))
                    queue.append((v, u, depth + 1))
    return True

def _random_k_regular_simple(k, n, max_attempts=50):
    """Generate a random simple k-regular graph using configuration model with retries."""
    for _ in range(max_attempts):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid and all(len(adj[i]) == k for i in range(n)):
            return adj
    return None

def _edge_switch_inplace(adj, n):
    """Perform a random 2-switch in-place. Returns True if successful, False otherwise."""
    # Collect all edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    if len(edges) < 2:
        return False
    
    # Try a few times to find a valid switch
    for _ in range(20):
        (a, b), (c, d) = random.sample(edges, 2)
        if len({a, b, c, d}) < 4:
            continue
        
        # Remove old edges
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        
        # Choose new pairing
        if random.random() < 0.5:
            new_pairs = [(a, c), (b, d)]
        else:
            new_pairs = [(a, d), (b, c)]
        
        # Check validity
        valid = True
        for u, v in new_pairs:
            if u == v or v in adj[u]:
                valid = False
                break
        
        if not valid:
            # Revert
            adj[a].add(b)
            adj[b].add(a)
            adj[c].add(d)
            adj[d].add(c)
            continue
        
        # Add new edges
        for u, v in new_pairs:
            adj[u].add(v)
            adj[v].add(u)
        
        return True
    
    return False

def _switching_construct(k, g, n, time_limit=20.0):
    """Construct a k-regular graph of girth >= g using edge switching."""
    start_time = time.time()
    
    # Generate initial random k-regular simple graph
    adj = _random_k_regular_simple(k, n)
    if adj is None:
        return None
    
    # Compute initial girth
    current_girth = _compute_girth_bounded(adj, n, g)
    
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Perform switching to improve girth
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            break
        
        # Make a copy for the switch
        new_adj = [set(s) for s in adj]
        
        if not _edge_switch_inplace(new_adj, n):
            no_improve += 1
            continue
        
        # Check new girth
        new_girth = _compute_girth_bounded(new_adj, n, g)
        
        if new_girth >= current_girth:
            adj = new_adj
            if new_girth > current_girth:
                current_girth = new_girth
                no_improve = 0
                if current_girth >= g:
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
            else:
                no_improve += 1
        else:
            no_improve += 1
    
    return None

def construct(k, g):
    if k == 5 and g == 9:
        # First try the LUW construction which is fast and gives a valid graph
        luw_start = time.time()
        luw_result = _luw_construct(k, g)
        luw_n = luw_result[0]
        luw_time = time.time() - luw_start
        
        # Now try to find a smaller graph using switching
        # The LUW construction for k=5, g=9 gives n=2500 (q=5, d=4, so 2*q*q^d = 2*5*625=6250?)
        # Actually let me compute: vertices are (side, x) where side in {0,1}, x is a tuple of length d+1
        # x[0] is None, x[1] in range(q), x[2..d] in range(q)
        # So q^d vertices per side, 2*q^d total. For q=5, d=4: 2*625=1250.
        
        # Try switching to find smaller graphs
        remaining_time = 110.0 - luw_time
        if remaining_time > 5.0:
            # Try a few smaller n values
            target_ns = [430, 450, 500, 550, 600]
            time_per_attempt = remaining_time / len(target_ns)
            
            for n in target_ns:
                if time.time() > 112:
                    break
                res = _switching_construct(k, g, n, time_limit=time_per_attempt)
                if res is not None:
                    return res
        
        # Return the LUW result
        return luw_result
    
    # Fallback to LUW
    return _luw_construct(k, g)
