"""Combined program: L-U-W construction with deadline-guarded random search refinement."""
import time
import random
import sys

# Increase recursion limit if needed, though we use iterative BFS
sys.setrecursionlimit(10000)

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit
    
    # 1. Always compute the L-U-W construction as a baseline.
    # This is guaranteed to be valid.
    try:
        n_luw, edges_luw = _luw_construct(k, g)
    except Exception:
        n_luw, edges_luw = 0, []

    best_n = n_luw
    best_edges = edges_luw

    # 2. If there is time and the parameters are "small enough" for random search to be plausible,
    # try to find a smaller graph.
    # For k=3, g=16, this is unlikely to succeed, but we try briefly.
    # Moore bound: 1 + k * sum((k-1)^i for i in range(g//2))
    if k == 3 and g <= 10:
        # For small g, random search is more likely to find optimal or near-optimal cages
        # which are much smaller than L-U-W.
        moore_bound = 1 + k * sum((k-1)**i for i in range(g//2))
        # Search range
        max_search_n = moore_bound + 100
        
        # Try random search
        while time.time() < deadline:
            for n in range(moore_bound, max_search_n + 1):
                if time.time() > deadline:
                    break
                # Try a few random instances
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    edges = _generate_random_regular_graph(n, k)
                    if edges is None:
                        continue
                    if _check_girth(edges, n, g):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
                            # If we hit the Moore bound, it's optimal
                            if n == moore_bound:
                                return n, edges
            
            # If we found something better, we can stop early if we are close to bound
            # Or continue if time permits to see if we can go lower
            if best_n == moore_bound:
                break
            
    # 3. Return the best result found
    return best_n, best_edges

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    # d is such that girth >= g
    # Girth of CD(d, q) is d + 5 for odd d.
    # So we need d + 5 >= g => d >= g - 5.
    # d must be odd.
    d_candidate = g - 5
    if d_candidate % 2 == 0:
        d_candidate -= 1
    if d_candidate < 3:
        d_candidate = 3
        if d_candidate % 2 == 0:
            d_candidate = 3 # 3 is odd
    
    d = d_candidate

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
    index, order, edges = {start: 0}, [start], []
    
    # BFS to build graph
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _generate_random_regular_graph(n, k):
    """Generate a random k-regular simple graph using configuration model."""
    if n * k % 2 != 0:
        return None
    if n < k:
        return None
    
    half_edges = n * k // 2
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)
    
    edges = set()
    adj = [[] for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Self-loop, try to fix
            fixed = False
            for j in range(i+2, len(stubs), 2):
                if stubs[j] != u:
                    stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                    v = stubs[i+1]
                    fixed = True
                    break
            if not fixed:
                return None
        
        if u == v:
            return None
            
        if (u, v) in edges or (v, u) in edges:
            # Multi-edge, try to fix by swapping
            fixed = False
            for j in range(i+2, len(stubs), 2):
                u2, v2 = stubs[j], stubs[j+1]
                # Check if swap resolves conflicts
                # This is a simplified check; full backtracking is expensive
                if u2 != u and u2 != v and v2 != u and v2 != v:
                    if (u, u2) not in edges and (u, v2) not in edges and (v, u2) not in edges and (v, v2) not in edges:
                        stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                        stubs[i+2], stubs[j+1] = stubs[j+1], stubs[i+2]
                        v = stubs[i+1]
                        fixed = True
                        break
            if not fixed:
                return None

        if (u, v) in edges or (v, u) in edges or u == v:
            return None

        edges.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        
    return list(edges)

def _check_girth(edges, n, g):
    """Check if graph has girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex up to depth g//2 - 1
    depth_limit = g // 2 - 1
    for start in range(n):
        visited = {start: 0}
        queue = [(start, 0, -1)]
        while queue:
            u, depth, parent = queue.pop(0)
            if depth > depth_limit:
                continue
            for v in adj[u]:
                if v == parent:
                    continue
                if v in visited:
                    cycle_len = depth + visited[v] + 1
                    if cycle_len < g:
                        return False
                else:
                    visited[v] = depth + 1
                    queue.append((v, depth + 1, u))
    return True
