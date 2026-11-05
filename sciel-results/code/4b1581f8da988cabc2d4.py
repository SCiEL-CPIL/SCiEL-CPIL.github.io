"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _luw_construct(k, g):
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


def _generate_random_regular(k, n, seed=None):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    for _ in range(100):
        half_edges = []
        for v in range(n):
            half_edges.extend([v] * k)

        rng.shuffle(half_edges)

        adj = {v: set() for v in range(n)}
        valid = True

        for i in range(0, n * k, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)

        if valid:
            return [(min(u, v), max(u, v)) for u in adj for v in sorted(adj[u]) if u < v]

    return None


def _check_girth(adj, n, g):
    """Verify girth >= g using non-backtracking walks from every vertex.
    A cycle of length < g exists iff there's a non-backtracking closed walk of length < g.
    We do BFS on directed edges (u,v) where v != prev, tracking distance."""
    # Build directed adjacency: for each vertex, list of (next, prev)
    # State: (prev, curr) where we arrived at curr from prev
    # We want to detect if we can return to prev (i.e., a non-backtracking cycle)
    
    # For efficiency, use the fact that girth >= g means:
    # For each directed edge (u,v), the shortest non-backtracking walk from (u,v) 
    # that ends at an edge (w,u) has length >= g-2 (plus 2 for the closing = g)
    
    # BFS on state (prev, curr) with distance = number of steps taken so far
    # Start: for each directed edge (u,v), state (u,v) at dist 0
    # Transition: from (prev, curr) go to (curr, next) for next in adj[curr], next != prev
    # Cycle found: if we reach state (curr, prev) from (prev, curr) ... 
    # Actually simpler: BFS from each directed edge, see if we can return to the start vertex
    # with a non-backtracking path of length < g-1
    
    # More efficient: BFS where state is (prev, curr), and we check if we can reach
    # any state (curr, prev) (i.e., come back to prev via a non-backtracking path)
    # The cycle length would be dist + 1 (the edge back to prev)
    
    # For g=9, we need to check walks up to length 8
    max_depth = g - 2  # if we find a return in <= max_depth steps, cycle < g
    
    # Use array-based BFS for speed
    # State encoding: prev * n + curr
    # We'll do BFS from each directed edge
    
    # Precompute: for each vertex, its neighbors as a list
    adj_list = [list(s) for s in adj.values()]
    
    # For each starting directed edge (s, t), do BFS on states (prev, curr)
    # State (prev, curr): we are at curr, came from prev
    # We want to know if we can reach state (curr, prev) i.e., come back to prev
    # without immediately backtracking
    
    # Optimization: since we just need to check if ANY cycle < g exists,
    # and g is small (9), we can do a limited BFS
    
    # Alternative efficient approach: 
    # For each vertex v, do BFS up to depth (g-1)//2, and check for edges between
    # vertices at depths that would form a short cycle.
    
    half = (g - 1) // 2  # for g=9, half=4
    
    for root in range(n):
        # BFS from root, track parent and distance
        parent = [-1] * n
        dist = [-1] * n
        parent[root] = -2
        dist[root] = 0
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            for neighbor in adj_list[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    if dist[node] + 1 <= half:
                        queue.append(neighbor)
                elif parent[node] != neighbor and parent[neighbor] != node:
                    # Back edge: potential cycle of length dist[node] + dist[neighbor] + 1
                    if dist[node] + dist[neighbor] + 1 < g:
                        return False
    
    return True


def _construct_lift(base_edges, base_n, m, voltage_seed, k):
    """Construct the lift of a base graph with voltages in Z_m."""
    rng = random.Random(voltage_seed)
    
    voltages = {}
    for u, v in base_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
    
    n = base_n * m
    adj = {i: set() for i in range(n)}
    
    for u, v in base_edges:
        vol = voltages[(u, v)]
        for a in range(m):
            u_lift = u * m + a
            v_lift = v * m + (a + vol) % m
            adj[u_lift].add(v_lift)
            adj[v_lift].add(u_lift)
    
    # Check degree and simplicity
    for i in range(n):
        if len(adj[i]) != k:
            return None, None
    
    edges = [(min(u, v), max(u, v)) for u in adj for v in sorted(adj[u]) if u < v]
    return n, edges


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (evaluator kills at 150s)
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (5,9): 2 * (5^4 + 5^3 + 5^2 + 5) / 2 + 1... 
    # Actually Moore bound for k-regular, girth g: 
    # For odd g: 1 + k * sum_{i=0}^{(g-1)/2} (k-1)^i
    # For g=9, k=5: 1 + 5*(1 + 4 + 16 + 64 + 256) = 1 + 5*341 = 1706? 
    # Wait, that's the Moore bound for the diameter. Let me recalculate.
    # Moore bound M(k,g) for odd g: 1 + k * sum_{i=0}^{(g-1)/2 - 1} (k-1)^i + k * (k-1)^{(g-1)/2}
    # For g=9: (g-1)/2 = 4, so sum from i=0 to 3: 1+4+16+64=85, plus 5*256=1280, total 1+5*85+1280 = 1+425+1280=1706
    # Hmm, but the problem says lower bound is 428. Let me just use the given bounds.
    
    # Try voltage graph lifts with small base graphs
    # Key insight: smaller lifts are faster to check, so try them first
    base_sizes = [10, 12, 14, 16, 18, 20]
    
    for base_n in base_sizes:
        if time.time() > deadline:
            break
        
        num_attempts = 8
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            seed = random.randint(0, 2**31)
            base_edges = _generate_random_regular(k, base_n, seed)
            if base_edges is None:
                continue
            
            # Try lift sizes, starting from smaller (faster to verify)
            m_values = []
            for m in range(2, 120):
                total = base_n * m
                if total < 428:  # below lower bound, skip
                    continue
                if total > 1200:  # don't go too large
                    break
                m_values.append(m)
            
            # Sort by total size (smaller first for faster verification)
            m_values.sort()
            
            for m in m_values:
                if time.time() > deadline:
                    break
                
                # Check time budget for girth verification
                n_total = base_n * m
                # Rough estimate: girth check is O(n * k * (g/2)) per vertex
                # For n=1000, k=5, g=9: about 1000*5*4 = 20000 operations per root, *1000 roots = 20M
                # That might be too slow. Let's limit.
                if n_total > 800 and time.time() > deadline - 30:
                    break
                
                for vs in range(3):
                    if time.time() > deadline:
                        break
                    
                    n, edges = _construct_lift(base_edges, base_n, m, seed + vs, k)
                    if n is None:
                        continue
                    
                    adj = {i: set() for i in range(n)}
                    for u, v in edges:
                        adj[u].add(v)
                        adj[v].add(u)
                    
                    if _check_girth(adj, n, g) and n < best_n:
                        best_n = n
                        best_edges = edges
                        # Found a valid one, can try to find smaller ones
                        # but break inner loops to save time
                        break
                # If we found a good one, try next m (smaller lifts)
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW construction
    n, edges = _luw_construct(k, g)
    return n, edges
# EVOLVE-BLOCK-END
