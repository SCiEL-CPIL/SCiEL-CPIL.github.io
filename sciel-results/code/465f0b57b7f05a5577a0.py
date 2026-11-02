"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d = g - 4
    d = max(3, d)

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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _check_girth_bfs(n, adj, g):
    """Check if graph has girth >= g using BFS from each vertex (non-backtracking)."""
    for s in range(n):
        # dist[v] = (shortest non-backtracking walk length from s to v, predecessor)
        dist = [(-1, -1)] * n
        dist[s] = (0, -1)
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            du, pu = dist[u]
            if du + 1 >= g:
                continue
            for v in adj[u]:
                if v == pu:
                    continue  # non-backtracking
                if dist[v][0] == -1:
                    dist[v] = (du + 1, u)
                    queue.append(v)
                elif dist[v][0] > du + 1:
                    # Found a shorter path -> cycle of length dist[v][0] + du + 1 - (something)
                    # Actually if we reach v again with a different predecessor, cycle length
                    # is dist[v][0] + (du + 1) - 2*common_prefix... 
                    # Simpler: if we can reach v in du+1 steps and dist[v] was already set
                    # to a value >= du+1, the cycle length is at most dist[v][0] + du + 1
                    # but we need to be careful. Let's use the standard approach:
                    # A cycle exists of length <= dist[v][0] + du + 1 if predecessors differ
                    # But for exact girth check, BFS tree + back edges gives cycle lengths.
                    # For non-backtracking: if dist[v] is already set and v != pu,
                    # cycle length = dist[v][0] + du + 1
                    cycle_len = dist[v][0] + du + 1
                    if cycle_len < g:
                        return False
                    if dist[v][0] > du + 1:
                        dist[v] = (du + 1, u)
                        queue.append(v)
                else:
                    # dist[v][0] == du + 1, check if it forms a cycle
                    if dist[v][1] != u:
                        cycle_len = dist[v][0] + du + 1
                        if cycle_len < g:
                            return False
    return True


def _voltage_graph_search(k, g, deadline, seed=42):
    """
    Voltage graph (lift) construction:
    - Base graph B is a small k-regular graph (or we use a tree-like base)
    - Actually, for k-regular covers, use a base graph with k edges per vertex
    - Group G = Z_m, total vertices = m * |V(B)|
    - Optimize voltage labels via simulated annealing
    
    Strategy: Use a base that is a single vertex with k loops? No, must be simple.
    Use a base cycle C_m? That gives 2-regular.
    
    Better approach: Start from a known good small k-regular graph and try to find
    a voltage assignment on a base that produces a larger girth cover.
    
    Actually, the most practical approach within time: 
    1. Try the LUW construction (gives specific n)
    2. Try random k-regular graphs of various sizes and check girth
    3. Use simulated annealing on edge swaps to increase girth
    
    For k=3, g=16: LUW gives some n. Let's also try random 3-regular graphs.
    """
    rng = random.Random(seed)
    best_n = float('inf')
    best_edges = None
    
    # Try LUW first
    try:
        n_luw, edges_luw = _luw_construct(k, g, deadline)
        if n_luw < best_n:
            best_n = n_luw
            best_edges = edges_luw
    except:
        pass
    
    # Try random k-regular graph generation with girth check
    # For 3-regular graphs, use configuration model
    def generate_random_k_regular(n, rng):
        """Generate random k-regular graph using configuration model."""
        if n * k % 2 != 0:
            return None
        if n < k + 1:
            return None
        # Create half-edges
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        rng.shuffle(half_edges)
        # Pair them up
        edges_set = set()
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                return None  # loop
            key = (min(u, v), max(u, v))
            if key in edges_set:
                return None  # multi-edge
            edges_set.add(key)
        # Check it's connected (optional, but let's skip for speed)
        return list(edges_set)
    
    def check_girth_fast(n, edges, g):
        """Check girth >= g using BFS."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return _check_girth_bfs(n, adj, g)
    
    # Try various sizes near the LUW result
    search_sizes = []
    if best_n < float('inf'):
        # Try sizes from Moore bound up to best_n
        # Moore bound for (3,16): 2 + 3*(2^7 - 2) = 2 + 3*126 = 380... 
        # Actually Moore bound: 2 + k*(k-1)^((g-2)/2) for even g
        # For k=3, g=16: 2 + 3*2^7 = 2 + 384 = 386? No...
        # Moore bound n >= 2 + k * sum_{i=0}^{(g/2-1)} (k-1)^i = 2 + 3*(1 + 2 + 4 + ... + 2^6) = 2 + 3*127 = 383
        moore = 2 + k * ((k - 1) ** (g // 2 - 1) - 1) // (k - 2) if k > 2 else 2
        # Actually for k=3: 2 + 3*(2^7 - 1)/(2-1) = 2 + 3*127 = 383
        # But the problem says lower bound is 512, so let's use that
        start_size = max(moore, 512)
        end_size = best_n
        
        # Try a few random sizes
        if end_size > start_size:
            num_tries = 50
            step = max(1, (end_size - start_size) // num_tries)
            for i in range(num_tries):
                size = start_size + i * step
                if size % 2 != 0 and k % 2 != 0:
                    size += 1  # n*k must be even
                search_sizes.append(size)
    
    # Also try some fixed sizes
    for size in [512, 520, 540, 560, 580, 600, 640, 700, 768, 800, 860, 900, 936]:
        if size not in search_sizes:
            if k * size % 2 == 0:
                search_sizes.append(size)
    
    search_sizes = sorted(set(search_sizes))
    
    # Try each size with random graphs
    for size in search_sizes:
        if time.time() > deadline - 10:
            break
        if size >= best_n:
            continue
        if k * size % 2 != 0:
            continue
        if size < k + 1:
            continue
        
        # Try several random graphs of this size
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            edges = generate_random_k_regular(size, rng)
            if edges is None:
                continue
            if check_girth_fast(size, edges, g):
                best_n = size
                best_edges = edges
                break
    
    if best_edges is not None:
        return best_n, best_edges
    else:
        return best_n, best_edges


def construct(k, g):
    deadline = time.time() + 110  # safety margin
    
    # Try voltage graph / random search approach
    try:
        n, edges = _voltage_graph_search(k, g, deadline)
        if n is not None and edges is not None:
            return n, edges
    except:
        pass
    
    # Fallback to LUW
    try:
        return _luw_construct(k, g, deadline)
    except:
        # Last resort: return a trivial result
        return 0, []

# EVOLVE-BLOCK-END
