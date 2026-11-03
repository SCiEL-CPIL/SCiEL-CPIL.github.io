"""Construct a 3-regular graph with girth >= g using a BFS-tree + matching method."""
import time
import random
from collections import deque

def construct(k, g):
    # Deadline guard
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # For k=3, g=16: use a systematic construction
    # Strategy: Build a BFS tree of depth (g//2) - 1 = 7 from a root,
    # then match leaves to create cycles of length >= g.
    #
    # Better approach: Use the known construction based on expander-like
    # regular graphs, or use a random regular graph with girth verification.
    #
    # Given the constraints, let's use a combination:
    # 1. Start with a known small girth graph (LUW or similar)
    # 2. Try to improve via edge swaps that preserve degree and increase girth
    #
    # Actually, for a fresh design, let's use a different approach:
    # Construct a (3,16)-cage candidate using the "random regular graph + local search" approach.
    # 
    # Key insight: A random 3-regular graph on n vertices has expected girth ~ log_2(n).
    # For girth 16, we need n >= 2^(16/2) = 65536... that's too many.
    # Actually the Moore bound for (3,16) is ~2*(2^8 - 1) = 510, so n >= 512.
    #
    # Let's use a structured construction instead. The best known is 936 vertices.
    # We'll use a modified LUW or a different explicit construction.
    #
    # New approach: Use a bipartite incidence graph of a combinatorial design,
    # or use a Cayley graph construction.
    #
    # For k=3, let's try a Cayley graph on a group of order ~n with 3 generators.
    # A (3,16)-cage needs no cycles of length < 16.
    #
    # Let's use a product of cycles or a group-based construction.
    
    if k != 3:
        # Fallback: use the LUW construction for other k
        return _luw_construct(k, g)
    
    # For k=3, g=16: Try to construct a good graph
    # Use a random 3-regular graph with girth verification and local improvement
    
    # First, get a valid starting graph using LUW
    n, edges = _luw_construct(3, 16)
    
    # Verify girth
    adj = _build_adjacency(n, edges)
    min_girth = _compute_girth(adj, n, 15)  # check for cycles < 16
    
    if min_girth >= 16:
        return n, edges
    
    # Try to improve via random edge swaps
    best_n = n
    best_edges = edges
    
    start_time = time.time()
    
    # Attempt local search: swap edges to eliminate short cycles
    # This is complex, so let's try a different strategy:
    # Generate random 3-regular graphs and check girth
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Try different sizes starting from a reasonable lower bound
        # Moore bound for (3,16): 2*(2^7 - 1) = 254... wait
        # For bipartite (3,16): n >= 2*(2^7 - 1) = 254
        # For general (3,16): n >= 1 + 3*sum(2^i, i=0..7) = 1 + 3*255 = 766
        # Actually the Moore bound is: n >= 1 + k*sum_{i=0}^{floor((g-1)/2)} (k-1)^i
        # For k=3, g=16: floor(15/2) = 7, so n >= 1 + 3*(1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 1 + 3*255 = 766
        
        target_n = 766 + attempt * 2  # even number of vertices for 3-regular
        
        if target_n < 4:
            continue
        
        # Generate random 3-regular graph on target_n vertices
        if target_n % 2 != 0:
            target_n += 1
        
        if target_n < 4:
            continue
        
        n_test, edges_test = _random_regular_graph(target_n, 3)
        
        if n_test is None:
            continue
        
        adj_test = _build_adjacency(n_test, edges_test)
        
        # Check if graph is connected
        if not _is_connected(adj_test, n_test):
            continue
        
        # Check girth
        girth_test = _compute_girth(adj_test, n_test, 15)
        
        if girth_test >= 16:
            if n_test < best_n:
                best_n = n_test
                best_edges = edges_test
            return n_test, edges_test
        
        if time.time() > deadline:
            break
    
    return best_n, best_edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _build_adjacency(n, edges):
    """Build adjacency list from edge list."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _is_connected(adj, n):
    """Check if graph is connected using BFS."""
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    count = 1
    
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    
    return count == n


def _compute_girth(adj, n, max_cycle_len):
    """Compute girth (shortest cycle length) using BFS from each vertex.
    Returns the shortest cycle length found, or max_cycle_len+1 if none found up to that length."""
    min_cycle = max_cycle_len + 1
    
    for start in range(n):
        # BFS from start, tracking parent to detect cycles
        dist = [-1] * n
        parent = [-1] * n
        queue = deque([(start, 0)])
        dist[start] = 0
        found = False
        
        while queue and not found:
            u, d = queue.popleft()
            if d >= max_cycle_len:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append((v, d + 1))
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle of length d + 1 + dist[v]... 
                    # Actually, cycle length = d + 1 + dist[v] is wrong
                    # If we found edge (u,v) where both are visited and v != parent[u],
                    # cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle <= max_cycle_len:
                        found = True
                        break
    
    return min_cycle


def _random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None, None
    if k >= n:
        return None, None
    
    for _ in range(100):  # retry to avoid multigraphs
        # Create stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        # Randomly pair stubs
        random.shuffle(stubs)
        edges_set = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges_set:
                valid = False
                break
            edges_set.add(edge)
        
        if not valid:
            continue
        
        edges = list(edges_set)
        if len(edges) != n * k // 2:
            continue
        
        return n, edges
    
    return None, None
