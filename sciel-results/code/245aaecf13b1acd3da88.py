import time
import sys
import random
from collections import deque

_DEADLINE = 115.0

def construct(k, g):
    t_start = time.time()
    
    # Fallback for general k, g
    if k != 4 or g != 10:
        return _luw(k, g)
    
    # Specific search for (4, 10) cage
    # Moore bound for (4, 10) is 246.
    # Best known is 320.
    # We search for a circulant graph C(n, {s1, s2})
    
    # Quick circulant search (fast, limited range)
    for n in range(246, 400):
        if time.time() - t_start > _DEADLINE - 10:
            break
        max_step = n // 2
        for s1 in range(1, max_step + 1):
            if time.time() - t_start > _DEADLINE - 10:
                break
            for s2 in range(s1 + 1, max_step + 1):
                if (s1 + s2) % n == 0:
                    continue
                if _check_girth_circulant(n, s1, s2, 10):
                    edges = []
                    for u in range(n):
                        v1 = (u + s1) % n
                        v2 = (u + s2) % n
                        if u < v1:
                            edges.append((u, v1))
                        if u < v2:
                            edges.append((u, v2))
                    return n, edges
                    
    # Stochastic local search: 2-switches to eliminate short cycles
    result = _stochastic_local_search(k, g, t_start)
    if result is not None:
        return result
    
    # Fallback to LUW
    return _luw(k, g)

def _stochastic_local_search(k, g, t_start):
    """
    Stochastic local search using 2-switches to maximize girth.
    Start with a random k-regular graph, then apply 2-switches to eliminate short cycles.
    """
    # Try various sizes starting from known best (320)
    target_sizes = [320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360]
    
    for n in target_sizes:
        if time.time() - t_start > _DEADLINE - 8:
            break
        
        # Generate random k-regular graph
        adjacency = _generate_random_k_regular(n, k)
        if adjacency is None:
            continue
        
        # Compute initial girth and cycle count
        girth, cycle_info = _compute_girth_and_cycles(adjacency, n, g)
        
        # If we already have girth >= g, return immediately
        if girth >= g:
            edges = _adjacency_to_edges(adjacency, n)
            return n, edges
        
        # Simulated annealing with 2-switches
        best_girth = girth
        best_adjacency = [list(row) for row in adjacency]
        current_girth = girth
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        max_iters = 50000
        
        for iteration in range(max_iters):
            if time.time() - t_start > _DEADLINE - 5:
                break
            
            # Generate a random 2-switch
            switch = _generate_2_switch(adjacency, n, k)
            if switch is None:
                continue
            
            u, v, w, x = switch
            # Apply 2-switch: remove (u,v) and (w,x), add (u,w) and (v,x)
            # or remove (u,v) and (w,x), add (u,x) and (v,w)
            # We need to check both possibilities
            
            # Try switch type 1: (u,w) and (v,x)
            adjacency[u][v] = 0
            adjacency[v][u] = 0
            adjacency[w][x] = 0
            adjacency[x][w] = 0
            adjacency[u][w] = 1
            adjacency[w][u] = 1
            adjacency[v][x] = 1
            adjacency[x][v] = 1
            
            new_girth, _ = _compute_girth_and_cycles(adjacency, n, g)
            
            if new_girth >= g:
                edges = _adjacency_to_edges(adjacency, n)
                return n, edges
            
            # Accept if improvement or with probability exp(-(delta)/T)
            delta = current_girth - new_girth  # positive means worse
            if delta <= 0:
                # Improvement or equal
                current_girth = new_girth
                if new_girth > best_girth:
                    best_girth = new_girth
                    best_adjacency = [list(row) for row in adjacency]
            else:
                # Worse, accept with probability
                T = T_start * (T_end / T_start) ** (iteration / max_iters)
                if random.random() < (2.718281828 ** (-delta / T)):
                    current_girth = new_girth
                else:
                    # Revert
                    adjacency[u][w] = 0
                    adjacency[w][u] = 0
                    adjacency[v][x] = 0
                    adjacency[x][v] = 0
                    adjacency[u][v] = 1
                    adjacency[v][u] = 1
                    adjacency[w][x] = 1
                    adjacency[x][w] = 1
        
        # If best found so far has girth >= g, return it
        if best_girth >= g:
            edges = _adjacency_to_edges(best_adjacency, n)
            return n, edges
    
    return None

def _generate_random_k_regular(n, k):
    """
    Generate a random k-regular graph using configuration model.
    Returns adjacency list (list of lists) or None if failed.
    """
    # Create half-edges: each vertex has k stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair up stubs to form edges
    adjacency = [[0] * n for _ in range(n)]
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            # Self-loop, retry
            return None
        if adjacency[u][v]:
            # Multi-edge, retry
            return None
        adjacency[u][v] = 1
        adjacency[v][u] = 1
    
    return adjacency

def _compute_girth_and_cycles(adjacency, n, g):
    """
    Compute the girth (shortest cycle length) and count cycles of length < g.
    Returns (girth, cycle_count).
    """
    # BFS from each vertex to find shortest cycle
    min_cycle = float('inf')
    cycle_count = 0
    
    for start in range(n):
        # BFS to find shortest cycle back to start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= g - 1:
                continue
            
            for v in range(n):
                if adjacency[u][v]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and dist[v] < dist[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            cycle_count += 1
    
    girth = min_cycle if min_cycle < float('inf') else g
    return girth, cycle_count

def _generate_2_switch(adjacency, n, k):
    """
    Generate a random valid 2-switch.
    Returns (u, v, w, x) where (u,v) and (w,x) are edges that can be switched.
    """
    # Collect all edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adjacency[u][v]:
                edges.append((u, v))
    
    if len(edges) < 2:
        return None
    
    # Try several random pairs
    for _ in range(10):
        e1, e2 = random.sample(edges, 2)
        u, v = e1
        w, x = e2
        
        # Check if the switch is valid (no self-loops, no multi-edges)
        # Switch type 1: (u,w) and (v,x)
        if u != w and v != x and not adjacency[u][w] and not adjacency[v][x]:
            return (u, v, w, x)
        # Switch type 2: (u,x) and (v,w)
        if u != x and v != w and not adjacency[u][x] and not adjacency[v][w]:
            return (u, v, x, w)
    
    return None

def _adjacency_to_edges(adjacency, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adjacency[u][v]:
                edges.append((u, v))
    return edges

def _check_girth_circulant(n, s1, s2, g):
    """
    Check if circulant graph C(n, {s1, s2}) has girth at least g.
    """
    expected = [1, 4, 12, 36, 108]
    
    visited = [-1] * n
    dist = [0] * n
    q = deque()
    q.append(0)
    visited[0] = 0
    dist[0] = 0
    
    counts = [0] * 5
    counts[0] = 1
    
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= 4:
            continue
        
        neighbors = [(u + s1) % n, (u - s1) % n, (u + s2) % n, (u - s2) % n]
        
        for v in neighbors:
            if visited[v] == -1:
                visited[v] = d + 1
                dist[v] = d + 1
                q.append(v)
                if d + 1 < 5:
                    counts[d + 1] += 1
            else:
                if dist[v] <= d:
                    cycle_len = d + 1 + dist[v]
                    if cycle_len < g:
                        return False
    
    for d in range(1, 5):
        if counts[d] < expected[d]:
            return False
            
    return True
    """
    Check if circulant graph C(n, {s1, s2}) has girth at least g.
    Girth < g iff there exists a non-backtracking walk of length < g that returns to start.
    Actually, girth is the length of the shortest cycle.
    We can check by BFS from 0 up to depth g//2 - 1.
    If we encounter a vertex at depth d that was already visited at depth d' < d, 
    and the path doesn't backtrack, we have a cycle.
    More simply: 
    A cycle of length L exists if there is a sequence of steps $x_1, ..., x_L \in \{s1, -s1, s2, -s2\}$
    such that sum $x_i = 0$ and the partial sums are distinct (except start/end) and no backtracking.
    
    Efficient check:
    For each vertex v, count the number of vertices at distance d.
    In a graph of girth >= g, the number of vertices at distance d from v is exactly
    1 + k * sum_{i=0}^{d-1} (k-1)^i for d < g/2.
    For k=4, g=10:
    d=0: 1
    d=1: 4
    d=2: 4 * 3 = 12
    d=3: 12 * 3 = 36
    d=4: 36 * 3 = 108
    Total vertices within distance 4: 1 + 4 + 12 + 36 + 108 = 161.
    If n < 161, impossible. (246 > 161, so ok).
    
    We can just run BFS from 0. If the BFS tree has any "cross edge" or "back edge" 
    that creates a cycle shorter than g, it fails.
    Actually, if the number of nodes at distance d is less than expected, a cycle exists.
    """
    # Expected counts
    expected = [1, 4, 12, 36, 108]
    
    # BFS from 0
    visited = [-1] * n
    dist = [0] * n
    q = deque()
    q.append(0)
    visited[0] = 0
    dist[0] = 0
    
    counts = [0] * 5
    counts[0] = 1
    
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= 4:
            continue
        
        # Neighbors: u+s1, u-s1, u+s2, u-s2
        neighbors = [(u + s1) % n, (u - s1) % n, (u + s2) % n, (u - s2) % n]
        
        for v in neighbors:
            if visited[v] == -1:
                visited[v] = d + 1
                dist[v] = d + 1
                q.append(v)
                if d + 1 < 5:
                    counts[d + 1] += 1
            else:
                # If visited, check if it creates a short cycle
                # If v is at distance d, it's a back edge to parent (ok, tree edge)
                # If v is at distance d+1, it's a cross edge to a node at same level? 
                # If dist[v] == d+1, then u-v is an edge between nodes at same distance?
                # No, u is at d, v is at d+1. This is a tree edge.
                # If dist[v] <= d, then we have a cycle.
                # Cycle length = dist[u] + 1 + dist[v]?
                # If dist[v] == d, cycle length is 2*d + 1 (odd cycle).
                # If dist[v] < d, cycle length is dist[u] + 1 + dist[v].
                if dist[v] <= d:
                    cycle_len = d + 1 + dist[v]
                    # We need to ensure this is a valid simple cycle.
                    # In BFS, an edge (u,v) where dist[v] <= dist[u] implies a cycle 
                    # of length dist[u] - dist[v] + 1? No.
                    # The cycle is path 0->...->v->...->u->v?
                    # Actually, if dist[v] == dist[u], cycle is even?
                    # Let's just check if the cycle length is < g.
                    if cycle_len < g:
                        return False
    
    # After BFS, check if counts match expected
    # If counts[d] < expected[d], it means some nodes were merged into shorter cycles
    for d in range(1, 5):
        if counts[d] < expected[d]:
            return False
            
    return True

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    start = (0, tuple([None] + [0] * d))

    def neighbours(vertex):
        side, x = vertex
        x = list(x)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
