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
    Uses adjacency lists for speed. Fast girth check via BFS depth counts.
    """
    target_sizes = [320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360]
    
    for n in target_sizes:
        if time.time() - t_start > _DEADLINE - 8:
            break
        
        # Generate random k-regular graph as adjacency list
        adj = _generate_random_k_regular_list(n, k)
        if adj is None:
            continue
        
        # Compute initial girth
        girth = _compute_girth_fast(adj, n, g)
        
        if girth >= g:
            edges = _adj_list_to_edges(adj, n)
            return n, edges
        
        # Simulated annealing with 2-switches
        best_girth = girth
        best_adj = [list(row) for row in adj]
        current_girth = girth
        
        T_start = 5.0
        T_end = 0.1
        max_iters = 20000
        iter_count = 0
        
        while iter_count < max_iters:
            if time.time() - t_start > _DEADLINE - 5:
                break
            iter_count += 1
            
            # Generate a random 2-switch
            switch = _generate_2_switch_list(adj, n, k)
            if switch is None:
                continue
            
            u, v, w, x, swap_type = switch
            
            if swap_type == 1:
                # Remove (u,v) and (w,x), add (u,w) and (v,x)
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                adj[u].append(w)
                adj[w].append(u)
                adj[v].append(x)
                adj[x].append(v)
            else:
                # Remove (u,v) and (w,x), add (u,x) and (v,w)
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                adj[u].append(x)
                adj[x].append(u)
                adj[v].append(w)
                adj[w].append(v)
            
            new_girth = _compute_girth_fast(adj, n, g)
            
            if new_girth >= g:
                edges = _adj_list_to_edges(adj, n)
                return n, edges
            
            delta = current_girth - new_girth
            if delta <= 0:
                current_girth = new_girth
                if new_girth > best_girth:
                    best_girth = new_girth
                    best_adj = [list(row) for row in adj]
            else:
                T = T_start * (T_end / T_start) ** (iter_count / max_iters)
                if random.random() < (2.718281828 ** (-delta / T)):
                    current_girth = new_girth
                else:
                    # Revert
                    if swap_type == 1:
                        adj[u].remove(w)
                        adj[w].remove(u)
                        adj[v].remove(x)
                        adj[x].remove(v)
                        adj[u].append(v)
                        adj[v].append(u)
                        adj[w].append(x)
                        adj[x].append(w)
                    else:
                        adj[u].remove(x)
                        adj[x].remove(u)
                        adj[v].remove(w)
                        adj[w].remove(v)
                        adj[u].append(v)
                        adj[v].append(u)
                        adj[w].append(x)
                        adj[x].append(w)
        
        if best_girth >= g:
            edges = _adj_list_to_edges(best_adj, n)
            return n, edges
    
    return None

def _generate_random_k_regular(n, k):
    """
    Generate a random k-regular graph using configuration model.
    Returns adjacency matrix (list of lists) or None if failed.
    """
    for attempt in range(20):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        adjacency = [[0] * n for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v or adjacency[u][v]:
                valid = False
                break
            adjacency[u][v] = 1
            adjacency[v][u] = 1
        
        if valid:
            return adjacency
    return None

def _generate_random_k_regular_list(n, k):
    """
    Generate a random k-regular graph using configuration model.
    Returns adjacency list (list of lists) or None if failed.
    """
    for attempt in range(50):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        edge_set = set()
        valid = True
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        
        if valid:
            return adj
    return None

def _compute_girth_fast(adj, n, g):
    """
    Compute the girth of a k-regular graph given as adjacency list.
    Uses BFS from each vertex: if any vertex has fewer neighbors at distance d
    than expected for a tree of depth d, a short cycle exists.
    For k-regular: expected at depth d is k*(k-1)^(d-1) for d>=1.
    Returns the girth (shortest cycle length), or g if girth >= g.
    """
    k = len(adj[0]) if n > 0 else 0
    max_depth = g // 2 - 1  # We only need to check up to depth g//2 - 1
    expected = [1] + [k * (k - 1) ** (d - 1) for d in range(1, max_depth + 1)]
    
    min_cycle = g  # Assume girth >= g initially
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        depth_counts = [0] * (max_depth + 1)
        depth_counts[0] = 1
        
        found_short_cycle = False
        while q:
            u = q.popleft()
            d = dist[u]
            if d >= max_depth:
                continue
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    depth_counts[d + 1] += 1
                    q.append(v)
                elif dist[v] <= d:
                    # Edge to same or earlier level: cycle
                    cycle_len = d + 1 + dist[v]
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if cycle_len < g:
                        found_short_cycle = True
        
        if found_short_cycle:
            return min_cycle
        
        # Check if depth counts match expected
        for d in range(1, max_depth + 1):
            if depth_counts[d] < expected[d]:
                # A short cycle exists
                return g - 1  # Girth is less than g
    
    return min_cycle

def _generate_2_switch_list(adj, n, k):
    """
    Generate a random valid 2-switch on an adjacency list.
    Returns (u, v, w, x, swap_type) or None.
    swap_type=1: remove (u,v),(w,x) add (u,w),(v,x)
    swap_type=2: remove (u,v),(w,x) add (u,x),(v,w)
    """
    # Build edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    if len(edges) < 2:
        return None
    
    # Build adjacency set once
    adj_set = set()
    for a in range(n):
        for b in adj[a]:
            adj_set.add((a, b))
    
    num_edges = len(edges)
    for _ in range(30):
        i = random.randrange(num_edges)
        j = random.randrange(num_edges)
        if i == j:
            continue
        u, v = edges[i]
        w, x = edges[j]
        
        # Switch type 1: (u,w) and (v,x)
        if u != w and v != x and (u, w) not in adj_set and (v, x) not in adj_set:
            return (u, v, w, x, 1)
        # Switch type 2: (u,x) and (v,w)
        if u != x and v != w and (u, x) not in adj_set and (v, w) not in adj_set:
            return (u, v, w, x, 2)
    
    return None

def _adjacency_to_edges(adjacency, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adjacency[u][v]:
                edges.append((u, v))
    return edges

def _adj_list_to_edges(adj, n):
    """Convert adjacency list to edge list."""
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _check_girth_circulant(n, s1, s2, g):
    """
    Check if circulant graph C(n, {s1, s2}) has girth at least g.
    For k=4, g=10: BFS from 0 up to depth 4.
    Expected counts: d=1:4, d=2:12, d=3:36, d=4:108.
    If any count is less than expected, or a cross-edge creates a short cycle, fail.
    """
    max_depth = g // 2 - 1  # 4
    expected = [1] + [4 * 3 ** (d - 1) for d in range(1, max_depth + 1)]
    
    dist = [-1] * n
    dist[0] = 0
    q = deque([0])
    depth_counts = [0] * (max_depth + 1)
    depth_counts[0] = 1
    
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= max_depth:
            continue
        
        neighbors = [(u + s1) % n, (u - s1) % n, (u + s2) % n, (u - s2) % n]
        
        for v in neighbors:
            if dist[v] == -1:
                dist[v] = d + 1
                depth_counts[d + 1] += 1
                q.append(v)
            else:
                if dist[v] <= d:
                    cycle_len = d + 1 + dist[v]
                    if cycle_len < g:
                        return False
    
    for d in range(1, max_depth + 1):
        if depth_counts[d] < expected[d]:
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
