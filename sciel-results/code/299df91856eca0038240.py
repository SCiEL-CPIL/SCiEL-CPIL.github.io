import time
import random
import sys

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0

    if k != 3 or g != 16:
        return _luw_construct(k, g)

    # Voltage graph (lift) construction
    # Base graph: a 3-regular graph on b vertices
    # Cover: vertices are (v, g) for v in V(B), g in G where |G| = m
    # Total vertices n = b * m
    # Edge labels in G determine the cover structure
    
    # Strategy: use a small base graph (e.g., 4-vertex K4 or 6-vertex prism)
    # with edge labels in Z_m, and search for labels that maximize girth
    
    best_n = float('inf')
    best_edges = None
    
    # Try different base graph sizes and group sizes
    # n = b * m, we want n >= 936 (known best)
    # Small b means larger m, which gives more label freedom
    
    # Base graph options for 3-regular:
    # b=4: K4 (complete graph on 4 vertices)
    # b=6: triangular prism
    # b=8: cube
    # b=10: Petersen graph
    
    for b in [4, 6, 8, 10]:
        if time.time() > deadline - 20:
            break
            
        base_edges = _get_base_3reg(b)
        if base_edges is None:
            continue
            
        # For each base graph, try different group sizes m
        # n = b * m, aim for n around 936-1500
        for m in [234, 240, 250, 300, 312, 333, 400, 500]:
            if time.time() > deadline - 15:
                break
                
            n = b * m
            if n < 936:
                continue
            if n > 2000:
                continue
                
            # Search for good edge labels
            result = _search_voltage_labels(base_edges, b, m, g, deadline)
            if result is not None:
                n_found, edges_found = result
                if n_found < best_n:
                    best_n = n_found
                    best_edges = edges_found
                    if best_n <= 1000:
                        break  # Good enough, try smaller b next
        if best_n <= 1000:
            break
    
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to LUW
    return _luw_construct(k, g)

def _get_base_3reg(b):
    """Return a 3-regular graph on b vertices as list of edges, or None."""
    if b == 4:
        # K4
        return [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    elif b == 6:
        # Triangular prism: two triangles connected by matching
        return [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3), (1,4), (2,5)]
    elif b == 8:
        # Cube graph
        return [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7),
                (4,5), (4,6), (5,7), (6,7)]
    elif b == 10:
        # Petersen graph
        outer = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        inner = [(5,7), (7,9), (9,6), (6,8), (8,5)]
        spokes = [(0,5), (1,6), (2,7), (3,8), (4,9)]
        return outer + inner + spokes
    return None

def _search_voltage_labels(base_edges, b, m, target_girth, deadline):
    """
    Search for edge labels in Z_m that give a cover with girth >= target_girth.
    Returns (n, edges) or None.
    """
    num_edges = len(base_edges)
    
    # Initial random labels
    best_labels = [random.randint(0, m-1) for _ in range(num_edges)]
    best_girth = 0
    
    # Simulated annealing
    temperature = 1.0
    cooling = 0.995
    min_temp = 0.01
    
    time_limit = deadline - 5.0
    
    while time.time() < time_limit:
        # Compute girth of the cover
        girth, short_cycle = _compute_cover_girth(base_edges, best_labels, b, m, target_girth)
        
        if girth > best_girth:
            best_girth = girth
            if girth >= target_girth:
                # Found valid graph!
                edges = _build_cover_edges(base_edges, best_labels, b, m)
                n = b * m
                return n, edges
        
        if temperature < min_temp:
            break
            
        # Propose a change: modify one random edge label
        idx = random.randint(0, num_edges - 1)
        old_label = best_labels[idx]
        new_label = random.randint(0, m - 1)
        
        if new_label == old_label:
            continue
            
        best_labels[idx] = new_label
        
        # Compute new girth
        new_girth, _ = _compute_cover_girth(base_edges, best_labels, b, m, target_girth)
        
        # Accept or reject
        if new_girth >= best_girth:
            # Always accept improvement
            best_girth = new_girth
        else:
            # Accept with probability based on temperature
            delta = best_girth - new_girth
            prob = 1.0 / (1.0 + delta * temperature * 0.1)
            if random.random() > prob:
                # Reject
                best_labels[idx] = old_label
        
        temperature *= cooling
    
    # Try a few more random restarts if time permits
    remaining_time = time_limit - time.time()
    if remaining_time > 10:
        for _ in range(3):
            if time.time() > time_limit:
                break
            best_labels = [random.randint(0, m-1) for _ in range(num_edges)]
            girth, _ = _compute_cover_girth(base_edges, best_labels, b, m, target_girth)
            if girth >= target_girth:
                edges = _build_cover_edges(base_edges, best_labels, b, m)
                n = b * m
                return n, edges
    
    return None

def _compute_cover_girth(base_edges, labels, b, m, max_check):
    """
    Compute girth of the voltage cover.
    Vertices are (v, g) where v in [0,b), g in [0,m).
    An edge (u,v) in base with label l lifts to edges (u,g) -> (v, g+l mod m).
    """
    n = b * m
    # Build adjacency for the cover
    # For speed, we'll do BFS on the implicit cover
    
    # We'll represent vertex (v, g) as v * m + g
    adj = [[] for _ in range(n)]
    for (u, v), label in zip(base_edges, labels):
        for g in range(m):
            w = (g + label) % m
            adj[u * m + g].append(v * m + w)
            adj[v * m + w].append(u * m + g)
    
    # BFS from each vertex to find shortest cycle
    # For efficiency, only check up to max_check/2 depth
    max_depth = max_check // 2
    min_cycle = max_check
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        
        while queue:
            u = queue.pop(0)
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle <= 3:
                            return min_cycle, None
        
        if min_cycle <= 3:
            break
    
    return min_cycle, None

def _build_cover_edges(base_edges, labels, b, m):
    """Build the full edge list for the voltage cover."""
    edges = []
    seen = set()
    for (u, v), label in zip(base_edges, labels):
        for g in range(m):
            w = (g + label) % m
            a = u * m + g
            c = v * m + w
            if a < c:
                edges.append((a, c))
            else:
                edges.append((c, a))
    return edges

def _graph_to_edges(adj, n):
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return n, edges

def _generate_random_3_reg(n):
    if n % 2 != 0:
        n += 1
        
    adj = [set() for _ in range(n)]
    half_edges = []
    for i in range(n):
        for _ in range(3):
            half_edges.append(i)
            
    random.shuffle(half_edges)
    
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
        
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    return adj

def _improve_girth(adj, n, target_girth, deadline):
    girth = _calc_girth(adj, n, target_girth)
    
    while time.time() < deadline - 5.0:
        if girth >= target_girth:
            break
            
        cycle = _find_short_cycle(adj, n, target_girth)
        if not cycle:
            girth = _calc_girth(adj, n, target_girth)
            if girth >= target_girth:
                break
            else:
                break
        
        success = False
        for _ in range(100):
            e_idx = random.randint(0, len(cycle)-1)
            u = cycle[e_idx]
            v = cycle[(e_idx+1) % len(cycle)]
            
            while True:
                w = random.randint(0, n-1)
                if w == u or w == v:
                    continue
                x = random.choice(list(adj[w]))
                if x == u or x == v:
                    continue
                if w in adj[u] or x in adj[v]:
                    continue
                
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                
                adj[u].add(w)
                adj[w].add(u)
                adj[v].add(x)
                adj[x].add(v)
                
                success = True
                break
            
        if not success:
            continue
            
        girth = _calc_girth(adj, n, target_girth)
        
    return adj, girth

def _calc_girth(adj, n, max_check):
    max_depth = max_check // 2
    min_cycle = max_check
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        
        while queue:
            u = queue.pop(0)
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        
        if min_cycle == 3:
            break
            
    return min_cycle

def _find_short_cycle(adj, n, max_check):
    max_depth = max_check // 2
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        
        while queue:
            u = queue.pop(0)
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < max_check:
                        path1 = [u]
                        cur = u
                        while cur != s:
                            cur = parent[cur]
                            path1.append(cur)
                        
                        path2 = [v]
                        cur = v
                        while cur != s:
                            cur = parent[cur]
                            path2.append(cur)
                        
                        p2_rev = path2[::-1]
                        cycle = path1 + p2_rev[1:]
                        return cycle
                        
    return None

def _luw_construct(k, g):
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
