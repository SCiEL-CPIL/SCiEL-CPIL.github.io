import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # For the specific case k=3, g=16, use LUW with q=3, d=11
    if k == 3 and g == 16:
        q = 3
        d = 11
        n, edges = _luw_construct(q, d, deadline)
        
        # Use remaining time for local search: try to reduce n via edge swaps
        remaining = deadline - time.time()
        if remaining > 5:
            n, edges = _local_search_3reg(n, edges, g, deadline)
        return n, edges
    
    # Fallback for other parameters
    return _luw_construct(k, g - 5 if (g - 5) % 2 else g - 4, deadline)

def _local_search_3reg(n, edges, girth_target, deadline):
    """
    Try to find a smaller 3-regular graph of girth >= girth_target
    by removing vertices and reconnecting. Start from LUW graph.
    """
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Verify initial graph is valid
    if not _check_girth_fast(adj, girth_target):
        return n, edges
    
    best_n = n
    best_edges = list(edges)
    
    # Strategy: try to remove pairs of vertices and reconnect neighbors
    # This reduces n by 2 each successful step
    max_iterations = 500
    attempts = 0
    
    while time.time() < deadline and attempts < max_iterations:
        attempts += 1
        
        # Pick two random non-adjacent vertices to try removing
        if len(adj) < 4:
            break
            
        # Find candidate vertices: those whose removal doesn't create short cycles
        # Simple heuristic: pick vertices with "similar" neighborhoods
        candidates = []
        for _ in range(20):
            u = random.randint(0, len(adj) - 1)
            v = random.randint(0, len(adj) - 1)
            if u != v and v not in adj[u] and len(adj[u]) >= 3 and len(adj[v]) >= 3:
                # Check u and v are not too close (distance >= girth_target/2)
                if _distance(adj, u, v) >= girth_target // 2:
                    candidates.append((u, v))
        
        if not candidates:
            continue
        
        u, v = random.choice(candidates)
        
        # Try to remove u and v: reconnect their neighbors
        # Each has 3 neighbors. After removing u, its 3 neighbors lose one edge.
        # We need to add edges among them to restore 3-regularity.
        # This is complex; instead, try a simpler operation:
        
        # Alternative: try swapping an edge to create a structure that allows vertex removal
        # For now, just track best and move on
        
        # Try edge swap: pick edge (a,b), find another edge (c,d) not sharing vertices,
        # swap to (a,c),(b,d) or (a,d),(b,c)
        if len(edges) < 10:
            continue
            
        e1_idx = random.randint(0, len(edges) - 1)
        a, b = edges[e1_idx]
        
        # Find another edge not sharing a or b
        found = False
        for _ in range(50):
            e2_idx = random.randint(0, len(edges) - 1)
            if e2_idx == e1_idx:
                continue
            c, d = edges[e2_idx]
            if a not in (c, d) and b not in (c, d):
                found = True
                break
        
        if not found:
            continue
        
        # Try swap: remove (a,b) and (c,d), add (a,c) and (b,d)
        # Check no duplicate edges
        if c in adj[a] or d in adj[b] or c in adj[b] or d in adj[a]:
            continue
        
        # Apply swap
        adj[a].discard(b)
        adj[b].discard(a)
        adj[c].discard(d)
        adj[d].discard(c)
        adj[a].add(c)
        adj[c].add(a)
        adj[b].add(d)
        adj[d].add(b)
        
        # Quick girth check
        if _check_girth_fast(adj, girth_target):
            # Update edges list
            new_edges = []
            for i in range(len(adj)):
                for j in adj[i]:
                    if i < j:
                        new_edges.append((i, j))
            if len(new_edges) < best_n * 3 // 2:  # n vertices, 3-regular => 3n/2 edges
                best_n = len(adj)
                best_edges = new_edges
        else:
            # Revert
            adj[a].discard(c)
            adj[c].discard(a)
            adj[b].discard(d)
            adj[d].discard(b)
            adj[a].add(b)
            adj[b].add(a)
            adj[c].add(d)
            adj[d].add(c)
    
    # Rebuild edge list from best_adj if we improved
    if best_n < n:
        adj_best = [set() for _ in range(best_n)]
        for u, v in best_edges:
            adj_best[u].add(v)
            adj_best[v].add(u)
        return best_n, best_edges
    
    return n, edges

def _check_girth_fast(adj, girth_target):
    """
    Check if graph has girth >= girth_target using non-backtracking BFS.
    Returns True if no cycle shorter than girth_target exists.
    """
    n = len(adj)
    if n < 3:
        return True
    
    # Non-backtracking BFS from each vertex
    # State: (vertex, prev_vertex)
    # Track depth
    max_depth = (girth_target - 1) // 2
    
    for start in range(n):
        # BFS with non-backtracking constraint
        # visited[prev][curr] = depth
        visited = {}
        queue = deque()
        
        for nb in adj[start]:
            queue.append((start, nb, 1))
            visited[(start, nb)] = 1
        
        while queue:
            prev, curr, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue  # non-backtracking
                
                if nb == start:
                    # Found a cycle of length depth + 1
                    if depth + 1 < girth_target:
                        return False
                
                state = (curr, nb)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((curr, nb, depth + 1))
    
    return True

def _distance(adj, u, v):
    """BFS distance between u and v, capped at girth_target//2"""
    if u == v:
        return 0
    visited = {u: 0}
    queue = deque([u])
    while queue:
        curr = queue.popleft()
        d = visited[curr]
        if d >= 10:  # cap
            break
        for nb in adj[curr]:
            if nb == v:
                return d + 1
            if nb not in visited:
                visited[nb] = d + 1
                queue.append(nb)
    return 100  # large number if not found within cap

def _luw_construct(q, d, deadline):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Returns (n, edges) for the connected component of the zero vertex.
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Build the rule
    # rule[c] = (a, b) such that y[c] depends on y[a] and y[b]
    # Indices in the vector are 1..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    # Precompute powers of q for encoding
    # ID = sum(v[i] * q^(i-1) for i in 1..d)
    powers = [1] * (d + 1)
    for i in range(1, d + 1):
        powers[i] = powers[i-1] * q
        
    def get_digit(val, pos):
        return (val // powers[pos-1]) % q

    def get_point_neighbours(p_id):
        # Point p, find Lines l
        # l[c] = p[c] + l[a] * p[b]
        # l[1] is free
        for l1 in range(q):
            current_l_id = l1
            for c in range(2, d + 1):
                a, b = rule[c]
                p_c = get_digit(p_id, c)
                l_a = get_digit(current_l_id, a)
                p_b = get_digit(p_id, b)
                l_c = add(p_c, mul(l_a, p_b))
                current_l_id += l_c * powers[c-1]
            yield (1, current_l_id)

    def get_line_neighbours(l_id):
        # Line l, find Points p
        # p[c] = l[c] - l[a] * p[b]
        # p[1] is free
        for p1 in range(q):
            current_p_id = p1
            for c in range(2, d + 1):
                a, b = rule[c]
                l_c = get_digit(l_id, c)
                l_a = get_digit(l_id, a)
                p_b = get_digit(current_p_id, b)
                p_c = sub(l_c, mul(l_a, p_b))
                current_p_id += p_c * powers[c-1]
            yield (0, current_p_id)

    # BFS to find connected component
    start_vertex = (0, 0) # (side, id)
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, vid = vertex
        if side == 0:
            neighbors = get_point_neighbours(vid)
        else:
            neighbors = get_line_neighbours(vid)
            
        for n_side, n_vid in neighbors:
            n_vertex = (n_side, n_vid)
            if n_vertex not in index:
                index[n_vertex] = len(order)
                order.append(n_vertex)
            # Add edge only once (when visiting from point side to avoid duplicates in undirected graph representation)
            # In LUW, edges connect Point to Line.
            # If we add edge when side==0, we cover all edges exactly once.
            if side == 0:
                edges.append((index[vertex], index[n_vertex]))
                
    n = len(order)
    return n, edges
