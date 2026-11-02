import time
import sys

def luw_construct(k, d):
    """
    Constructs the Lazebnik-Ustimenko-Woldar graph CD(d, k).
    Returns (n, edges).
    """
    q = k
    # d must be at least 2
    if d < 2:
        d = 2
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Define the rule for coordinates
    # rule[c] = (a, b) such that y[c] depends on y[a] and x[b] (or vice versa)
    # Based on the seed program logic
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The list in seed: [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        # Index r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1? 
        # Seed: start = (0, (None,) + (0,) * d) -> length d+1.
        # Index 0 is None, indices 1..d are values.
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                    # y is line, x is point.
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: p[c] - l[c] = l[a] * p[b] ? 
                    # Seed: y[c] = sub(x[c], mul(x[a], y[b]))
                    # Here x is line, y is point.
                    # p[c] = l[c] + l[a] * p[b] ? No.
                    # The relation is symmetric in a specific way.
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def check_girth(n, edges, g):
    """
    Checks if the graph has girth >= g.
    Returns True if no cycle shorter than g exists.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # For each vertex, do a BFS up to depth g//2
    # We are looking for cycles of length < g.
    # Since it's bipartite (LUW is bipartite), we only need to check even lengths.
    # But to be general, we check all.
    
    # Optimization: If we find a cycle, return False immediately.
    # We use a simple DFS/BFS with parent tracking.
    # To avoid O(n * 2^g) complexity, we limit depth.
    # Max depth to check: (g-1)//2. 
    # If we find a neighbor at depth < (g-1)//2 that is already visited 
    # and is not the parent, we have a cycle.
    # Actually, standard girth check:
    # For each edge (u,v), check if there is a path from u to v of length < g-1.
    # This is expensive.
    
    # Better: For each vertex, expand non-backtracking walks.
    # If we hit a vertex already in the current walk's history, we have a cycle.
    # Depth limit: g//2.
    
    max_depth = g // 2
    if g % 2 == 0:
        # Bipartite graphs have even cycles.
        # If g=9, we check for cycles < 9, i.e., 4, 6, 8.
        # Max cycle length to find is 8.
        # Walk depth 4.
        pass
        
    for start in range(n):
        # BFS/DFS
        # State: (node, parent, depth)
        # We can use a set of visited nodes for this start?
        # No, we need to detect cycles back to the start or intermediate nodes.
        # Simple approach:
        # For each start, do a BFS up to depth max_depth.
        # If we encounter a node that was visited at a shallower depth in the SAME BFS tree?
        # No, that's just a tree.
        # We need to detect if a node is reached via two different paths from start?
        # That forms a cycle.
        
        # Efficient check:
        # For each start, maintain distance.
        # If we find an edge (u,v) such that dist[u] + 1 + dist[v] < g?
        # This requires all-pairs shortest paths, which is O(n^3).
        # n=1250, n^3 is too big.
        
        # Alternative:
        # Just do a DFS from each start, up to depth max_depth.
        # If we hit a node that is in the current stack, we have a cycle.
        # The length of the cycle is the difference in depths + 1.
        # If length < g, return False.
        
        stack = [(start, -1, 0)] # node, parent, depth
        # We need to track the path to calculate cycle length.
        # Since max_depth is small (4 or 5), we can store the path.
        
        while stack:
            node, parent, depth = stack.pop()
            if depth >= max_depth:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                # Check if nb is in the current path?
                # With stack, it's hard to know the current path easily.
                # Use recursion?
                pass
        
    # Given the low depth, a recursive DFS with path tracking is fine.
    def dfs(u, parent, depth, path):
        if depth >= max_depth:
            return True
        for v in adj[u]:
            if v == parent:
                continue
            if v in path:
                # Found a cycle
                # Cycle length is current depth - depth of v in path + 1
                # We need the index of v in path.
                # path is a list.
                idx = path.index(v)
                cycle_len = len(path) - idx
                if cycle_len < g:
                    return False
            else:
                path.append(v)
                if not dfs(v, u, depth + 1, path):
                    return False
                path.pop()
        return True

    for start in range(n):
        if not dfs(start, -1, 0, [start]):
            return False
            
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Determine candidate d values
    # We want smallest n, so smallest d.
    # girth >= g.
    # LUW girth >= d+4.
    # So d >= g-4.
    # Try d = g-4, g-3, ...
    
    candidates = []
    for d in range(g - 4, g + 5):
        if d >= 2:
            candidates.append(d)
            
    best_n = float('inf')
    best_edges = None
    best_d = None
    
    for d in candidates:
        if time.time() > deadline:
            break
            
        try:
            n, edges = luw_construct(k, d)
        except Exception:
            continue
            
        # Check if this graph is valid (girth >= g)
        # Note: LUW is bipartite.
        # If g is odd, e.g., 9, we need girth >= 9.
        # Since bipartite, girth is even. So girth >= 10.
        # So we need to ensure no cycles of length 4, 6, 8.
        
        # Optimization: If d is large enough, we trust the bound.
        # Bound: girth >= d+4.
        # If d+4 >= g, then for even g, we need d+4 >= g+1?
        # If g=9, d+4 >= 9 implies d>=5.
        # If d=5, girth >= 9. Since bipartite, girth >= 10.
        # So d=5 is guaranteed valid.
        # d=4 gives girth >= 8. Might be 8 or 10.
        
        if d + 4 >= g:
            # Check if we need to verify.
            # If d+4 >= g and g is even, d+4 >= g is sufficient?
            # If g=8, d+4>=8 -> d>=4. Girth>=8. Valid.
            # If g=9, d+4>=9 -> d>=5. Girth>=9 (actually 10). Valid.
            # So if d+4 >= g, it is likely valid.
            # However, the bound is "at least". It could be larger.
            # But it cannot be smaller.
            # So if d+4 >= g, the graph is valid.
            
            # Wait, is the bound strict?
            # "Girth is at least d+4".
            # So if d+4 >= g, then girth >= g.
            # So we don't need to check girth if d+4 >= g.
            
            if n < best_n:
                best_n = n
                best_edges = edges
                best_d = d
        else:
            # d+4 < g.
            # We must check if the actual girth is >= g.
            # This is only possible if the bound is loose.
            # For LUW, the bound is often tight.
            # So this case will likely fail.
            # But we check to be sure.
            if check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    best_d = d
                    
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback: return the largest d constructed (safest)
    # Or the first one.
    # The loop tries smallest d first.
    # If all failed, we return the last valid one?
    # Actually, the loop breaks on time.
    # If no valid graph found, we must return SOMETHING.
    # The largest d is most likely valid.
    # Let's just return the construction for the largest d tried.
    
    # Re-construct for the largest candidate
    max_d = candidates[-1]
    n, edges = luw_construct(k, max_d)
    return n, edges
