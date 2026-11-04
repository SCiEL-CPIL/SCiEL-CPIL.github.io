import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    if k != 4:
        # Fallback for other k: use LUW
        return luw_construct(k, g)
    
    # Target: find smallest n with a 4-regular graph of girth >= g
    # Known best: 320 for g=10. Moore bound is 243.
    
    # Try n from 320 upward
    for n in range(320, 360):
        if time.time() > deadline:
            break
        
        # Multiple restarts for this n
        max_restarts = 3 if n == 320 else 2
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            result = local_search_4_reg(n, g, deadline)
            if result is not None:
                return result
    
    # If we didn't find one, fall back to LUW
    return luw_construct(4, g)

def luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
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
        
    n = len(order)
    return n, edges

def generate_random_4_regular(n):
    """Generate a random simple 4-regular graph on n vertices."""
    for attempt in range(200):
        stubs = []
        for i in range(n):
            stubs.extend([i] * 4)
        
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        
        if valid:
            return list(edges)
    
    return None

def count_short_cycles(adj, n, g_limit):
    """Count number of cycles of length < g_limit using non-backtracking BFS.
    Returns (count, min_cycle_len) where count is total short cycles found,
    min_cycle_len is the shortest cycle found (or g_limit if none)."""
    total_count = 0
    min_len = g_limit
    
    for start in range(n):
        # BFS with non-backtracking: track (node, prev) pairs
        # dist[(u,v)] = shortest non-backtracking path from (start, -1) to (u, v)
        # Actually, let's use a simpler approach:
        # For each start, do BFS but don't go back to the node we came from
        
        visited = {}  # (node, prev) -> dist
        queue = deque()
        
        # Start: from start, we can go to any neighbor
        for v in adj[start]:
            visited[(v, start)] = 1
            queue.append((v, start))
        
        while queue:
            u, prev = queue.popleft()
            dist_u = visited[(u, prev)]
            
            for v in adj[u]:
                if v == prev:
                    continue  # Non-backtracking
                
                if (v, u) in visited:
                    # Found a cycle: dist_u + 1 + (distance from start to v via non-backtracking)
                    # Actually, the cycle length is dist_u + 1
                    # Wait, that's not right. Let me think again.
                    # If we're at (u, prev) with distance d, and v is a neighbor of u (v != prev),
                    # and (v, u) has been visited at distance d2,
                    # then we have a cycle of length d + 1 + d2... no.
                    # The path from start to (u,prev) has length d.
                    # The edge (u,v) adds 1.
                    # The path from start to (v,u) has length d2.
                    # But (v,u) means we arrived at v from u, so the path is start->...->u->v.
                    # So the cycle is: the path to (u,prev) [length d], then edge u-v [length 1],
                    # then... we need to get back to start.
                    # This approach is getting complicated. Let me use a different method.
                    pass
                else:
                    visited[(v, u)] = dist_u + 1
                    queue.append((v, u))
    
    # Let me use a cleaner approach: for each start vertex, do BFS tracking parent
    # and detect cycles properly.
    return total_count, min_len

def girth_check(adj, n, g_limit):
    """Check if graph has girth >= g_limit.
    Returns True if girth >= g_limit, False otherwise.
    Uses non-backtracking walk detection from each vertex."""
    for start in range(n):
        # BFS from start, tracking parent to avoid backtracking
        # dist[v] = shortest distance from start
        # parent[v] = the node we came from to reach v
        # When we find an edge (u,v) where v is already visited and v != parent[u] and u != parent[v],
        # we have a cycle of length dist[u] + dist[v] + 1
        
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # v is already visited
                    # Check if this forms a cycle (not just going back to parent)
                    if v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g_limit:
                            return False
    
    return True

def local_search_4_reg(n, g, deadline):
    """Local search to find a 4-regular graph on n vertices with girth >= g."""
    
    # Generate initial random graph
    edges = generate_random_4_regular(n)
    if edges is None:
        return None
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    adj_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        key = (min(u, v), max(u, v))
        adj_set.add(key)
    
    # Check initial girth
    if girth_check(adj, n, g):
        return n, edges
    
    # Local search: edge swaps
    # Pick two random edges (u,v) and (x,y), replace with (u,x) and (v,y) or (u,y) and (v,x)
    
    best_girth = g - 1  # We'll track the best we've seen
    edge_list = list(edges)
    
    num_steps = 5000
    
    for step in range(num_steps):
        if time.time() > deadline:
            break
        
        # Pick two random edges
        e1_idx = random.randint(0, len(edge_list) - 1)
        e2_idx = random.randint(0, len(edge_list) - 1)
        if e1_idx == e2_idx:
            continue
        
        u, v = edge_list[e1_idx]
        x, y = edge_list[e2_idx]
        
        # Try both swap patterns
        swaps = [
            ((u, x), (v, y)),
            ((u, y), (v, x)),
        ]
        
        for (a, b), (c, d) in swaps:
            # Check for loops
            if a == b or c == d:
                continue
            
            # Check if edges already exist
            key1 = (min(a, b), max(a, b))
            key2 = (min(c, d), max(c, d))
            
            if key1 in adj_set or key2 in adj_set:
                continue
            
            # Perform swap
            # Remove old edges
            old_key1 = (min(u, v), max(u, v))
            old_key2 = (min(x, y), max(x, y))
            
            adj_set.remove(old_key1)
            adj_set.remove(old_key2)
            
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            # Add new edges
            adj_set.add(key1)
            adj_set.add(key2)
            adj[a].append(b)
            adj[b].append(a)
            adj[c].append(d)
            adj[d].append(c)
            
            # Update edge list
            edge_list[e1_idx] = (min(a, b), max(a, b))
            edge_list[e2_idx] = (min(c, d), max(c, d))
            
            # Check girth
            if girth_check(adj, n, g):
                return n, list(adj_set)
            
            # Revert (we could keep if girth improved, but for simplicity, revert)
            # Actually, let's keep the swap only if it doesn't make things worse
            # For now, always revert unless it's a success
            
            # Revert
            adj_set.remove(key1)
            adj_set.remove(key2)
            
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            
            adj_set.add(old_key1)
            adj_set.add(old_key2)
            
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            
            edge_list[e1_idx] = (min(u, v), max(u, v))
            edge_list[e2_idx] = (min(x, y), max(x, y))
    
    return None
