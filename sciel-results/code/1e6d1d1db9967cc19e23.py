"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque, defaultdict

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Fallback: LUW construction
    luw_result = _luw_construct(k, g)
    best_n, best_edges = luw_result
    
    # Try to find a smaller graph using random search + local improvement
    # Moore bound for k=4, g=10 is 485.
    # Try sizes from 500 upwards.
    # If we find a valid graph with n < best_n, update best.
    
    # We only attempt this if we have enough time and k, g match our optimized search
    if k == 4 and g == 10:
        for n_attempt in range(500, 1200, 20):
            if time.time() > deadline - 10:
                break
            
            # Generate random 4-regular graph
            edges = _gen_random_4reg(n_attempt)
            if edges is None:
                continue
                
            # Local search to increase girth
            edges = _local_search_girth(n_attempt, edges, g, deadline)
            
            # Check if girth is at least g
            if _has_girth_at_least(n_attempt, edges, g):
                if n_attempt < best_n:
                    best_n = n_attempt
                    best_edges = edges
                    # We found a smaller one, but maybe we can find even smaller?
                    # Break to save time or continue? 
                    # Given the difficulty, finding one is a big win.
                    # Let's try to find slightly smaller ones if time permits.
                    # But 500 is very close to bound.
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _gen_random_4reg(n):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4 or n * 4 % 2 != 0:
        return None
    
    # Use configuration model
    for _ in range(50):
        stubs = []
        for v in range(n):
            stubs.extend([v] * 4)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            e = (min(u, v), max(u, v))
            if e in edges:
                valid = False
                break
            edges.add(e)
        
        if valid:
            return list(edges)
    return None

def _has_girth_at_least(n, edges, g):
    """Check if graph has girth at least g."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        # BFS
        dist = {start: 0}
        parent = {start: -1}
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    # But we need to be careful with BFS tree properties
                    # Simpler: if we find a back edge to a node with dist < g-1, it's a short cycle
                    # Actually, just checking if any cycle < g exists.
                    # A cycle is formed if we encounter a visited node v != parent[u].
                    # The cycle length is dist[u] + dist[v] + 1.
                    # If this length < g, return False.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _local_search_girth(n, edges, g, deadline):
    """Try to increase girth by swapping edges."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    steps = 0
    max_steps = 5000
    while steps < max_steps and time.time() < deadline - 5:
        steps += 1
        # Find a short cycle
        short_cycle = _find_short_cycle(n, adj, g)
        if short_cycle is None:
            break
        
        # Pick a random edge from the cycle to break
        # Cycle is a list of vertices [v0, v1, ..., vk-1]
        # Edges are (vi, vi+1)
        idx = random.randint(0, len(short_cycle) - 1)
        u = short_cycle[idx]
        v = short_cycle[(idx + 1) % len(short_cycle)]
        
        # Try to swap edge (u, v) with another edge (a, b)
        # such that a is neighbor of u (not v) and b is neighbor of v (not u)
        # New edges: (u, b), (v, a)
        # Must ensure (u, b) and (v, a) are not existing edges
        
        neighbors_u = list(adj[u])
        neighbors_v = list(adj[v])
        
        # Filter neighbors
        # a must be in neighbors_u, a != v
        # b must be in neighbors_v, b != u
        # Also (u, b) not in edge_set, (v, a) not in edge_set
        
        found_swap = False
        random.shuffle(neighbors_u)
        random.shuffle(neighbors_v)
        
        for a in neighbors_u:
            if a == v:
                continue
            for b in neighbors_v:
                if b == u:
                    continue
                # Check if new edges are valid
                e1 = (min(u, b), max(u, b))
                e2 = (min(v, a), max(v, a))
                
                # If e1 or e2 is the edge we are removing, skip?
                # e1 is (u, b). We are removing (u, v).
                # If b == v, we skipped.
                # If e1 is already in edge_set (and not (u, v)), invalid.
                if e1 in edge_set and e1 != (min(u, v), max(u, v)):
                    continue
                if e2 in edge_set and e2 != (min(u, v), max(u, v)):
                    continue
                
                # Perform swap
                edge_set.remove((min(u, v), max(u, v)))
                edge_set.add(e1)
                edge_set.add(e2)
                
                adj[u].remove(v)
                adj[v].remove(u)
                adj[u].add(b)
                adj[b].add(u)
                adj[v].add(a)
                adj[a].add(v)
                
                found_swap = True
                break
            if found_swap:
                break
        
        if not found_swap:
            # Try another edge in the cycle or give up
            pass
            
    return list(edge_set)

def _find_short_cycle(n, adj, g):
    """Find a cycle of length < g. Returns list of vertices or None."""
    for start in range(n):
        dist = {start: 0}
        parent = {start: -1}
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Cycle found
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        # Path from u to LCA and v to LCA
                        # Simple reconstruction:
                        path_u = []
                        curr = u
                        while curr != -1:
                            path_u.append(curr)
                            curr = parent[curr]
                        path_v = []
                        curr = v
                        while curr != -1:
                            path_v.append(curr)
                            curr = parent[curr]
                        
                        # Find LCA
                        set_u = set(path_u)
                        lca = -1
                        for node in path_v:
                            if node in set_u:
                                lca = node
                                break
                                
                        if lca == -1:
                            continue
                            
                        cycle = []
                        curr = u
                        while curr != lca:
                            cycle.append(curr)
                            curr = parent[curr]
                        cycle.append(lca)
                        
                        curr = v
                        while curr != lca:
                            cycle.append(curr)
                            curr = parent[curr]
                            
                        return cycle
    return None
# EVOLVE-BLOCK-END
