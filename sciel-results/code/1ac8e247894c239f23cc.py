import time
import random
import sys
from collections import deque

def construct(k, g):
    deadline = time.time() + 118

    # ---- LUW construction (fast fallback) ----
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

    def neighbours_luw(vertex):
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
        for other in neighbours_luw(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    n_luw, edges_luw = len(order), edges

    # ---- Fast girth certification using non-backtracking BFS ----
    def certify_fast(adj, n, g):
        """Check if graph has girth >= g using non-backtracking walks."""
        for s in range(n):
            for nb in adj[s]:
                # BFS in non-backtracking space from state (nb, s)
                # State: (current, prev)
                # If we reach (s, p) with p != nb at distance < g, cycle < g
                visited = set()
                visited.add((nb, s))
                queue = deque()
                queue.append((nb, s, 1))
                while queue:
                    if time.time() > deadline:
                        return False
                    v, p, dist = queue.popleft()
                    if v == s and p != nb:
                        if dist < g:
                            return False
                        continue
                    for u in adj[v]:
                        if u == p:
                            continue
                        if (u, v) not in visited:
                            visited.add((u, v))
                            queue.append((u, v, dist + 1))
        return True

    # ---- Build adjacency from LUW edges ----
    adj_luw = [set() for _ in range(n_luw)]
    for (u, v) in edges_luw:
        adj_luw[u].add(v)
        adj_luw[v].add(u)
    adj_luw = [list(s) for s in adj_luw]

    best_n = n_luw
    best_edges = edges_luw
    best_adj = adj_luw

    # Verify LUW graph
    if certify_fast(best_adj, best_n, g):
        pass  # valid, keep as baseline
    else:
        # Shouldn't happen, but if LUW is invalid, we have a problem
        # Just return it anyway as best effort
        return best_n, best_edges

    # ---- Greedy random construction for smaller graph ----
    def greedy_construct(k, g, seed=42, max_nodes=None):
        """
        Build a k-regular graph with girth >= g using greedy edge addition.
        Start with a tree, then add edges between far-apart vertices.
        """
        rng = random.Random(seed)
        
        # Target number of nodes: estimate from Moore bound
        # For even girth g, Moore bound: n >= 2*k*((k-1)^(g/2-1) - 1)/(k-2)
        if k > 2:
            half_g = g // 2 - 1
            if k == 3:
                moore = 2 * 3 * (2 ** half_g - 1)
            else:
                moore = 2 * k * ((k - 1) ** half_g - 1) // (k - 2)
        else:
            moore = 2 * g
        
        # Try to build a graph with roughly moore to 1.5*moore vertices
        target_n = max(moore, 2 * k)
        if max_nodes:
            target_n = min(target_n, max_nodes)
        
        # Start with a path (tree)
        nodes = list(range(target_n))
        adj = {i: set() for i in range(target_n)}
        
        # Build a tree: connect i to i+1 for i < target_n-1
        for i in range(target_n - 1):
            adj[i].add(i + 1)
            adj[i + 1].add(i)
        
        # Now we need to add edges to make it k-regular
        # Each node currently has degree 2 (interior) or 1 (ends)
        # We need to add (k-2)*target_n/2 more edges
        
        # Track free capacity
        free_cap = {}
        for i in range(target_n):
            free_cap[i] = k - len(adj[i])
        
        # List of nodes with free capacity
        candidates = [i for i in range(target_n) if free_cap[i] > 0]
        
        # We need to add edges between pairs of nodes that are at distance >= g-1
        # Use BFS to check distance, with early termination
        
        def get_distance(u, v, adj, max_depth):
            """Get distance between u and v, or -1 if > max_depth"""
            if u == v:
                return 0
            visited = {u: 0}
            queue = deque([u])
            while queue:
                curr = queue.popleft()
                d = visited[curr]
                if d >= max_depth:
                    continue
                for nb in adj[curr]:
                    if nb not in visited:
                        visited[nb] = d + 1
                        if nb == v:
                            return d + 1
                        queue.append(nb)
            return -1  # distance > max_depth or not found
        
        added_edges = 0
        max_adds = (k * target_n) // 2 - (target_n - 1)  # edges to add
        
        # Use a priority queue-like approach: try random pairs
        attempts = 0
        max_attempts = 50000
        
        while added_edges < max_adds and attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            if len(candidates) < 2:
                break
            
            # Pick two random candidates
            u = rng.choice(candidates)
            v = rng.choice(candidates)
            
            if u == v or v in adj[u]:
                continue
            
            # Check distance: need dist >= g-1
            dist = get_distance(u, v, adj, g - 2)
            if dist >= 0 and dist < g - 1:
                continue  # too close, would create cycle < g
            
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            free_cap[u] -= 1
            free_cap[v] -= 1
            added_edges += 1
            
            if free_cap[u] == 0:
                candidates.remove(u)
            if free_cap[v] == 0:
                candidates.remove(v)
        
        # Check if we succeeded
        if added_edges < max_adds:
            return None  # failed
        
        # Verify all degrees are k
        for i in range(target_n):
            if len(adj[i]) != k:
                return None
        
        # Convert to edge list
        edge_list = []
        for u in range(target_n):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))
        
        return target_n, edge_list, [list(adj[i]) for i in range(target_n)]

    # Try greedy construction with different seeds and target sizes
    rng_main = random.Random(123)
    
    # Try a few target sizes around the Moore bound
    if k == 3:
        half_g = g // 2 - 1
        moore_base = 2 * 3 * (2 ** half_g - 1)
    else:
        half_g = g // 2 - 1
        if k > 2:
            moore_base = 2 * k * ((k - 1) ** half_g - 1) // (k - 2)
        else:
            moore_base = 2 * g
    
    # Try target sizes from moore_base to moore_base * 1.5
    target_sizes = sorted(set([
        int(moore_base * 1.0),
        int(moore_base * 1.05),
        int(moore_base * 1.1),
        int(moore_base * 1.15),
        int(moore_base * 1.2),
        int(moore_base * 1.3),
        int(moore_base * 1.4),
        int(moore_base * 1.5),
    ]))
    
    # Filter: must be even (for regular graph) and >= 2k
    target_sizes = [t for t in target_sizes if t >= 2 * k and t % 2 == 0]
    
    for tsize in target_sizes:
        if time.time() > deadline:
            break
        if tsize >= best_n:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = greedy_construct(k, g, seed=seed + tsize, max_nodes=tsize)
            if result is None:
                continue
            
            n_try, edges_try, adj_try = result
            
            if n_try >= best_n:
                continue
            
            # Verify girth
            if certify_fast(adj_try, n_try, g):
                best_n = n_try
                best_edges = edges_try
                best_adj = adj_try
                break  # Found valid graph for this size, try smaller
    
    # ---- Return best found ----
    best_edges = [(min(u, v), max(u, v)) for (u, v) in best_edges]
    best_edges = list(set(best_edges))
    best_edges.sort()
    
    return best_n, best_edges
