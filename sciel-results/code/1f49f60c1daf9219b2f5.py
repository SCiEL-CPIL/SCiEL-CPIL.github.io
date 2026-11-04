import time
import random
import math
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        if k == 4 and g == 10:
            # The LUW construction for k=4, g=10 is known to be the 320-vertex graph.
            # We can construct it directly or rely on the general LUW code.
            # The general LUW code below should produce the correct graph.
            pass
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

    if k == 4 and g == 10:
        deadline = time.time() + 110  # Safety margin
        n_target = 320
        
        def random_4_regular(n):
            if n % 2 != 0:
                return None
            
            def random_perfect_matching(n):
                if n % 2 != 0: return None
                items = list(range(n))
                random.shuffle(items)
                match = {}
                for i in range(0, n, 2):
                    u, v = items[i], items[i+1]
                    match[u] = v
                    match[v] = u
                return match

            m1 = random_perfect_matching(n)
            if m1 is None: return None
            m2 = random_perfect_matching(n)
            if m2 is None: return None
            
            adj = [[] for _ in range(n)]
            for u in range(n):
                v1 = m1[u]
                v2 = m2[u]
                if v1 == v2:
                    return None
                adj[u].append(v1)
                adj[v1].append(u)
                adj[u].append(v2)
                adj[v2].append(u)
                
            for u in range(n):
                if len(set(adj[u])) < 4:
                    return None
            return adj

        def has_path_shorter_than(u, v, max_len, adj, forbidden_edges=None):
            """Check if there is a path from u to v of length < max_len, avoiding forbidden_edges"""
            if u == v:
                return 0 < max_len
            visited = {u: 0}
            q = deque([u])
            while q:
                node = q.popleft()
                dist = visited[node]
                if dist + 1 >= max_len:
                    continue
                for neighbor in adj[node]:
                    if forbidden_edges and (min(node, neighbor), max(node, neighbor)) in forbidden_edges:
                        continue
                    if neighbor == v:
                        return dist + 1 < max_len
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        q.append(neighbor)
            return False

        def get_girth_and_cycle(adj):
            n = len(adj)
            best_girth = float('inf')
            best_cycle = None
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] * 2 >= best_girth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif dist[v] < dist[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < best_girth:
                                best_girth = cycle_len
                                # Reconstruct cycle
                                path = []
                                cur = u
                                while cur != v:
                                    path.append(cur)
                                    cur = parent[cur]
                                path.append(v)
                                best_cycle = path
            return best_girth, best_cycle

        def try_improve(adj, current_girth, deadline):
            n = len(adj)
            # Set of edges for quick lookup
            edge_set = set()
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edge_set.add((u, v))
            
            while time.time() < deadline:
                if current_girth >= g:
                    return adj, True
                
                # Find a short cycle
                girth, cycle = get_girth_and_cycle(adj)
                if girth >= g:
                    return adj, True
                if not cycle:
                    break
                
                # Pick a random edge from the cycle to swap
                improved = False
                cycle_len = len(cycle)
                
                # Randomize order of cycle edges
                edges_in_cycle = [(cycle[i], cycle[(i + 1) % cycle_len]) for i in range(cycle_len)]
                random.shuffle(edges_in_cycle)
                
                for u, v in edges_in_cycle:
                    # Sample candidates instead of iterating all
                    # We want to find a swap that breaks the cycle
                    # Try a few random edges
                    attempts = 0
                    while attempts < 50 and not improved:
                        attempts += 1
                        c = random.randint(0, n - 1)
                        if c == u or c == v:
                            continue
                        if not adj[c]:
                            continue
                        d = random.choice(adj[c])
                        if c > d:
                            c, d = d, c
                        if (c, d) == (u, v) or (c, d) == (v, u):
                            continue
                        if d in adj[u] or c in adj[v]:
                            continue
                            
                        forbidden = {(min(u,v), max(u,v)), (min(c,d), max(c,d))}
                        
                        if has_path_shorter_than(d, u, current_girth, adj, forbidden):
                            continue
                        if has_path_shorter_than(c, v, current_girth, adj, forbidden):
                            continue
                            
                        # Perform swap
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[c].remove(d)
                        adj[d].remove(c)
                        
                        adj[u].append(d)
                        adj[d].append(u)
                        adj[v].append(c)
                        adj[c].append(v)
                        
                        edge_set.discard((min(u,v), max(u,v)))
                        edge_set.discard((min(c,d), max(c,d)))
                        edge_set.add((min(u,d), max(u,d)))
                        edge_set.add((min(v,c), max(v,c)))
                        
                        improved = True
                        current_girth = girth # Update girth estimate? No, we need to recompute.
                        
                    if improved:
                        break
                        
                if not improved:
                    # Stuck, break
                    break

            return adj, False

        # Start search
        adj = None
        for _ in range(50):
            adj_try = random_4_regular(n_target)
            if adj_try:
                adj = adj_try
                break
        
        if not adj:
            return luw_construct(k, g)

        current_girth, _ = get_girth_and_cycle(adj)
        
        # Run improvement loop
        adj, success = try_improve(adj, current_girth, deadline)
        
        # Final check
        final_girth, _ = get_girth_and_cycle(adj)
        if final_girth >= g:
            edges = []
            for u in range(n_target):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n_target, edges
        else:
            # Fallback
            return luw_construct(k, g)

    else:
        return luw_construct(k, g)
