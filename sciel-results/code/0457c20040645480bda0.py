import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # 5s margin

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))

    # ---- Lazebnik-Ustimenko-Woldar construction (seed) ----
    def seed_construct(k, g):
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
            r_c, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_c]

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

    seed_n, seed_edges = seed_construct(k, g)

    # ---- Helper functions ----
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Check girth using BFS from each vertex, stopping at depth g//2."""
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def find_shortest_cycle(n, adj, g):
        """Find a cycle shorter than g, or return None if none exists."""
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                # Reconstruct the cycle
                                cycle = [s]
                                # Walk back from u to s
                                path_u = []
                                cur = u
                                while cur != -1:
                                    path_u.append(cur)
                                    cur = parent[cur]
                                path_u.reverse()
                                # Walk back from v to s
                                path_v = []
                                cur = v
                                while cur != -1:
                                    path_v.append(cur)
                                    cur = parent[cur]
                                path_v.reverse()
                                # Find LCA
                                set_v = set(path_v)
                                lca = None
                                for node in path_u:
                                    if node in set_v:
                                        lca = node
                                        break
                                if lca is not None:
                                    idx_u = path_u.index(lca)
                                    idx_v = path_v.index(lca)
                                    cycle = path_u[idx_u:] + list(reversed(path_v[idx_v + 1:]))
                                    return cycle
                                return [u, v]  # fallback
        return None

    def count_short_cycles(n, adj, g, max_cycles=100):
        """Count number of cycles shorter than g (up to max_cycles)."""
        half_g = g // 2
        count = 0
        seen_cycles = set()
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u] and dist[u] + dist[v] + 1 < g:
                            # Canonical form of the cycle edge
                            key = (min(u, v), max(u, v))
                            if key not in seen_cycles:
                                seen_cycles.add(key)
                                count += 1
                                if count >= max_cycles:
                                    return count
        return count

    def generate_simple_regular(n, k):
        """Generate a simple k-regular graph using the configuration model with backtracking."""
        for attempt in range(20):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            edge_set = set()
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append((u, v))
            if valid:
                return edges
        return None

    def repair_graph(edges, n, k, g, max_iterations=1000):
        """Try to repair a graph by swapping edges to eliminate short cycles."""
        adj = build_adj(edges, n)
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            cycle = find_shortest_cycle(n, adj, g)
            if cycle is None:
                return edges
            if len(cycle) < 3:
                return None
            
            # Try to break the cycle by swapping an edge
            # Find an edge in the cycle and try to swap it
            cycle_edges = []
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                cycle_edges.append((u, v))
            
            swapped = False
            for u, v in cycle_edges:
                # Find a neighbor of u that is not in the cycle, and a neighbor of v that is not in the cycle
                u_neighbors = [x for x in adj[u] if x != v]
                v_neighbors = [x for x in adj[v] if x != u]
                
                for w in u_neighbors:
                    for x in v_neighbors:
                        if w == x:
                            continue
                        # Check if (w, x) is already an edge
                        if (min(w, x), max(w, x)) in edge_set:
                            continue
                        # Check that w is not adjacent to v and x is not adjacent to u (to avoid creating triangles)
                        if v in adj[w] or u in adj[x]:
                            continue
                        # Perform the swap
                        edge_set.remove((min(u, v), max(u, v)))
                        edge_set.add((min(u, w), max(u, w)))
                        edge_set.add((min(v, x), max(v, x)))
                        
                        # Update adj
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[u].append(w)
                        adj[w].append(u)
                        adj[v].append(x)
                        adj[x].append(v)
                        
                        # Update edges list
                        edges.remove((u, v))
                        edges.append((min(u, w), max(u, w)))
                        edges.append((min(v, x), max(v, x)))
                        
                        swapped = True
                        break
                    if swapped:
                        break
                if swapped:
                    break
            
            if not swapped:
                break
        
        # Final check
        adj = build_adj(edges, n)
        if exact_girth_check(n, adj, g):
            return edges
        return None

    # ---- Search for better graphs ----
    best_n = seed_n
    best_edges = seed_edges

    # Try n values from Moore bound up to seed_n
    # Use more trials for smaller n values
    max_n_to_try = min(seed_n, moore_bound + 200)
    
    for n in range(moore_bound, max_n_to_try):
        if time.time() > deadline - 20:
            break
        
        # Number of trials decreases as n increases (larger n is harder)
        trials = max(2, 10 - (n - moore_bound) // 30)
        
        for trial in range(trials):
            if time.time() > deadline - 20:
                break
            
            edges = generate_simple_regular(n, k)
            if edges is None:
                continue

            adj = build_adj(edges, n)
            edge_set = set((min(u, v), max(u, v)) for u, v in edges)

            # Quick heuristic check with a few sources
            half_g = g // 2
            quick_fail = False
            sample_size = min(5, n)
            for s in random.sample(range(n), sample_size):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    if dist[u] >= half_g:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            if v != parent[u]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < g:
                                    quick_fail = True
                                    break
                    if quick_fail:
                        break
                if quick_fail:
                    break
            
            if quick_fail:
                continue

            # Full check
            if exact_girth_check(n, adj, g):
                return n, edges

    # Try repair-based approach on random graphs
    for n in range(moore_bound, min(moore_bound + 100, seed_n)):
        if time.time() > deadline - 10:
            break
        
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            edges = generate_simple_regular(n, k)
            if edges is None:
                continue
            
            adj = build_adj(edges, n)
            edge_set = set((min(u, v), max(u, v)) for u, v in edges)
            
            # Count short cycles first
            n_short = count_short_cycles(n, adj, g, max_cycles=50)
            if n_short > 20:
                continue  # Too many short cycles, skip repair
            
            # Try to repair
            repaired = repair_graph(edges, n, k, g, max_iterations=500)
            if repaired is not None:
                adj = build_adj(repaired, n)
                if exact_girth_check(n, adj, g):
                    return n, repaired

    return best_n, best_edges
