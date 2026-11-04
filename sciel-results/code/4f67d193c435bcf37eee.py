import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    # Moore Bound Calculation
    r = g // 2
    if g % 2 == 0:
        if k == 2:
            n_min = g
        else:
            n_min = 1 + k * sum((k-1)**i for i in range(r-1))
    else:
        if k == 2:
            n_min = g
        else:
            n_min = 1 + k * sum((k-1)**i for i in range(r-1))

    # Fallback: Lazebnik-Ustimenko-Woldar
    def luw_fallback():
        q = k
        # Choose minimal d such that girth >= g
        if g <= 5:
            d = 3
        else:
            d = 3
            while True:
                girth_est = d + 5 if d % 2 == 1 else d + 4
                if girth_est >= g:
                    break
                d += 1
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_val, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_val]

        # Precompute rule arrays for faster access
        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        for c in range(2, d + 1):
            rule_a[c], rule_b[c] = rule[c]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (0,) * (d + 1))
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Girth check using BFS from each vertex (non-backtracking)
    def check_girth(adj, g):
        n = len(adj)
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = [u]
            head = 0
            while head < len(queue):
                v = queue[head]
                head += 1
                if dist[v] >= g // 2:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v] and parent[w] != v:
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < g:
                            return False
        return True

    def generate_random_k_regular(n, k):
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        
        while True:
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj

    def find_short_cycle(adj, g):
        """Find a cycle shorter than g. Returns list of vertices in cycle, or None."""
        n = len(adj)
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = [u]
            head = 0
            while head < len(queue):
                v = queue[head]
                head += 1
                if dist[v] >= g // 2:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v] and parent[w] != v:
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            cycle = []
                            # Path from w to u
                            path_w = []
                            cur = w
                            while cur != -1:
                                path_w.append(cur)
                                cur = parent[cur]
                            path_w.reverse()  # now from u to w
                            # Path from v to u
                            path_v = []
                            cur = v
                            while cur != -1:
                                path_v.append(cur)
                                cur = parent[cur]
                            path_v.reverse()  # now from u to v
                            # Cycle: u -> ... -> v -> w -> ... -> u
                            cycle = path_v + path_w[1:]  # v's path includes u, w's path includes u
                            # Actually, let's be more careful
                            # path_v = [u, ..., v], path_w = [u, ..., w]
                            # cycle = path_v + [w] + reversed(path_w[1:])
                            # But we need to avoid duplicating u
                            cycle = path_v + list(reversed(path_w[1:]))
                            return cycle
        return None

    def find_graph_local_search(k, g, n, time_budget=5.0):
        start = time.time()
        max_restarts = 30
        
        for _ in range(max_restarts):
            if time.time() - start > time_budget:
                break
            
            adj = generate_random_k_regular(n, k)
            
            # Check if valid
            if check_girth(adj, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return edges
            
            # Try to fix short cycles via edge swaps
            for iteration in range(200):
                if time.time() - start > time_budget:
                    break
                
                if check_girth(adj, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return edges
                
                # Find a short cycle
                cycle = find_short_cycle(adj, g)
                if cycle is None:
                    # Shouldn't happen if check_girth failed, but be safe
                    break
                
                # Pick an edge on the cycle to swap
                if len(cycle) < 2:
                    break
                    
                # Get edges along the cycle
                cycle_edges = []
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    if u < v:
                        cycle_edges.append((u, v))
                    else:
                        cycle_edges.append((v, u))
                
                if not cycle_edges:
                    break
                
                # Pick a random edge from the cycle
                e1 = random.choice(cycle_edges)
                u1, v1 = e1
                
                # Find a non-cycle edge to swap with
                # Collect all edges not in the cycle
                cycle_edge_set = set()
                for u in cycle:
                    for v in cycle:
                        if u != v:
                            cycle_edge_set.add((min(u,v), max(u,v)))
                
                all_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            all_edges.append((u, v))
                
                # Filter out cycle edges and the edge e1
                non_cycle_edges = [e for e in all_edges if e not in cycle_edge_set]
                
                if not non_cycle_edges:
                    break
                
                e2 = random.choice(non_cycle_edges)
                u2, v2 = e2
                
                # Perform swap: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
                # Try both orientations
                for swap_type in range(2):
                    if swap_type == 0:
                        new_e1 = (u1, u2)
                        new_e2 = (v1, v2)
                    else:
                        new_e1 = (u1, v2)
                        new_e2 = (v1, u2)
                    
                    # Check validity
                    if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                        continue
                    if new_e1[0] == new_e2[0] or new_e1[0] == new_e2[1] or new_e1[1] == new_e2[0] or new_e1[1] == new_e2[1]:
                        continue
                    if new_e1 in adj[new_e1[0]] or new_e2 in adj[new_e2[0]]:
                        continue
                    
                    # Perform the swap
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                    a, b = new_e1
                    c, d = new_e2
                    adj[a].append(b)
                    adj[b].append(a)
                    adj[c].append(d)
                    adj[d].append(c)
                    
                    break
                else:
                    # No valid swap found, try a random swap
                    all_edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                all_edges.append((u, v))
                    if len(all_edges) < 2:
                        break
                    e1, e2 = random.sample(all_edges, 2)
                    u1, v1 = e1
                    u2, v2 = e2
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    if u1 != u2 and v1 != v2 and u1 != v2 and v1 != u2:
                        if u2 not in adj[u1] and v2 not in adj[v1]:
                            adj[u1].append(u2)
                            adj[u2].append(u1)
                            adj[v1].append(v2)
                            adj[v2].append(v1)
                        else:
                            adj[u1].append(v1)
                            adj[v1].append(u1)
                            adj[u2].append(v2)
                            adj[v2].append(u2)
                    else:
                        adj[u1].append(v1)
                        adj[v1].append(u1)
                        adj[u2].append(v2)
                        adj[v2].append(u2)

        return None

    # Main Search Loop
    max_n = n_min + 150
    
    for n in range(n_min, max_n):
        if time.time() > deadline:
            break
        
        edges = find_graph_local_search(k, g, n, time_budget=3.0)
        if edges:
            return n, edges

    # Fallback
    return luw_fallback()
