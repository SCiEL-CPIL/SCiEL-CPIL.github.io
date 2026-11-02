import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0 # 5s margin
    
    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
    def seed_construct(k, g):
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

    seed_n, seed_edges = seed_construct(k, g)
    
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Check girth using BFS from each vertex, stopping early if possible."""
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

    def generate_random_regular(n, k):
        """Generate a simple k-regular graph using configuration model with repair."""
        if n * k % 2 != 0:
            return None
        
        for attempt in range(100):
            if time.time() > deadline - 5:
                return None
            
            # Configuration model
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            edges = []
            for i in range(0, n*k, 2):
                u, v = stubs[i], stubs[i+1]
                edges.append((u, v))
            
            # Check for self-loops and multi-edges
            edge_set = set()
            valid = True
            for u, v in edges:
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
            
            if valid:
                return edges
        
        return None

    def eliminate_short_cycles(adj, n, k, g, max_iterations=5000):
        """Use 2-switch moves to eliminate cycles shorter than g."""
        adj_set = [set(adj[i]) for i in range(n)]
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Find a short cycle
            short_cycle = None
            for s in range(n):
                # BFS up to depth (g-1)//2
                dist = {s: 0}
                parent = {s: -1}
                q = deque([s])
                while q:
                    u = q.popleft()
                    if dist[u] >= (g-1)//2:
                        continue
                    for v in adj_set[u]:
                        if v == parent[u]:
                            continue
                        if v in dist:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                # Reconstruct cycle
                                cycle = []
                                cu = u
                                while cu != v:
                                    cycle.append(cu)
                                    cu = parent[cu]
                                cycle.append(v)
                                # Add path from v back to s
                                cv = v
                                while cv != s:
                                    cycle.append(cv)
                                    cv = parent[cv]
                                cycle.append(s)
                                short_cycle = cycle
                                break
                        else:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                if short_cycle:
                    break
            
            if not short_cycle:
                break
            
            # Try to break the cycle using a 2-switch
            # Pick two edges in the cycle and swap their endpoints
            cycle_len = len(short_cycle)
            if cycle_len < 3:
                continue
            
            # Find edges in the cycle
            cycle_edges = []
            for i in range(len(short_cycle)):
                u = short_cycle[i]
                v = short_cycle[(i+1) % len(short_cycle)]
                if v in adj_set[u]:
                    cycle_edges.append((u, v))
            
            if len(cycle_edges) < 2:
                continue
            
            # Try random 2-switches
            found_switch = False
            for _ in range(50):
                i, j = random.sample(range(len(cycle_edges)), 2)
                u1, v1 = cycle_edges[i]
                u2, v2 = cycle_edges[j]
                
                # Try swapping: (u1,v1) and (u2,v2) -> (u1,v2) and (u2,v1)
                new_e1 = (u1, v2)
                new_e2 = (u2, v1)
                
                # Check validity
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue
                if v2 in adj_set[u1] and v1 in adj_set[u2]:
                    # Already have these edges
                    continue
                if v1 in adj_set[u1] and v2 in adj_set[u2]:
                    # Would create multi-edge
                    continue
                
                # Apply switch
                adj_set[u1].discard(v1)
                adj_set[u2].discard(v2)
                adj_set[u1].add(v2)
                adj_set[u2].add(v1)
                adj_set[v1].discard(u1)
                adj_set[v2].discard(u2)
                adj_set[v1].add(u2)
                adj_set[v2].add(u1)
                found_switch = True
                break
            
            if not found_switch:
                # Try a different approach: random 2-switch anywhere
                for _ in range(100):
                    u1 = random.randint(0, n-1)
                    u2 = random.randint(0, n-1)
                    if u1 == u2:
                        continue
                    if len(adj_set[u1]) < k or len(adj_set[u2]) < k:
                        continue
                    
                    v1 = random.choice(list(adj_set[u1]))
                    v2 = random.choice(list(adj_set[u2]))
                    
                    # Skip if it would create self-loop or multi-edge
                    if v1 == u2 or v2 == u1:
                        continue
                    if v2 in adj_set[u1] or v1 in adj_set[u2]:
                        continue
                    
                    # Apply switch
                    adj_set[u1].discard(v1)
                    adj_set[u2].discard(v2)
                    adj_set[u1].add(v2)
                    adj_set[u2].add(v1)
                    adj_set[v1].discard(u1)
                    adj_set[v2].discard(u2)
                    adj_set[v1].add(u2)
                    adj_set[v2].add(u1)
                    found_switch = True
                    break
            
            if not found_switch:
                break
        
        # Convert back to list
        adj_list = [list(adj_set[i]) for i in range(n)]
        edges = []
        for i in range(n):
            for j in adj_list[i]:
                if i < j:
                    edges.append((i, j))
        return edges, adj_list

    # Search
    best_n = seed_n
    best_edges = seed_edges
    
    # For k=5, g=9: Moore bound is 428, seed is 1116
    # Try to find smaller graphs with random generation + cycle elimination
    
    search_start = max(moore_bound, 430)
    search_end = min(seed_n - 1, 700)
    
    if search_start < search_end:
        # Try n values in increasing order
        for n in range(search_start, min(search_start + 50, search_end + 1), 2):
            if time.time() > deadline - 15:
                break
            
            # More attempts for smaller n
            attempts = max(2, 8 - (n - search_start) // 10)
            
            for attempt in range(attempts):
                if time.time() > deadline - 15:
                    break
                
                edges = generate_random_regular(n, k)
                if edges is None:
                    continue
                
                adj = build_adj(edges, n)
                
                # Try to eliminate short cycles
                new_edges, new_adj = eliminate_short_cycles(adj, n, k, g, max_iterations=2000)
                
                # Check girth
                if exact_girth_check(n, new_adj, g):
                    return n, new_edges
    
    return seed_n, seed_edges
