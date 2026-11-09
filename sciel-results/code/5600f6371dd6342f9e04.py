import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth g using Iterative Local Search.
    Falls back to LUW construction if search fails.
    """
    start_time = time.time()
    time_limit = 115  # Leave buffer

    # Moore Bound Calculation
    def moore_bound(k, g):
        if k == 2:
            return g
        if g % 2 == 1:
            terms = (g - 1) // 2
            if k == 1:
                return g
            if k == 2:
                return 1 + 2 * terms
            return 1 + k * (( (k-1)**terms - 1 ) // (k - 2))
        else:
            terms = g // 2 - 1
            if k == 2:
                return 2 * terms
            return 2 * (( (k-1)**terms - 1 ) // (k - 2))

    n_min = moore_bound(k, g)
    
    # LUW Fallback
    def luw_construct(k, g):
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

    # Check if graph has girth >= g
    def has_girth(adj, g):
        n = len(adj)
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            dist = [0] * n
            queue = deque()
            queue.append(start)
            visited[start] = start
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = u
                        parent[v] = u
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def find_short_cycle(adj, g):
        n = len(adj)
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            dist = [0] * n
            queue = deque()
            queue.append(start)
            visited[start] = start
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = u
                        parent[v] = u
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            cycle = [u]
                            curr = u
                            while curr != v:
                                curr = parent[curr]
                                cycle.append(curr)
                            cycle.append(v)
                            return cycle
        return None

    # Generate random k-regular graph
    def generate_random_regular(n, k):
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
            
        adj = [[] for _ in range(n)]
        adj_sets = [set() for _ in range(n)]
        
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            key = (min(u,v), max(u,v))
            if key in edges:
                return None
            edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            
        for i in range(n):
            if len(adj[i]) != k:
                return None
                
        return adj, adj_sets

    def local_search(n, k, g, t_limit):
        adj = None
        adj_sets = None
        for _ in range(100):
            res = generate_random_regular(n, k)
            if res:
                adj, adj_sets = res
                break
        if not adj:
            return None
            
        if has_girth(adj, g):
            return adj
            
        while time.time() - start_time < t_limit:
            cycle = find_short_cycle(adj, g)
            if not cycle:
                if has_girth(adj, g):
                    return adj
                break
                
            # Pick random edge from cycle
            edge_idx = random.randint(0, len(cycle) - 1)
            u = cycle[edge_idx]
            v = cycle[(edge_idx + 1) % len(cycle)]
            
            # Find swap partner
            # Randomly pick another edge
            swapped = False
            for _ in range(50):
                x = random.randint(0, n-1)
                if x == u or x == v:
                    continue
                y_list = adj[x]
                if not y_list:
                    continue
                y = random.choice(y_list)
                if y == u or y == v:
                    continue
                if x == y:
                    continue
                # Check if (x,y) is in cycle?
                # If x,y are in cycle, swapping might not help or might create multi-edge
                # Simple check: if (x,y) is the same as (u,v), skip.
                if (min(x,y), max(x,y)) == (min(u,v), max(u,v)):
                    continue
                
                # Check validity of swap (u,v)->(u,x) and (x,y)->(v,y)
                # Need (u,x) not in adj[u] and (v,y) not in adj[v]
                if x in adj_sets[u] or y in adj_sets[v]:
                    continue
                # Also need to ensure we don't create self loops (x!=u, y!=v handled)
                
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].append(x)
                adj[x].append(u)
                adj[v].append(y)
                adj[y].append(v)
                
                adj_sets[u].discard(v)
                adj_sets[v].discard(u)
                adj_sets[x].discard(y)
                adj_sets[y].discard(x)
                
                adj_sets[u].add(x)
                adj_sets[x].add(u)
                adj_sets[v].add(y)
                adj_sets[y].add(v)
                
                swapped = True
                break
                
            if not swapped:
                # Try other edge in cycle?
                continue

        return None

    # Main Loop
    n_try = max(n_min + 50, 100)
    while time.time() - start_time < time_limit:
        adj = local_search(n_try, k, g, time_limit)
        if adj:
            if has_girth(adj, g):
                edges = []
                for i in range(n_try):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n_try, edges
        n_try += 50
        if n_try > 5000:
             break

    return luw_construct(k, g)
