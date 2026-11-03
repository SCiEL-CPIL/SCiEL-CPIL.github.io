import time
import random
import itertools
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Fallback: LUW graph
    def luw_construct():
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
        i = 0
        while i < len(order):
            if time.time() > deadline:
                break
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        n = len(order)
        return n, edges

    if k != 5 or g != 9:
        return luw_construct()

    # Try to find a smaller graph using random generation and repair
    # Target: beat LUW (n=6250)
    # Known best: 1116. Moore bound: 426.
    
    def check_girth_fast(n, adj, min_girth):
        """
        Quick girth check using BFS from random vertices first.
        Returns True if girth >= min_girth, False otherwise.
        """
        depth_limit = (min_girth - 1) // 2
        
        # Quick check: BFS from a few random vertices
        for _ in range(min(5, n)):
            s = random.randint(0, n - 1)
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= depth_limit:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
        
        # Full check from all vertices
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= depth_limit:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
        
        return True

    def find_short_cycle(n, adj, min_girth):
        """
        Returns a list of vertices forming a cycle of length < min_girth, or None.
        """
        depth_limit = (min_girth - 1) // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= depth_limit:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                # Reconstruct cycle using LCA
                                path_u = []
                                cur = u
                                while cur != s:
                                    path_u.append(cur)
                                    cur = parent[cur]
                                path_u.append(s)
                                path_u.reverse()
                                
                                path_v = []
                                cur = v
                                while cur != s:
                                    path_v.append(cur)
                                    cur = parent[cur]
                                path_v.append(s)
                                path_v.reverse()
                                
                                # Find LCA
                                set_u = set(path_u)
                                lca = s
                                for node in path_v:
                                    if node in set_u:
                                        lca = node
                                        break
                                
                                # Path from lca to u
                                p1 = []
                                cur = u
                                while cur != lca:
                                    p1.append(cur)
                                    cur = parent[cur]
                                p1.append(lca)
                                p1.reverse()
                                
                                # Path from lca to v
                                p2 = []
                                cur = v
                                while cur != lca:
                                    p2.append(cur)
                                    cur = parent[cur]
                                p2.append(lca)
                                p2.reverse()
                                
                                # Cycle: p1 + p2[1:]
                                cycle = p1 + p2[1:]
                                return cycle
        return None

    def generate_random_regular(n, k):
        # Configuration model
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        edges = set()
        adj = [[] for _ in range(n)]
        
        # Pair stubs
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if (u, v) in edges or (v, u) in edges:
                return None
            edges.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
            
        return adj, list(edges)

    def repair(n, adj, edges, g):
        """
        Attempts to increase girth by swapping edges.
        Returns (n, edges) if successful, else None.
        """
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                return None
            
            # Quick girth check
            if check_girth_fast(n, adj, g):
                return n, list(edges)
            
            # Find a short cycle
            cycle = find_short_cycle(n, adj, g)
            if cycle is None:
                break
            
            # Get edges in cycle
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                cycle_edges.append((min(u,v), max(u,v)))
            
            if len(cycle_edges) < 2:
                break
            
            # Try multiple random swaps to break the cycle
            found_swap = False
            for _ in range(10):
                e1, e2 = random.sample(cycle_edges, 2)
                a, b = e1
                c, d = e2
                
                # Try swap 1: (a,c) and (b,d)
                if a != c and b != d and a != d and b != c:
                    if (a,c) not in edges and (b,d) not in edges:
                        edges.discard(e1)
                        edges.discard(e2)
                        edges.add((min(a,c), max(a,c)))
                        edges.add((min(b,d), max(b,d)))
                        
                        adj[a].remove(b)
                        adj[b].remove(a)
                        adj[c].remove(d)
                        adj[d].remove(c)
                        
                        adj[a].append(c)
                        adj[c].append(a)
                        adj[b].append(d)
                        adj[d].append(b)
                        
                        found_swap = True
                        break
                
                # Try swap 2: (a,d) and (b,c)
                if a != d and b != c:
                    if (a,d) not in edges and (b,c) not in edges:
                        edges.discard(e1)
                        edges.discard(e2)
                        edges.add((min(a,d), max(a,d)))
                        edges.add((min(b,c), max(b,c)))
                        
                        adj[a].remove(b)
                        adj[b].remove(a)
                        adj[c].remove(d)
                        adj[d].remove(c)
                        
                        adj[a].append(d)
                        adj[d].append(a)
                        adj[b].append(c)
                        adj[c].append(b)
                        
                        found_swap = True
                        break
            
            if not found_swap:
                break
        
        return None

    # Search for n
    best_n = float('inf')
    best_edges = None
    
    # Try n from 1000 to 1500, step 2
    for n in range(1000, 1501, 2):
        if time.time() > deadline - 15:
            break
            
        attempts = 3
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
                
            gen = generate_random_regular(n, k)
            if gen is None:
                continue
                
            adj, edges = gen
            result = repair(n, adj, set(edges), g)
            if result:
                rn, r_edges = result
                if rn < best_n:
                    best_n = rn
                    best_edges = r_edges
                return rn, r_edges
                
    # If no graph found, return LUW
    return luw_construct()
