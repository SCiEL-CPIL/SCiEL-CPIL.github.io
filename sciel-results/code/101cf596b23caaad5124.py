import numpy as np
import random
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Parameters for the search
    # The known best is 320. We try a range of n.
    target_ns = list(range(320, 450, 10))
    
    best_n = None
    best_edges = None
    
    # Fallback: LUW construction from seed
    def luw_fallback():
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
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start_v = (0, (None,) + (0,) * d)
        index, order, edges = {start_v: 0}, [start_v], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def generate_random_4_reg(n):
        # Random 4-regular graph using configuration model
        # 2n stubs, each vertex 4 stubs
        stubs = []
        for i in range(n):
            stubs.extend([i]*4)
        random.shuffle(stubs)
        adj = {i: set() for i in range(n)}
        for i in range(0, 2*n, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Retry pairing (simple restart for simplicity in random gen)
                # For n>20, probability is low
                pass 
            if v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
        # Verify degree
        for i in range(n):
            if len(adj[i]) != 4:
                return None
        return adj

    def get_girth(adj, n):
        # BFS from each node to find shortest cycle
        # Optimization: Only check up to g
        min_girth = g
        # Use a simple BFS for each start node
        # For n=320, this is fast
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            # Early exit if we find 4
                            if min_girth == 4:
                                return min_girth
            if min_girth == 4:
                return min_girth
        return min_girth

    def find_short_cycle(adj, n, limit):
        # Returns (cycle_nodes, length) or None
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < limit:
                            # Reconstruct path
                            path1 = []
                            cur = u
                            while cur != start:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.append(start)
                            path1.reverse()
                            
                            path2 = []
                            cur = v
                            while cur != start:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.append(start)
                            path2.reverse()
                            
                            # Merge
                            # Find common prefix? No, BFS tree from start.
                            # u and v are children of different branches or same?
                            # Simple cycle: path1 + path2 (excluding start)
                            cycle = path1 + path2[1:]
                            # Note: this might not be simple if parents overlap weirdly, 
                            # but in a tree, path to u and v intersect at LCA.
                            # The cycle is path(u) + path(v) - 2*path(LCA).
                            # For girth check, we just need existence.
                            # For 2-switch, we need the actual edges.
                            return cycle, cycle_len
        return None, limit

    def apply_2_switch(adj, n):
        # Pick two edges (u,v) and (w,x) and swap to (u,w), (v,x) or (u,x), (v,w)
        # Must maintain simple graph (no loops, no multiedges)
        # And degree 4.
        
        # Collect all edges
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        if not edges:
            return False
        
        # Randomly pick two edges
        # To be effective, pick edges that are part of short cycles?
        # For now, random 2-switches.
        
        for _ in range(20): # Try 20 random switches
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            if e1 == e2:
                continue
            
            u, v = e1
            w, x = e2
            
            # Check if (u,w) and (v,x) are valid
            # Valid if:
            # 1. u != w, v != x (no loops)
            # 2. w not in adj[u], x not in adj[v] (no multi-edge for new)
            # 3. The old edges are removed.
            
            if u == w or v == x:
                continue
            if w in adj[u] or x in adj[v]:
                continue
            
            # Also need to ensure (u,x) and (v,w) are not the ones we are checking?
            # We are replacing (u,v) and (w,x) with (u,w) and (v,x).
            # We must ensure (u,w) and (v,x) are not existing edges.
            # We checked that.
            
            # What if (u,x) is an edge? It stays an edge? No, we don't touch it.
            # But if (u,x) was an edge, and we make (u,w), it's fine.
            # Wait, we need to ensure we don't create a multi-edge.
            # If (u,w) is already an edge, we skip.
            
            # Perform switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj[u].add(w)
            adj[w].add(u)
            adj[v].add(x)
            adj[x].add(v)
            
            return True
            
        return False

    # Main Search Loop
    for n_target in target_ns:
        if time.time() > deadline - 5:
            break
            
        # Try to find a valid graph for this n
        # Limit iterations per n
        max_iter = 500
        found = False
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            adj = generate_random_4_reg(n_target)
            if not adj:
                continue
            
            # Check girth
            girth = get_girth(adj, n_target)
            if girth >= g:
                # Valid!
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                best_n = n_target
                best_edges = edges
                found = True
                break
            
            # If girth < g, try to repair
            # If girth is very small (4, 5), repair is easier
            # If girth is 9, it's hard.
            
            # Do some 2-switches
            for _ in range(10):
                apply_2_switch(adj, n_target)
                girth = get_girth(adj, n_target)
                if girth >= g:
                    edges = []
                    for u in range(n_target):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n_target
                    best_edges = edges
                    found = True
                    break
            
            if found:
                break
        
        if found:
            # Found a valid graph for this n. 
            # Since we iterate n ascending, this is the best n for this n_target.
            # But maybe smaller n_target worked? 
            # We iterate 320, 330... so the first success is the smallest.
            # We can break here? 
            # Yes, because we want minimal n.
            break

    if best_n is not None:
        return best_n, best_edges
    else:
        # Fallback to LUW
        return luw_fallback()

# EVOLVE-BLOCK-END
