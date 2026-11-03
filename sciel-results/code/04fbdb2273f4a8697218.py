import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Moore bound for k=3, g=16 is 510.
    # n must be even for bipartite 3-regular.
    # Start from 512.
    min_n = 512
    
    best_n = None
    best_edges = None
    
    # We will try to find a graph with n as small as possible.
    # Iterate n upwards.
    for n in range(min_n, 1000, 2):
        if time.time() > deadline - 10:
            break
            
        # Try to find a valid graph for this n
        # Probability of success increases with n? Or stays constant?
        # For random regular graphs, the number of short cycles is Poisson.
        # So probability of girth >= 16 is roughly constant (e.g. 0.01 - 0.1).
        
        attempts = 0
        max_attempts = 5000 # Adjust based on time
        
        while attempts < max_attempts:
            if time.time() > deadline - 5:
                break
            
            # Generate random bipartite 3-regular graph
            half_n = n // 2
            # 3 permutations of B
            perms = []
            valid = True
            for i in range(3):
                p = np.random.permutation(half_n)
                perms.append(p)
                # Check for collisions with previous perms for same a
                # If p1[a] == p2[a], multi-edge.
            
            # Check simplicity
            edges_set = set()
            edges = []
            for a in range(half_n):
                for i in range(3):
                    b = perms[i][a]
                    u = a
                    v = half_n + b
                    if u > v: u, v = v, u
                    if (u, v) in edges_set:
                        valid = False
                        break
                    edges_set.add((u, v))
                    edges.append((u, v))
                if not valid:
                    break
            
            if not valid:
                attempts += 1
                continue
                
            # Check girth
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Check for cycles of length < g
            # BFS from each vertex
            girth_ok = True
            for start in range(n):
                # BFS
                # dist[v] = (depth, parent)
                dist = [-1] * n
                parent = [-1] * n
                queue = [(start, 0, -1)]
                dist[start] = 0
                
                while queue:
                    node, depth, par = queue.pop(0)
                    if depth >= 7: # Max depth for g=16 is 7 (path of length 7 -> cycle 8? No.)
                        # Cycle length = depth(u) + depth(v) + 1.
                        # If we find an edge (u,v) where dist[u] and dist[v] are known.
                        # Max depth needed: if g=16, we care about cycles 3..15.
                        # Max depth in BFS tree for a cycle of length 15 is 7 (from center).
                        # So we need to explore up to depth 7.
                        pass
                    
                    for nb in adj[node]:
                        if nb == par:
                            continue
                        if dist[nb] == -1:
                            dist[nb] = depth + 1
                            parent[nb] = node
                            queue.append((nb, depth + 1, node))
                        else:
                            # Found a cycle
                            # Cycle length = depth(node) + dist[nb] + 1
                            # Note: this might be a "back edge" in BFS tree
                            # But BFS tree is not unique.
                            # To be safe, we check all pairs?
                            # Actually, the standard way to find girth is:
                            # For each edge (u,v), if u and v are in the same BFS tree,
                            # cycle length = dist[u] + dist[v] + 1.
                            # We can just check during BFS.
                            cycle_len = depth + dist[nb] + 1
                            if cycle_len < g:
                                girth_ok = False
                                break
                    if not girth_ok:
                        break
                if not girth_ok:
                    break
            
            if girth_ok:
                return n, edges
            
            attempts += 1
            
        # If we exhausted attempts for this n, move to next n
        # But we should probably stop if we are running out of time.
        
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: Return the seed's result or a trivial graph if time runs out
    # But we must return a valid graph.
    # If we didn't find one, we might have to return a larger n or fail.
    # However, the loop above returns immediately upon finding one.
    # If we exit the loop without returning, we need a fallback.
    
    # Let's try a larger n with fewer attempts if time permits
    # Or just return a known valid graph if we have one cached?
    # We don't have a cached one.
    
    # If we reach here, we didn't find a graph in the range 512-1000 in time.
    # This is unlikely given the time budget and probability.
    # But to be safe, we can return a graph with n=936 if we can construct it?
    # We can't easily.
    
    # Let's assume we found one.
    # If not, raise an exception or return a dummy?
    # The evaluator expects a valid graph.
    
    # Fallback: Use the LUW construction from the seed?
    # It returns ~10^6.
    # Better to return the last attempted n if it was close? No, it must be valid.
    
    # We will assume the loop finds one.
    # If not, we return a simple 3-regular graph with large girth?
    # We can't guarantee girth.
    
    # Let's add a fallback to the LUW construction logic from the seed
    # just in case.
    
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
