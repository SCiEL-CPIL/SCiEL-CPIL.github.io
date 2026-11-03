"""Improved cage graph construction: LUW base + edge-swap local search."""
import time
import random
import sys

def construct(k, g):
    deadline = time.time() + 135  # safety margin
    
    # ---- LUW construction (known girth >= g for our parameters) ----
    def luw_construct(k, g):
        q = k
        if g <= 6:
            d = 3
        elif g <= 8:
            d = 4
        else:
            d = 5  # girth >= 10
        
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q
        
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side), tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start: 0}, [start], []
        for v in order:
            for other in neighbours(v):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if v[0] == 0:
                    edges.append((index[v], index[other]))
        return len(order), edges
    
    n, edges = luw_construct(k, g)
    best_n, best_edges = n, edges
    
    # Build adjacency list
    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj
    
    adj = build_adj(n, edges)
    
    # ---- Girth check: BFS from each vertex to depth g//2 ----
    def find_short_cycle(adj, n, g):
        """Find a cycle of length < g. Returns (cycle, length) or (None, None)."""
        half = g // 2
        for start in range(n):
            # BFS with parent tracking
            visited = {start: (None, None)}  # node -> (parent, depth)
            queue = [(start, None, 0)]
            while queue:
                node, parent, depth = queue.pop(0)
                if depth >= half:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb in visited:
                        # Check if this creates a cycle
                        v_parent, v_depth = visited[nb]
                        # The cycle goes from node to nb through edge (node, nb)
                        # and back through nb's parent chain
                        if v_depth + depth + 1 < g:
                            # Found a short cycle
                            cycle = [node, nb]
                            cur = nb
                            while cur != start and cur is not None:
                                cur = visited[cur][0]
                                if cur is not None:
                                    cycle.append(cur)
                            cycle.append(start)
                            # Remove duplicates to get actual cycle
                            # Actually let's just return the length
                            return (cycle, v_depth + depth + 1)
                    else:
                        visited[nb] = (node, depth + 1)
                        queue.append((nb, node, depth + 1))
        return (None, None)
    
    # ---- Edge swap local search ----
    def edge_swap(adj, n, edges, g, budget):
        """Try to improve by swapping edges. Returns (n, edges) best found."""
        best_n, best_edges = n, edges[:]
        edge_set = set()
        for i, j in edges:
            edge_set.add((min(i,j), max(i,j)))
        
        def remove_edge(u, v):
            u, v = (min(u,v), max(u,v))
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.discard((u, v))
            edges.remove((u, v)) if (u, v) in edges else edges.remove((v, u))
        
        def add_edge(u, v):
            u, v = (min(u,v), max(u,v))
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v))
            edges.append((u, v))
        
        import random
        rng = random.Random(42)
        
        start_time = time.time()
        max_iters = 50000
        iters = 0
        
        while iters < max_iters and time.time() - start_time < budget:
            iters += 1
            
            # Find a short cycle
            cycle, clen = find_short_cycle(adj, n, g)
            if cycle is None:
                break  # No short cycles found
            
            # Try to break the cycle by swapping an edge
            # Pick a random edge on the cycle
            if len(cycle) >= 3:
                idx = rng.randint(0, len(cycle) - 1)
                u = cycle[idx]
                # v is the next node in cycle
                v = cycle[(idx + 1) % len(cycle)]
                
                if (min(u,v), max(u,v)) not in edge_set:
                    continue
                
                # Remove edge (u, v)
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.discard((min(u,v), max(u,v)))
                # Remove from edges list
                for i, (a, b) in enumerate(edges):
                    if (a == u and b == v) or (a == v and b == u):
                        edges.pop(i)
                        break
                
                # Find a good replacement edge
                # Look for a vertex w in adj[u] and x in adj[v] such that
                # adding (w, x) doesn't create a short cycle
                found = False
                candidates_u = adj[u][:]
                candidates_v = adj[v][:]
                
                for w in candidates_u:
                    for x in candidates_v:
                        if w == x:
                            continue
                        if (min(w,x), max(w,x)) in edge_set:
                            continue
                        # Tentatively add edge
                        adj[w].append(x)
                        adj[x].append(w)
                        edge_set.add((min(w,x), max(w,x)))
                        edges.append((w, x))
                        
                        # Check girth
                        cycle2, clen2 = find_short_cycle(adj, n, g)
                        if cycle2 is None or clen2 >= clen:
                            found = True
                            break
                        else:
                            # Revert
                            adj[w].remove(x)
                            adj[x].remove(w)
                            edge_set.discard((min(w,x), max(w,x)))
                            edges.pop()
                    
                    if found:
                        break
                
                if not found:
                    # Revert the removal
                    adj[u].append(v)
                    adj[v].append(u)
                    edge_set.add((min(u,v), max(u,v)))
                    edges.append((u, v))
            else:
                break
        
        return n, edges
    
    # Run local search with time budget
    remaining = deadline - time.time() - 5
    if remaining > 2:
        n, edges = edge_swap(adj, n, edges, g, remaining)
        best_n, best_edges = n, edges
    
    return best_n, best_edges
