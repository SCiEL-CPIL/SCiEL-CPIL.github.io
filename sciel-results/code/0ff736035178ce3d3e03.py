import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def get_luw():
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

    luw_n, luw_edges = get_luw()
    best_n = luw_n
    best_edges = luw_edges

    # Heuristic Search
    def get_girth(adj, n, max_depth):
        # Returns the girth (shortest cycle length)
        # If no cycle found within max_depth, returns infinity
        for u in range(n):
            # BFS from u
            depth = [-1] * n
            parent = [-1] * n
            queue = [u]
            depth[u] = 0
            idx = 0
            while idx < len(queue):
                v = queue[idx]
                idx += 1
                if depth[v] >= max_depth:
                    continue
                for w in adj[v]:
                    if depth[w] == -1:
                        depth[w] = depth[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v]:
                        # Cycle found
                        cycle_len = depth[v] + depth[w] + 1
                        if cycle_len < g:
                            return cycle_len
        return float('inf')

    def generate_random_regular(n, k):
        # Simple configuration model with retries
        for _ in range(10):
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                # Check degrees
                ok = True
                for i in range(n):
                    if len(adj[i]) != k:
                        ok = False
                        break
                if ok:
                    return adj
        return None

    def anneal_search(n, k, g, time_limit):
        limit_time = time.time() + time_limit
        adj = generate_random_regular(n, k)
        if adj is None:
            return None
        
        # Convert to list of lists for faster access? Sets are fine.
        
        current_girth = get_girth(adj, n, g//2)
        if current_girth >= g:
            # Found a valid graph
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        
        # Annealing
        T = 1.0
        cooling = 0.999
        while time.time() < limit_time:
            # Pick a random edge (u, v) and another (x, y)
            # Try to swap to (u, x) and (v, y)
            # We need to pick 4 distinct vertices u, v, x, y such that (u,v) and (x,y) are edges
            # And (u,x) and (v,y) are NOT edges (and not loops)
            
            # To do this efficiently, we can pick a random edge (u,v)
            # Then pick a random neighbor x of u (x != v)
            # Then pick a random neighbor y of v (y != u)
            # Check if (x,y) is an edge. If so, we can swap (u,v) and (x,y) with (u,x) and (v,y)?
            # No, that's not a valid 2-switch that preserves regularity unless we are careful.
            # A 2-switch: Remove (u,v) and (x,y), add (u,x) and (v,y).
            # This preserves degrees.
            # We need (u,v) and (x,y) to be edges.
            # And (u,x) and (v,y) to be non-edges.
            
            # Let's pick a random edge (u,v).
            u = random.randint(0, n-1)
            if not adj[u]: continue
            v = random.choice(list(adj[u]))
            
            # Pick x from adj[u] \ {v}
            neighbors_u = list(adj[u])
            if len(neighbors_u) <= 1: continue
            neighbors_u.remove(v)
            x = random.choice(neighbors_u)
            
            # Pick y from adj[v] \ {u}
            neighbors_v = list(adj[v])
            if len(neighbors_v) <= 1: continue
            neighbors_v.remove(u)
            y = random.choice(neighbors_v)
            
            # Check if (x,y) is an edge
            if y in adj[x]:
                # Check if (u,x) and (v,y) are non-edges
                # (u,x) is an edge (we picked x from adj[u])
                # So we can't swap (u,v) and (x,y) to (u,x) and (v,y) because (u,x) already exists.
                # We need to find a 4-cycle u-v-y-x-u to break?
                # If (u,v), (v,y), (y,x), (x,u) are edges, we have a 4-cycle.
                # Breaking a 4-cycle: Remove (u,v) and (x,y), add (u,y) and (v,x).
                # Check if (u,y) and (v,x) are non-edges.
                if y not in adj[u] and x not in adj[v]:
                    # Perform swap
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].add(y)
                    adj[y].add(u)
                    adj[v].add(x)
                    adj[x].add(v)
                    
                    # Check girth
                    new_girth = get_girth(adj, n, g//2)
                    if new_girth >= g:
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return n, edges
                    
                    # Update current_girth for acceptance criteria
                    # If new_girth > current_girth, accept.
                    # Else accept with prob exp(-(new_girth - current_girth)/T) ?
                    # Actually, we want to MAXIMIZE girth.
                    # So if new_girth > current_girth, accept.
                    # If new_girth < current_girth, reject (or accept with prob).
                    if new_girth > current_girth:
                        current_girth = new_girth
                    elif new_girth == current_girth:
                        pass
                    else:
                        # Revert
                        adj[u].remove(y)
                        adj[y].remove(u)
                        adj[v].remove(x)
                        adj[x].remove(v)
                        
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[x].add(y)
                        adj[y].add(x)
            
            T *= cooling

    # Try to find a smaller graph
    # Start with n around 500
    for n_try in [500, 600, 800]:
        if time.time() > deadline - 10:
            break
        result = anneal_search(n_try, k, g, 10.0)
        if result:
            best_n, best_edges = result
            break

    return best_n, best_edges
