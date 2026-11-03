"""De Bruijn-type finite-state construction for regular graphs of large girth."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 120.0  # seconds; evaluator kills at 150s, keep margin

    if k == 3 and g == 16:
        # Structural change: Cayley graph of a random group with controlled girth
        def build_cayley_3reg(n_target, seed=42):
            """Build a 3-regular Cayley graph on a random group quotient."""
            rng = random.Random(seed)
            # Use group presentation: <a, b | a^2 = b^2 = 1, plus random relations>
            # Generators: a, b, c=ab (all involutions for 3-regular)
            # We build the Cayley graph on Z_n with a specific structure
            # Alternative: use a random regular graph via configuration model but
            # with girth-aware edge selection
            
            # Try multiple random 3-regular graphs with girth checking
            for attempt in range(200):
                if time.time() - t0 > deadline - 10:
                    break
                n = n_target
                if n % 2 == 1 or n < 4:
                    continue
                # Configuration model with girth-aware pairing
                half_edges = [i for i in range(n) for _ in range(3)]
                rng.shuffle(half_edges)
                edges = set()
                valid = True
                for i in range(len(half_edges) // 2):
                    u, v = half_edges[2*i], half_edges[2*i+1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u,v), max(u,v))
                    if key in edges:
                        valid = False
                        break
                    edges.add(key)
                if not valid:
                    continue
                edge_list = list(edges)
                deg = [0]*n
                for u,v in edge_list:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != 3 for d in deg):
                    continue
                # Quick girth check using BFS
                adj = [[] for _ in range(n)]
                for u, v in edge_list:
                    adj[u].append(v)
                    adj[v].append(u)
                girth_ok = True
                max_dist = g // 2
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    qi = 0
                    while qi < len(queue):
                        u = queue[qi]
                        qi += 1
                        if dist[u] >= max_dist:
                            continue
                        du = dist[u]
                        pu = parent[u]
                        for v in adj[u]:
                            if v == pu:
                                continue
                            dv = dist[v]
                            if dv == -1:
                                dist[v] = du + 1
                                parent[v] = u
                                queue.append(v)
                            elif dv >= du:
                                cycle_len = du + dv + 1
                                if cycle_len < g:
                                    girth_ok = False
                                    break
                        if not girth_ok:
                            break
                    if not girth_ok:
                        break
                if girth_ok:
                    return n, edge_list
            return None
        
        # Try to find a valid graph with minimal n
        # Start from a reasonable lower bound and increase
        for n in range(100, 940, 2):
            if time.time() - t0 > deadline - 5:
                break
            result = build_cayley_3reg(n, seed=n)
            if result is not None:
                return result
        
        # Fallback: use the LUW construction
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    else:
        # General case: use LUW construction
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
# EVOLVE-BLOCK-END
