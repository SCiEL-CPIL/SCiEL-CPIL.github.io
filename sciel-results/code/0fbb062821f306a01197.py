"""De Bruijn-type finite-state construction for regular graphs of large girth."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 120.0  # seconds; evaluator kills at 150s, keep margin

    # If the task is the specific (3, 16) cage problem, use optimized random search
    if k == 3 and g == 16:
        random.seed(42)
        
        def check_girth_fast(n, edges, min_girth):
            """Check if graph has girth >= min_girth using BFS from each vertex."""
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            max_dist = min_girth // 2
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
                            if cycle_len < min_girth:
                                return False
            return True
        
        def generate_random_3_reg(n):
            """Generate a random 3-regular graph on n vertices using configuration model."""
            if n < 2 or n % 2 == 1:
                return None
            half_edges = [i for i in range(n) for _ in range(3)]
            random.shuffle(half_edges)
            edges = set()
            for i in range(len(half_edges) // 2):
                u, v = half_edges[2*i], half_edges[2*i+1]
                if u == v:
                    continue
                key = (min(u,v), max(u,v))
                if key in edges:
                    continue
                edges.add(key)
            edge_list = list(edges)
            deg = [0]*n
            for u,v in edge_list:
                deg[u] += 1
                deg[v] += 1
            if any(d != 3 for d in deg):
                return None
            return edge_list
        
        # Theoretical lower bound for girth g in 3-regular graph is approx 2 log_3(n)
        # For g=16, n >= 3^(16/2) = 3^8 = 6561.
        # We search from 2200 (conservative lower bound) up to 20000.
        # We step by 50 to balance search granularity and time.
        
        # If we find a graph, we return it.
        # We want the smallest n.
        
        found = False
        for n in range(2200, 20000, 50):
            if n % 2 != 0:
                n += 1
            if time.time() - t0 > deadline - 10:
                break
                
            # Try a few random instances for this n
            # If n is close to the bound, it might take more attempts.
            attempts = 10 if n < 8000 else 5
            for _ in range(attempts):
                edges = generate_random_3_reg(n)
                if edges is None:
                    continue
                if check_girth_fast(n, edges, g):
                    return n, edges
        
        # Fallback: If random search fails, use LUW construction
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
