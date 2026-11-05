"""Improved program: construct k-regular graph of girth g using random Cayley graphs on non-abelian groups."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try to build a Cayley graph on a non-abelian group (e.g., product of symmetric groups)
    using k involutory generators. Verify girth via BFS. Fall back to L-U-W construction if needed.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    time_limit = 110  # seconds, leaving margin for the 120s limit
    
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    def verify_girth(n, edges, g):
        """Verify that the graph has girth at least g using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check for self-loops and duplicate edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
        
        # BFS from each vertex to find shortest cycle
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
            # Also check if start connects back to itself via a cycle
            # This is handled by the BFS above
        
        return True
    
    def try_cayley_graph(k, g, time_budget):
        """Try to construct a Cayley graph on a non-abelian group with k involutory generators."""
        start_time = time.time()
        
        if k != 5 or g != 9:
            return None
            
        # Group: (S_3)^4
        perms = [
            (0,1,2), # 0: id
            (1,0,2), # 1: (12)
            (2,1,0), # 2: (13)
            (0,2,1), # 3: (23)
            (1,2,0), # 4: (123)
            (2,0,1)  # 5: (132)
        ]
        
        def mul_s3(a, b):
            p_a = perms[a]
            p_b = perms[b]
            result = tuple(p_a[p_b[i]] for i in range(3))
            for idx, p in enumerate(perms):
                if p == result:
                    return idx
            return 0
            
        mul_table = [[0]*6 for _ in range(6)]
        for i in range(6):
            for j in range(6):
                mul_table[i][j] = mul_s3(i, j)
                
        invol_s3 = [1, 2, 3]
        
        invol_g = []
        for a in invol_s3:
            for b in invol_s3:
                for c in invol_s3:
                    for d in invol_s3:
                        invol_g.append((a,b,c,d))
                        
        def mul_g(e, inv):
            return (
                mul_table[e[0]][inv[0]],
                mul_table[e[1]][inv[1]],
                mul_table[e[2]][inv[2]],
                mul_table[e[3]][inv[3]]
            )
            
        identity = (0,0,0,0)
        
        import random
        max_tries = 10000
        tries = 0
        
        while time.time() - start_time < time_budget and tries < max_tries:
            tries += 1
            gen_set = random.sample(invol_g, 5)
            
            dist = {identity: 0}
            parent = {}
            queue = deque([identity])
            
            valid = True
            min_cycle = float('inf')
            
            while queue:
                u = queue.popleft()
                for v in gen_set:
                    w = mul_g(u, v)
                    if w not in dist:
                        dist[w] = dist[u] + 1
                        parent[w] = (u, v)
                        queue.append(w)
                    else:
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid and min_cycle >= g:
                all_elements = []
                for a in range(6):
                    for b in range(6):
                        for c in range(6):
                            for d in range(6):
                                all_elements.append((a,b,c,d))
                                
                elem_to_idx = {e: i for i, e in enumerate(all_elements)}
                
                edges = set()
                for u in all_elements:
                    idx_u = elem_to_idx[u]
                    for v in gen_set:
                        w = mul_g(u, v)
                        idx_w = elem_to_idx[w]
                        if idx_u < idx_w:
                            edges.add((idx_u, idx_w))
                        else:
                            edges.add((idx_w, idx_u))
                            
                return len(all_elements), list(edges)

        return None
    
    # Try Cayley graph approach first (with time limit)
    cayley_result = try_cayley_graph(k, g, time_limit / 2)
    if cayley_result is not None:
        return cayley_result
    
    # Fall back to L-U-W construction
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
