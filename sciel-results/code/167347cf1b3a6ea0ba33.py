import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    sys.setrecursionlimit(100000)
    
    # Latin Square of order 6
    L = [
        [0, 1, 2, 3, 4, 5],
        [3, 4, 5, 0, 1, 2],
        [1, 2, 3, 4, 5, 0],
        [4, 5, 0, 1, 2, 3],
        [2, 3, 4, 5, 0, 1],
        [5, 0, 1, 2, 3, 4]
    ]
    
    k = 6
    n_target = 35
    graph = [set() for _ in range(n_target)]
    
    # Build graph based on Latin Square
    # Vertices (i, j) for i, j in 0..5
    # Edge if i != i', j != j' and L[i][j] == L[i'][j']
    for i in range(k):
        for j in range(k):
            u = i * k + j
            if u >= n_target: continue
            for i2 in range(k):
                if i2 == i: continue
                for j2 in range(k):
                    if j2 == j: continue
                    v = i2 * k + j2
                    if v >= n_target: continue
                    if L[i][j] == L[i2][j2]:
                        graph[u].add(v)
                        graph[v].add(u)
                        
    # Function to find a C5
    def find_c5(g, n):
        for u in range(n):
            for v in g[u]:
                if v <= u: continue
                for w in g[v]:
                    if w == u or w == v: continue
                    for x in g[w]:
                        if x == u or x == v or x == w: continue
                        for y in g[x]:
                            if y == u or y == v or y == w or y == x: continue
                            if u in g[y]:
                                return [u, v, w, x, y]
        return None

    # Iteratively remove edges to break C5s
    for _ in range(100):
        if time.time() > deadline:
            break
        c5 = find_c5(graph, n_target)
        if c5 is None:
            break
        u, v = c5[0], c5[1]
        graph[u].discard(v)
        graph[v].discard(u)
        
    # Check Alpha
    def has_independent_set_size_9(g, n):
        neighbor_mask = [0] * n
        for u in range(n):
            for v in g[u]:
                neighbor_mask[u] |= (1 << v)
        
        all_mask = (1 << n) - 1
        
        def dfs(available, size):
            if size >= 9:
                return True
            if available == 0:
                return False
            
            if size + bin(available).count('1') < 9:
                return False
            
            v = (available & -available).bit_length() - 1
            rest = available & ~(1 << v)
            
            new_available = rest & ~neighbor_mask[v]
            if dfs(new_available, size + 1):
                return True
            
            if dfs(rest, size):
                return True
            
            return False
        
        return dfs(all_mask, 0)

    n = n_target
    if has_independent_set_size_9(graph, n):
        # Fallback to n=32
        n = 32
        graph = [set() for _ in range(n)]
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    u = base + i
                    v = base + j
                    graph[u].add(v)
                    graph[v].add(u)
    else:
        n = n_target

    # Final Verification
    def strict_has_c5(g, n):
        for u in range(n):
            nu = g[u]
            for v in nu:
                if v <= u: continue
                nv = g[v]
                for w in nv:
                    if w == u: continue
                    nw = g[w]
                    for x in nw:
                        if x == u or x == v: continue
                        nx = g[x]
                        for y in nx:
                            if y == u or y == v or y == w: continue
                            if u in g[y]:
                                return True
        return False

    def strict_alpha_at_least_9(g, n):
        neighbor_mask = [0] * n
        for u in range(n):
            for v in g[u]:
                neighbor_mask[u] |= (1 << v)
        
        all_mask = (1 << n) - 1
        
        def dfs(available, size):
            if size >= 9:
                return True
            if available == 0:
                return False
            
            if size + bin(available).count('1') < 9:
                return False
            
            v = (available & -available).bit_length() - 1
            rest = available & ~(1 << v)
            
            new_available = rest & ~neighbor_mask[v]
            if dfs(new_available, size + 1):
                return True
            
            if dfs(rest, size):
                return True
            
            return False
        
        return dfs(all_mask, 0)

    if strict_has_c5(graph, n) or strict_alpha_at_least_9(graph, n):
        # Fallback to 32
        n = 32
        graph = [set() for _ in range(n)]
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    u = base + i
                    v = base + j
                    graph[u].add(v)
                    graph[v].add(u)

    edges = []
    for u in range(n):
        for v in graph[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
