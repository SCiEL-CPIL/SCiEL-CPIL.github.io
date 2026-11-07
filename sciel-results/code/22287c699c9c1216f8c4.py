"""Solver for few-color graph coloring with no bicolored cycles."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_valid(coloring, adj):
    """Check adjacent differ and no bicolored cycles. Returns True if valid."""
    n = len(coloring)
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    color_pairs = set()
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            color_pairs.add((min(cu, cv), max(cu, cv)))
    for c1, c2 in color_pairs:
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        for u, v in P['edges']:
            cu, cv = coloring[u], coloring[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    adj = _build_adj()
    n = P['n']
    edge_list = P['edges']
    
    # Try many structured patterns quickly
    patterns = []
    for k in range(2, 10):
        patterns.append([v % k for v in range(n)])
    for k in range(2, 8):
        patterns.append([(v // 10 + v % 10) % k for v in range(n)])
    for k in range(2, 8):
        patterns.append([(v // 5) % k for v in range(n)])
    for k in range(2, 8):
        patterns.append([(v * 7) % k for v in range(n)])
    for a in range(1, 10):
        for k in range(2, 8):
            patterns.append([(a * v) % k for v in range(n)])
    
    for coloring in patterns:
        if time.time() > deadline:
            break
        if _check_valid(coloring, adj):
            return [int(x) for x in coloring]
    
    # Fast backtracking with incremental bicolored cycle check
    # Use degeneracy ordering: process vertices in order that maximizes constraints early
    # For this 4-regular graph, use a simple heuristic
    
    def solve_k(k, time_limit):
        coloring = [-1] * n
        assigned = [False] * n
        # Track for each color pair: union-find state
        # We'll do targeted checks only
        
        # Order: start with vertex 0, then pick most-constrained uncolored
        # Actually, let's try a fixed good order first
        # The graph has structure: vertices 0-9 connect to 10-19 via +7 edges
        # Let's try ordering by "layer"
        
        # Simple DFS with strong pruning
        def dfs(idx, unassigned_count):
            if time.time() > time_limit:
                return None
            if unassigned_count == 0:
                return coloring[:]
            
            # Pick vertex: most colored neighbors among unassigned
            best_v = -1
            best_score = -1
            for v in range(n):
                if not assigned[v]:
                    score = sum(1 for nb in adj[v] if assigned[nb])
                    if score > best_score:
                        best_score = score
                        best_v = v
            
            if best_v == -1:
                return None
            
            # Try colors in order
            used = set()
            for nb in adj[best_v]:
                if assigned[nb]:
                    used.add(coloring[nb])
            
            for c in range(k):
                if c in used:
                    continue
                
                coloring[best_v] = c
                assigned[best_v] = True
                
                # Check if this creates a bicolored cycle
                valid = True
                for nb in adj[best_v]:
                    if not assigned[nb]:
                        continue
                    c_nb = coloring[nb]
                    if c_nb == c:
                        continue
                    c1, c2 = min(c, c_nb), max(c, c_nb)
                    # Check: is there already a path from best_v to nb using only c1/c2 vertices?
                    # If yes, adding edge (best_v, nb) creates a cycle
                    visited = set()
                    stack = [best_v]
                    visited.add(best_v)
                    found = False
                    while stack:
                        u = stack.pop()
                        if u == nb:
                            found = True
                            break
                        for w in adj[u]:
                            if w in visited:
                                continue
                            if not assigned[w]:
                                continue
                            cw = coloring[w]
                            if cw == c1 or cw == c2:
                                visited.add(w)
                                stack.append(w)
                    if found:
                        valid = False
                        break
                
                if valid:
                    result = dfs(idx + 1, unassigned_count - 1)
                    if result is not None:
                        return result
                
                assigned[best_v] = False
                coloring[best_v] = -1
            
            return None
        
        # Initialize: color vertex 0
        coloring[0] = 0
        assigned[0] = True
        return dfs(0, n - 1)
    
    # Try increasing k
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 5:
            break
        result = solve_k(k, time.time() + (deadline - time.time()) * 0.6)
        if result is not None:
            if _check_valid(result, adj):
                return [int(x) for x in result]
    
    # Fallback: 50 colors
    return list(range(n))
