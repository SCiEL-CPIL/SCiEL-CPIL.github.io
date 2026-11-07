"""Minimize distinct colors with bicolored-cycle-free constraint on degree-4 graph."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = [[] for _ in range(N)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)


def _check_valid(coloring, k):
    """Check proper coloring and no bicolored cycles."""
    # Proper coloring
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Bicolored cycle check
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            parent = list(range(N))
            rank = [0] * N
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return False
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if union(u, v):
                        return False
    return True


def _try_k_coloring(k, deadline):
    """Try to find a valid k-coloring using backtracking with pruning."""
    coloring = [-1] * N
    # Order vertices by degree descending, then by index
    order = sorted(range(N), key=lambda v: (-len(adj[v]), v))
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == N:
            return True
        v = order[idx]
        # Get colors used by already-colored neighbors
        used = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        
        for c in range(k):
            if c in used:
                continue
            coloring[v] = c
            # Quick check: for each colored neighbor, verify no bicolored cycle
            ok = True
            for nb in adj[v]:
                if coloring[nb] != -1:
                    cn = coloring[nb]
                    # Check if adding edge (v,nb) creates a cycle in c1-c2 subgraph
                    parent = list(range(N))
                    rank = [0] * N
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    def union(x, y):
                        rx, ry = find(x), find(y)
                        if rx == ry:
                            return True
                        if rank[rx] < rank[ry]:
                            rx, ry = ry, rx
                        parent[ry] = rx
                        if rank[rx] == rank[ry]:
                            rank[rx] += 1
                        return False
                    for u2, v2 in edges:
                        cu2, cv2 = coloring[u2], coloring[v2]
                        if (cu2 == c and cv2 == cn) or (cu2 == cn and cv2 == c):
                            if u2 != -1 and v2 != -1 and coloring[u2] != -1 and coloring[v2] != -1:
                                if union(u2, v2):
                                    ok = False
                                    break
                    if not ok:
                        break
            if ok:
                if backtrack(idx + 1):
                    return True
            coloring[v] = -1
        return False
    
    if backtrack(0):
        return coloring[:]
    return None


def _greedy_with_repair(k, deadline):
    """Greedy coloring followed by local repair."""
    coloring = [0] * N
    # Initial greedy
    for v in range(N):
        used = set(coloring[nb] for nb in adj[v])
        for c in range(k):
            if c not in used:
                coloring[v] = c
                break
    
    # Local search to fix bicolored cycles
    best = coloring[:]
    best_cost = float('inf')
    
    def cost(col):
        # Count proper violations + bicolored cycle violations
        proper_v = 0
        for u, v in edges:
            if col[u] == col[v]:
                proper_v += 1
        if proper_v > 0:
            return proper_v * 1000
        cyc_v = 0
        used_colors = set(col)
        clist = list(used_colors)
        for i in range(len(clist)):
            for j in range(i + 1, len(clist)):
                a, b = clist[i], clist[j]
                parent = list(range(N))
                rank = [0] * N
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return True
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return False
                for u, v in edges:
                    cu, cv = col[u], col[v]
                    if (cu == a and cv == b) or (cu == b and cv == a):
                        if union(u, v):
                            cyc_v += 1
                            break
        return cyc_v
    
    current_cost = cost(coloring)
    if current_cost == 0:
        return coloring
    
    no_improve = 0
    while time.time() < deadline and no_improve < 200:
        v = (hash(time.time()) % N)
        old = coloring[v]
        best_c = old
        best_c_cost = current_cost
        for c in range(k):
            if c == old:
                continue
            coloring[v] = c
            cc = cost(coloring)
            if cc < best_c_cost:
                best_c_cost = cc
                best_c = c
        coloring[v] = best_c
        if best_c_cost < current_cost:
            current_cost = best_c_cost
            no_improve = 0
            if current_cost == 0:
                return coloring
            if current_cost < best_cost:
                best_cost = current_cost
                best = coloring[:]
        else:
            no_improve += 1
    
    if best_cost == 0:
        return best
    return None


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s
    
    # Try small k values with backtracking first
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
        remaining = deadline - time.time() - 2.0
        if remaining < 1.0:
            break
        result = _try_k_coloring(k, start + remaining)
        if result is not None:
            if _check_valid(result, k):
                return [int(c) for c in result]
    
    # Try local search for k=4,5,6
    for k in [4, 5, 6]:
        if time.time() > deadline:
            break
        remaining = deadline - time.time() - 2.0
        if remaining < 1.0:
            break
        result = _greedy_with_repair(k, start + remaining)
        if result is not None:
            if _check_valid(result, k):
                return [int(c) for c in result]
    
    # Fallback: use many colors to guarantee validity
    coloring = list(range(N))
    return [int(c) for c in coloring]
