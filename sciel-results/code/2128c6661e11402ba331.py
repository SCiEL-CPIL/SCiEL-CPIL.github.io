"""Improved CSP-based search for minimum-color valid coloring."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
    """Check if coloring is valid: adjacent differ and no bicolored cycle."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Check bicolored cycles using union-find for each color pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _try_pattern_k3(adj, n):
    """Try structured 3-coloring patterns for the 5x10 cylindrical grid."""
    patterns = []
    # All affine patterns: (a*r + b*c + c0) % 3
    for a in range(3):
        for b in range(3):
            for c0 in range(3):
                patterns.append((a, b, c0))
    # Additional non-affine patterns
    extra_patterns = [
        lambda r, c: (r + c + (r * c) % 3) % 3,
        lambda r, c: (r + c + ((r + c) // 3) % 3) % 3,
        lambda r, c: (r * 2 + c * 2 + (r + c) % 2) % 3,
    ]
    
    for a, b, c0 in patterns:
        coloring = [(a * (i // 10) + b * (i % 10) + c0) % 3 for i in range(n)]
        if _check_coloring(coloring, P['edges']):
            return coloring
    
    for pat in extra_patterns:
        coloring = [pat(i // 10, i % 10) for i in range(n)]
        if _check_coloring(coloring, P['edges']):
            return coloring
    
    return None

def _csp_search_k(k, adj, n, deadline):
    """CSP backtracking with MRV heuristic and forward checking."""
    # Vertex ordering: by degree descending, then index
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    pos_of = [0] * n
    for i, v in enumerate(order):
        pos_of[v] = i
    
    coloring = [-1] * n
    domains = [list(range(k)) for _ in range(n)]
    
    # Precompute which vertices are neighbors
    def get_mrv_vertex(start_pos):
        """Find unassigned vertex with minimum remaining values."""
        best_v = -1
        best_domain_size = k + 1
        for j in range(start_pos, n):
            v = order[j]
            if coloring[v] != -1:
                continue
            # Compute remaining domain
            used_by_neighbors = set()
            for nb in adj[v]:
                if coloring[nb] != -1:
                    used_by_neighbors.add(coloring[nb])
            remaining = [c for c in range(k) if c not in used_by_neighbors]
            if len(remaining) < best_domain_size:
                best_domain_size = len(remaining)
                best_v = v
                if best_domain_size <= 1:
                    break
        return best_v
    
    def check_bicolor_cycle_after_assign(v, c):
        """Check if assigning color c to v creates a bicolored cycle."""
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                if cn == c:
                    return False
                continue
            # Check if there's a path from v to nb using only colors {c, cn}
            # that doesn't use edge (v, nb)
            visited = {v}
            stack = [v]
            while stack:
                cur = stack.pop()
                for w in adj[cur]:
                    if w in visited:
                        continue
                    cw = coloring[w]
                    if cw != c and cw != cn:
                        continue
                    # Skip the direct edge (v, nb)
                    if (cur == v and w == nb) or (cur == nb and w == v):
                        continue
                    if w == nb:
                        return False  # Found a cycle
                    visited.add(w)
                    stack.append(w)
        return True
    
    def forward_check(v, c, current_pos):
        """Check if any unassigned vertex has empty domain."""
        for j in range(current_pos, n):
            w = order[j]
            if coloring[w] != -1:
                continue
            used = set()
            for nb in adj[w]:
                if coloring[nb] != -1:
                    used.add(coloring[nb])
            if c == coloring[v]:  # v is now assigned
                used.add(c)
            # Check if w has at least one available color
            has_color = False
            for cc in range(k):
                if cc not in used:
                    has_color = True
                    break
            if not has_color:
                return False
        return True
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        # Find MRV vertex among unassigned
        v = -1
        best_domain = None
        best_size = k + 1
        
        for j in range(n):
            cand = order[j]
            if coloring[cand] != -1:
                continue
            used = set()
            for nb in adj[cand]:
                if coloring[nb] != -1:
                    used.add(coloring[nb])
            remaining = [c for c in range(k) if c not in used]
            if len(remaining) < best_size:
                best_size = len(remaining)
                v = cand
                best_domain = remaining
                if best_size <= 1:
                    break
        
        if v == -1:
            return True  # All assigned
        
        if best_size == 0:
            return False  # No available color
        
        for c in best_domain:
            coloring[v] = c
            if check_bicolor_cycle_after_assign(v, c):
                if forward_check(v, c, 0):
                    if backtrack():
                        return True
            coloring[v] = -1
        
        return False
    
    if backtrack():
        return coloring[:]
    return None

def _local_search_reduce(coloring, adj, n, deadline, target_k):
    """Try to reduce number of colors to target_k via local search."""
    coloring = coloring[:]
    
    # Map colors to 0..target_k-1
    distinct = sorted(set(coloring))
    if len(distinct) <= target_k:
        return coloring if _check_coloring(coloring, P['edges']) else None
    
    color_map = {c: i % target_k for i, c in enumerate(distinct)}
    coloring = [color_map[c] for c in coloring]
    
    import random
    random.seed(42)
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        if _check_coloring(coloring, P['edges']):
            if len(set(coloring)) <= target_k:
                return coloring
            # Try to merge two colors
            colors_used = sorted(set(coloring))
            if len(colors_used) > target_k:
                from collections import Counter
                cnt = Counter(coloring)
                c1 = min(colors_used, key=lambda c: cnt[c])
                c2 = min((c for c in colors_used if c != c1), key=lambda c: cnt[c])
                new_coloring = [c1 if c == c2 else c for c in coloring]
                if _check_coloring(new_coloring, P['edges']):
                    coloring = new_coloring
                else:
                    new_coloring = [c2 if c == c1 else c for c in coloring]
                    if _check_coloring(new_coloring, P['edges']):
                        coloring = new_coloring
        else:
            # Find a violation and fix it
            v = random.randint(0, n - 1)
            old_c = coloring[v]
            for new_c in range(target_k):
                if new_c == old_c:
                    continue
                ok = True
                for nb in adj[v]:
                    if coloring[nb] == new_c:
                        ok = False
                        break
                if not ok:
                    continue
                coloring[v] = new_c
                if _check_coloring(coloring, P['edges']):
                    break
                coloring[v] = old_c
    
    return coloring if _check_coloring(coloring, P['edges']) else None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(edges, n)
    
    # First try structured 3-coloring patterns (very fast)
    result = _try_pattern_k3(adj, n)
    if result is not None:
        if _check_coloring(result, edges):
            return [int(c) for c in result]
    
    # Try CSP search for k=3
    if time.time() < deadline - 20:
        result = _csp_search_k(3, adj, n, deadline - 20)
        if result is not None:
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Try CSP search for k=4
    if time.time() < deadline - 5:
        result = _csp_search_k(4, adj, n, deadline - 5)
        if result is not None:
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Fallback: greedy coloring
    coloring = [0] * n
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    for v in order:
        used = set()
        for nb in adj[v]:
            used.add(coloring[nb])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    
    # Try to reduce
    result = _local_search_reduce(coloring, adj, n, deadline, 4)
    if result is not None:
        return [int(c) for c in result]
    
    return [int(c) for c in coloring]
