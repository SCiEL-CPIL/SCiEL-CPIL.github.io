"""CSP-based solver with forward checking and MRV heuristic for bicolored-forest coloring."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _validate(coloring, edges):
    """Check adjacency constraint and bicolored forest constraint via union-find per pair."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu < cv:
            pair = (cu, cv)
        else:
            pair = (cv, cu)
        pair_edges[pair].append((u, v))
    for pair, elist in pair_edges.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True


def _pair_has_cycle(coloring, adj, c1, c2):
    """Check if subgraph induced by colors c1,c2 has a cycle using union-find."""
    parent = {}
    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True
    for u in range(len(coloring)):
        if coloring[u] in (c1, c2):
            for v in adj[u]:
                if v > u and coloring[v] in (c1, c2):
                    if not union(u, v):
                        return True
    return False


def _csp_solve(n, edges, adj, k, deadline, time_limit=30):
    """
    CSP solver with:
    - MRV heuristic (pick variable with fewest remaining values)
    - Forward checking (prune neighbor domains)
    - Bicolored cycle constraint propagation
    """
    start = time.time()
    coloring = [-1] * n
    # domains[v] = set of available colors for vertex v
    domains = [set(range(k)) for _ in range(n)]
    
    # Precompute: for each vertex, which neighbors have which colors (dynamic)
    # We'll track assigned neighbors' colors for forward checking
    
    def count_cycles_in_pair(c1, c2, coloring_local):
        """Count if assigning would create a cycle. Returns True if cycle exists."""
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u in range(n):
            if coloring_local[u] in (c1, c2):
                for v in adj[u]:
                    if v > u and coloring_local[v] in (c1, c2):
                        if not union(u, v):
                            return True
        return False
    
    def check_bicolored_ok(v, c, coloring_local):
        """Check if assigning color c to vertex v creates any bicolored cycle."""
        for nb in adj[v]:
            if coloring_local[nb] == -1:
                continue
            d = coloring_local[nb]
            if d == c:
                return False
            c1, c2 = (min(c, d), max(c, d))
            # Temporarily assign and check
            old = coloring_local[v]
            coloring_local[v] = c
            if count_cycles_in_pair(c1, c2, coloring_local):
                coloring_local[v] = old
                return False
            coloring_local[v] = old
        return True
    
    def forward_check(v, c, domains_local, coloring_local):
        """Prune domains of unassigned neighbors. Returns False if domain wipeout."""
        for nb in adj[v]:
            if coloring_local[nb] != -1:
                continue
            # Remove c from nb's domain (adjacency)
            if c in domains_local[nb]:
                domains_local[nb].discard(c)
                if not domains_local[nb]:
                    return False
            # Also check: for each color d in nb's domain, would (c,d) create a cycle?
            # This is expensive; do a lighter check
            for d in list(domains_local[nb]):
                if d == c:
                    continue
                c1, c2 = (min(c, d), max(c, d))
                # Check if adding edge (v,nb) with colors (c,d) creates a cycle
                # We need to check the bicolored subgraph of (c1,c2) including this new edge
                # Simplified: just check if there's already a path between v and nb in the (c1,c2) subgraph
                # This is complex; skip for now and rely on assignment-time check
        return True
    
    def solve(pos, assigned_count):
        if time.time() - start > time_limit:
            return False
        if assigned_count == n:
            return True
        
        # MRV: pick unassigned vertex with smallest domain
        min_size = k + 1
        var = -1
        for v in range(n):
            if coloring[v] == -1:
                sz = len(domains[v])
                if sz < min_size:
                    min_size = sz
                    var = v
                    if sz <= 1:
                        break
                if sz == 0:
                    return False
        
        if var == -1:
            return True
        
        # Degree ordering tiebreak: prefer higher degree
        candidates = [v for v in range(n) if coloring[v] == -1 and len(domains[v]) == min_size]
        candidates.sort(key=lambda x: -len(adj[x]))
        var = candidates[0]
        
        # LCV: try colors that constrain neighbors least (or just iterate)
        colors = list(domains[var])
        # Shuffle for diversity
        random.shuffle(colors)
        
        for c in colors:
            if time.time() - start > time_limit:
                return False
            
            # Check adjacency
            adj_ok = True
            for nb in adj[var]:
                if coloring[nb] == c:
                    adj_ok = False
                    break
            if not adj_ok:
                continue
            
            # Check bicolored cycle
            if not check_bicolored_ok(var, c, coloring):
                continue
            
            # Assign
            coloring[var] = c
            domains[var] = set()  # Mark as assigned
            
            # Forward check
            saved_domains = {}
            fc_ok = True
            for nb in adj[var]:
                if coloring[nb] != -1:
                    continue
                saved_domains[nb] = domains[nb].copy()
                domains[nb].discard(c)
                if not domains[nb]:
                    fc_ok = False
                    break
            
            if fc_ok:
                if solve(pos + 1, assigned_count + 1):
                    return True
            
            # Undo
            coloring[var] = -1
            domains[var] = set(range(k))
            for nb in saved_domains:
                domains[nb] = saved_domains[nb]
        
        return False
    
    return solve(0, 0)


def _csp_solve_v2(n, edges, adj, k, deadline, time_limit=20):
    """
    Alternative CSP: use recursive backtracking with smarter pruning.
    Order vertices by degree, use incremental union-find tracking.
    """
    start = time.time()
    coloring = [-1] * n
    
    # Order by degree descending
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos_of = [0] * n
    for i, v in enumerate(order):
        pos_of[v] = i
    
    # Track bicolored connectivity incrementally
    # For each pair (c1, c2), maintain a DSU of vertices with those colors
    # This is complex; use a simpler approach: just check at assignment time
    
    def check_assignment(v, c):
        """Check if assigning c to v is valid given current partial coloring."""
        for nb in adj[v]:
            if coloring[nb] == c:
                return False
        # Check bicolored cycles involving c
        for nb in adj[v]:
            if coloring[nb] == -1:
                continue
            d = coloring[nb]
            if d == c:
                return False
            c1, c2 = (min(c, d), max(c, d))
            # Check if (v, nb) edge creates a cycle in the (c1, c2) subgraph
            # Need to find if there's a path from v to nb using only vertices colored c1 or c2
            # BFS/DFS in the subgraph
            visited = set()
            stack = [v]
            visited.add(v)
            found_path = False
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if w == nb:
                        found_path = True
                        break
                    if w not in visited and coloring[w] in (c1, c2):
                        visited.add(w)
                        stack.append(w)
                if found_path:
                    break
            if found_path:
                return False
        return True
    
    def backtrack(idx):
        if time.time() - start > time_limit:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        for c in range(k):
            if not check_assignment(v, c):
                continue
            coloring[v] = c
            if backtrack(idx + 1):
                return True
            coloring[v] = -1
        return False
    
    if backtrack(0):
        return coloring
    return None


def _try_pattern(n, k, edges, adj):
    """Try a structured periodic pattern."""
    # For a 5x10 grid, try patterns based on (row, col) mod something
    best = None
    for r_mod in range(1, k + 1):
        for c_mod in range(1, k + 1):
            coloring = []
            for i in range(5):
                for j in range(10):
                    coloring.append((i * r_mod + j * c_mod) % k)
            if _validate(coloring, edges):
                return coloring
    return None


def _local_search_improved(start_coloring, k, n, edges, adj, deadline, max_iter=20000):
    """Improved local search with better violation detection and repair."""
    coloring = start_coloring[:]
    rng = random.Random(123)
    start = time.time()
    
    def count_violations(coloring_local):
        """Count adjacency violations + bicolored cycle count."""
        violations = 0
        for u, v in edges:
            if coloring_local[u] == coloring_local[v]:
                violations += 1
        # Count bicolored cycles (approximate: count pairs that have cycles)
        pair_edges = defaultdict(list)
        for u, v in edges:
            cu, cv = coloring_local[u], coloring_local[v]
            if cu != cv:
                pair = (min(cu, cv), max(cu, cv))
                pair_edges[pair].append((u, v))
        for pair, elist in pair_edges.items():
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                parent[rx] = ry
                return True
            for u, v in elist:
                if not union(u, v):
                    violations += 1
                    break
        return violations
    
    current_violations = count_violations(coloring)
    best_coloring = coloring[:]
    best_violations = current_violations
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if current_violations == 0:
            return coloring
        
        # Pick a random vertex to reassign
        v = rng.randint(0, n - 1)
        old_c = coloring[v]
        
        # Try all colors, pick the one that minimizes violations
        best_c = old_c
        best_v = current_violations
        for c in range(k):
            if c == old_c:
                continue
            # Check adjacency
            adj_ok = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    adj_ok = False
                    break
            if not adj_ok:
                continue
            coloring[v] = c
            new_v = count_violations(coloring)
            if new_v < best_v:
                best_v = new_v
                best_c = c
        coloring[v] = best_c
        
        if best_c != old_c:
            new_violations = count_violations(coloring)
            if new_violations < best_violations:
                best_violations = new_violations
                best_coloring = coloring[:]
            current_violations = new_violations
            if current_violations == 0:
                return coloring
        else:
            # No improvement, keep going
            continue
    
    if _validate(best_coloring, edges):
        return best_coloring
    if best_violations < current_violations:
        return best_coloring
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    # Strategy 1: Try structured patterns with few colors
    for k in [4, 5, 6]:
        if time.time() > deadline - 60:
            break
        result = _try_pattern(n, k, edges, adj)
        if result is not None and _validate(result, edges):
            return [int(c) for c in result]
    
    # Strategy 2: CSP with backtracking (MRV + forward checking)
    for k in [5, 4, 6, 7]:
        if time.time() > deadline - 40:
            break
        tl = min(15, max(5, deadline - time.time() - 25))
        result = _csp_solve(n, edges, adj, k, deadline, time_limit=tl)
        if result and _validate(result, edges):
            return [int(c) for c in result]
    
    # Strategy 3: CSP v2 (simpler backtracking with degree ordering)
    for k in [5, 6, 7]:
        if time.time() > deadline - 25:
            break
        tl = min(10, max(3, deadline - time.time() - 15))
        result = _csp_solve_v2(n, edges, adj, k, deadline, time_limit=tl)
        if result and _validate(result, edges):
            return [int(c) for c in result]
    
    # Strategy 4: Local search from various initial patterns
    patterns = []
    # Random patterns
    rng = random.Random(42)
    for _ in range(10):
        patterns.append([rng.randint(0, 4) for _ in range(n)])
    # Structured patterns
    for r_mod in [1, 2, 3]:
        for c_mod in [1, 2, 3]:
            p = []
            for i in range(5):
                for j in range(10):
                    p.append((i * r_mod + j * c_mod) % 5)
            patterns.append(p)
    
    for pat in patterns:
        if time.time() > deadline - 10:
            break
        result = _local_search_improved(pat, 5, n, edges, adj, deadline, max_iter=5000)
        if result is not None and _validate(result, edges):
            return [int(c) for c in result]
    
    # Fallback: valid coloring with many colors
    # Use a proper coloring: color by vertex index mod something
    # For a 4-regular graph, 5 colors suffice for proper coloring
    coloring = [i % 5 for i in range(n)]
    if not _validate(coloring, edges):
        # Use more colors
        coloring = [i for i in range(n)]
    
    return [int(c) for c in coloring]
# EVOLVE-BLOCK-END
