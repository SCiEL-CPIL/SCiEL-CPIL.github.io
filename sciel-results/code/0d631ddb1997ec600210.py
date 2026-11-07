"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_forest(colors, n, edges):
    """Check if the coloring is valid: proper and no bicolored cycles."""
    # Check proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    # Check bicolored forest constraint using union-find per color pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
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

def _try_pattern(n, edges, adj, k, pattern_func):
    """Try a specific coloring pattern."""
    colors = pattern_func(n, k)
    if colors is None:
        return None
    if _check_bicolored_forest(colors, n, edges):
        return colors
    return None

def _mod_pattern(n, k):
    """Color by vertex index mod k."""
    return [i % k for i in range(n)]

def _stripe_pattern(n, k, adj):
    """Color by BFS layer mod k."""
    from collections import deque
    colors = [-1] * n
    visited = [False] * n
    # BFS from vertex 0
    queue = deque([0])
    visited[0] = True
    level = [0] * n
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                level[v] = level[u] + 1
                queue.append(v)
    # Assign colors based on level mod k
    for i in range(n):
        colors[i] = level[i] % k
    return colors

def _row_col_pattern(n, k, adj):
    """Try a 2D grid-like pattern."""
    # For a 5x10 grid, use (row + col) % k
    # Detect grid structure: assume 5 rows, 10 cols
    rows, cols = 5, 10
    colors = [0] * n
    for i in range(n):
        r = i // cols
        c = i % cols
        colors[i] = (r + c) % k
    return colors

def _checkerboard_pattern(n, k, adj):
    """Checkerboard-like pattern for bipartite-like structure."""
    # Try (i // 10 + i % 10) % k for grid
    rows, cols = 5, 10
    colors = [0] * n
    for i in range(n):
        r = i // cols
        c = i % cols
        colors[i] = (2*r + c) % k
    return colors

def _solve_csp(n, edges, adj, k, deadline):
    """
    CSP solver with backtracking, forward checking, and MRV heuristic.
    Uses union-find with rollback for bicolored forest constraint.
    """
    colors = [-1] * n
    
    # Precompute pair indices
    num_pairs = k * (k - 1) // 2
    pair_to_idx = {}
    pair_idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            pair_to_idx[(i, j)] = pair_idx
            pair_idx += 1
    
    class RollbackDSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.history = []
        
        def find(self, x):
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.history.append((ry, rx, self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def rollback(self, count):
            for _ in range(count):
                child, parent, rank_parent_before = self.history.pop()
                self.parent[child] = child
                self.rank[parent] = rank_parent_before
    
    dsus = [RollbackDSU(n) for _ in range(num_pairs)]
    
    def get_pair_idx(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        return pair_to_idx[(c1, c2)]
    
    # Track which colors are used
    color_used = [0] * k
    max_color_used = -1
    
    # Node count for limiting search
    node_count = [0]
    max_nodes = 200000
    
    def try_assign(v, color):
        """Try to assign color to vertex v. Returns list of DSU changes or None."""
        assignments = []
        for nb in adj[v]:
            if colors[nb] == -1:
                continue
            if colors[nb] == color:
                # Undo and return failure
                for (pi, u2, v2) in assignments:
                    dsus[pi].rollback(1)
                return None
            pi = get_pair_idx(color, colors[nb])
            if dsus[pi].union(v, nb):
                assignments.append((pi, v, nb))
            else:
                for (pi2, u2, v2) in assignments:
                    dsus[pi2].rollback(1)
                return None
        return assignments
    
    def undo_assign(assignments):
        for (pi, u, v_edge) in assignments:
            dsus[pi].rollback(1)
    
    def get_mrv_vertex(unassigned):
        """Get vertex with minimum remaining values (MRV heuristic)."""
        best_v = -1
        best_count = k + 1
        for v in unassigned:
            # Count forbidden colors
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] != -1:
                    forbidden.add(colors[nb])
            # Count remaining
            remaining = k - len(forbidden)
            if remaining < best_count:
                best_count = remaining
                best_v = v
                if remaining <= 1:
                    break
        return best_v
    
    def backtrack(unassigned, idx_in_order):
        nonlocal max_color_used, color_used
        if time.time() > deadline:
            return False
        if node_count[0] > max_nodes:
            return False
        node_count[0] += 1
        
        if not unassigned:
            return True
        
        # MRV: pick vertex with fewest remaining colors
        v = get_mrv_vertex(unassigned)
        if v == -1:
            return False
        
        # Get candidate colors
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        
        candidates = []
        for c in range(k):
            if c not in forbidden:
                # Symmetry breaking: only use color if <= max_color_used or == max_color_used+1
                if c <= max_color_used or c == max_color_used + 1:
                    candidates.append(c)
        
        # Order candidates: prefer already-used colors first (to minimize new colors)
        candidates.sort(key=lambda c: (c > max_color_used, c))
        
        new_unassigned = [u for u in unassigned if u != v]
        
        for color in candidates:
            assignments = try_assign(v, color)
            if assignments is None:
                continue
            
            colors[v] = color
            color_used[color] += 1
            old_max = max_color_used
            if color > max_color_used:
                max_color_used = color
            
            if backtrack(new_unassigned, idx_in_order + 1):
                return True
            
            colors[v] = -1
            color_used[color] -= 1
            undo_assign(assignments)
            max_color_used = old_max
        
        return False
    
    # Initial symmetry breaking: force vertex 0 to color 0
    colors[0] = 0
    color_used[0] = 1
    max_color_used = 0
    
    unassigned = list(range(1, n))
    
    if backtrack(unassigned, 0):
        return colors
    return None

def construct():
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(n, edges)
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # First, try simple patterns for small k
    for k in range(3, 8):
        if time.time() > deadline:
            break
        
        # Try various patterns
        patterns = [
            lambda nk: _mod_pattern(n, nk),
            lambda nk: _stripe_pattern(n, nk, adj),
            lambda nk: _row_col_pattern(n, nk, adj),
            lambda nk: _checkerboard_pattern(n, nk, adj),
        ]
        
        found = False
        for pf in patterns:
            if time.time() > deadline:
                break
            colors = pf(k)
            if colors is not None and _check_bicolored_forest(colors, n, edges):
                return [int(c) for c in colors]
        
        # If no pattern worked, try CSP solver
        if time.time() > deadline:
            break
        result = _solve_csp(n, edges, adj, k, deadline)
        if result is not None:
            if _check_bicolored_forest(result, n, edges):
                return [int(c) for c in result]
    
    # Fallback: greedy coloring with many colors
    colors = [0] * n
    for i in range(n):
        used = set()
        for nb in adj[i]:
            if nb < i:
                used.add(colors[nb])
        c = 0
        while c in used:
            c += 1
        colors[i] = c
    return [int(c) for c in colors]
# EVOLVE-BLOCK-END
