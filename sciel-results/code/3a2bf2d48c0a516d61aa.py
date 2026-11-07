"""Combined approach: exploit graph structure + incremental union-find backtracking + local search.
The graph is a 2-row grid: top row 0-24, bottom row 25-49, with specific edge patterns."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
adj_set = [set() for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)
    adj_set[u].add(v)
    adj_set[v].add(u)

# Graph structure: 2 rows of 25 vertices each
# Top: 0..24, Bottom: 25..49
# Key insight: try to find a small set of colors that works via structured search


def _check_coloring(coloring, edges, adj):
    """Validate: adjacent differ, and each color-pair subgraph is a forest."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Union-find per color pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
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
                return False
    return True


def _fast_check(coloring, edges):
    """Fast validation using only edges list."""
    return _fast_check_v2(coloring, edges, adj)


class UnionFind:
    __slots__ = ('parent', 'rank', 'n')
    def __init__(self, n=0):
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        r = self.rank
        if r[rx] < r[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if r[rx] == r[ry]:
            r[rx] += 1
        return True
    def clone(self):
        c = UnionFind(self.n)
        c.parent = self.parent[:]
        c.rank = self.rank[:]
        return c


def _fast_check_v2(coloring, edges, adj):
    """Optimized validation: adjacency check + per-pair forest check with small UF."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Group edges by color pair
    pair_edges = {}
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            key = (cu, cv) if cu < cv else (cv, cu)
            if key not in pair_edges:
                pair_edges[key] = []
            pair_edges[key].append((u, v))
    # For each pair, check if the induced subgraph is a forest
    # A graph with V vertices and E edges is a forest iff E <= V - components
    # Equivalently, no cycle. Use DSU but only on vertices that appear.
    for key, elist in pair_edges.items():
        # Collect unique vertices
        verts = set()
        for u, v in elist:
            verts.add(u)
            verts.add(v)
        nv = len(verts)
        if len(elist) >= nv:
            # Has a cycle (edges >= vertices means at least one cycle in the subgraph)
            # Actually need to be careful: a forest on nv vertices has at most nv-1 edges
            # But the subgraph might not be connected. Let's just do DSU.
            pass
        # DSU on the small vertex set
        vlist = list(verts)
        vmap = {v: i for i, v in enumerate(vlist)}
        parent = list(range(nv))
        rank = [0] * nv
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in elist:
            ru, rv = find(vmap[u]), find(vmap[v])
            if ru == rv:
                return False
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
    return True


def _find_coloring(k, edges, n, adj, deadline, order=None):
    """Backtracking with incremental per-pair union-find and MRV heuristic."""
    if order is None:
        order = list(range(n))
    
    coloring = [-1] * n
    pair_uf = {}
    colored_count = [0]  # mutable counter for timing checks
    
    def try_assign(v, c):
        """Try to assign color c to vertex v. Returns (success, affected_keys, snapshots)."""
        # Check adjacency
        for nb in adj[v]:
            if coloring[nb] == c:
                return False, [], {}
        
        # Collect affected pairs
        affected_keys = []
        for nb in adj[v]:
            cn = coloring[nb]
            if cn != -1 and cn != c:
                key = (min(cn, c), max(cn, c))
                if key not in affected_keys:
                    affected_keys.append(key)
        
        # Snapshot
        snapshots = {}
        for key in affected_keys:
            if key in pair_uf:
                snapshots[key] = (pair_uf[key].parent[:], pair_uf[key].rank[:])
        
        # Try unions
        valid = True
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            key = (min(cn, c), max(cn, c))
            if key not in pair_uf:
                pair_uf[key] = UnionFind(n)
            uf = pair_uf[key]
            if not uf.union(v, nb):
                valid = False
                break
        
        if not valid:
            return False, [], {}
        
        return True, affected_keys, snapshots
    
    def undo_assign(v, c, affected_keys, snapshots):
        """Undo assignment of color c to vertex v."""
        coloring[v] = -1
        for key in affected_keys:
            if key in snapshots:
                pair_uf[key].parent = snapshots[key][0]
                pair_uf[key].rank = snapshots[key][1]
            elif key in pair_uf:
                del pair_uf[key]
    
    def backtrack(pos, deadline):
        if time.time() > deadline:
            return None
        if pos == n:
            return coloring[:]
        
        v = order[pos]
        
        # Determine available colors
        forbidden = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                forbidden.add(coloring[nb])
        
        # Try colors in a smart order: prefer colors already used (to minimize new colors)
        used_colors = set(coloring)
        available = [c for c in range(k) if c not in forbidden]
        # Sort: prefer already-used colors first
        available.sort(key=lambda c: (c not in used_colors, c))
        
        for c in available:
            success, affected_keys, snapshots = try_assign(v, c)
            if success:
                coloring[v] = c
                result = backtrack(pos + 1, deadline)
                if result is not None:
                    return result
                undo_assign(v, c, affected_keys, snapshots)
            
            if time.time() > deadline:
                return None
        
        return None
    
    result = backtrack(0, deadline)
    return result


def _structured_search(k, edges, n, adj, deadline):
    """Exploit the 2-row structure: assign colors to pairs (i, i+25) together."""
    coloring = [-1] * n
    pair_uf = {}
    
    def try_assign(v, c):
        for nb in adj[v]:
            if coloring[nb] == c:
                return False, [], {}
        affected_keys = []
        for nb in adj[v]:
            cn = coloring[nb]
            if cn != -1 and cn != c:
                key = (min(cn, c), max(cn, c))
                if key not in affected_keys:
                    affected_keys.append(key)
        snapshots = {}
        for key in affected_keys:
            if key in pair_uf:
                snapshots[key] = (pair_uf[key].parent[:], pair_uf[key].rank[:])
        valid = True
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            key = (min(cn, c), max(cn, c))
            if key not in pair_uf:
                pair_uf[key] = UnionFind(n)
            if not pair_uf[key].union(v, nb):
                valid = False
                break
        if not valid:
            return False, [], {}
        return True, affected_keys, snapshots
    
    def undo_assign(v, c, affected_keys, snapshots):
        coloring[v] = -1
        for key in affected_keys:
            if key in snapshots:
                pair_uf[key].parent = snapshots[key][0]
                pair_uf[key].rank = snapshots[key][1]
            elif key in pair_uf:
                del pair_uf[key]
    
    # Process column by column: for each i in 0..24, assign (i, i+25)
    def backtrack(col, deadline):
        if time.time() > deadline:
            return None
        if col == 25:
            return coloring[:]
        
        v_top = col
        v_bot = col + 25
        
        # Try all valid color pairs for (v_top, v_bot)
        # First determine what colors are available for each
        forbidden_top = set()
        for nb in adj[v_top]:
            if coloring[nb] != -1:
                forbidden_top.add(coloring[nb])
        
        forbidden_bot = set()
        for nb in adj[v_bot]:
            if coloring[nb] != -1:
                forbidden_bot.add(coloring[nb])
        
        # They can share a color only if not adjacent (they're not directly adjacent in most cases)
        # Actually check: are col and col+25 adjacent?
        # From edges: [0,49],[6,49],[42,49],[43,44],[44,45]... let me check
        # Edge [i, i+25] exists for some i. Let's just check adj.
        adjacent = v_bot in adj_set[v_top]
        
        for ct in range(k):
            if ct in forbidden_top:
                continue
            for cb in range(k):
                if cb in forbidden_bot:
                    continue
                if adjacent and ct == cb:
                    continue
                
                # Try assigning both
                ok1, ak1, sn1 = try_assign(v_top, ct)
                if not ok1:
                    continue
                coloring[v_top] = ct
                
                ok2, ak2, sn2 = try_assign(v_bot, cb)
                if not ok2:
                    undo_assign(v_top, ct, ak1, sn1)
                    continue
                coloring[v_bot] = cb
                
                result = backtrack(col + 1, deadline)
                if result is not None:
                    return result
                
                undo_assign(v_bot, cb, ak2, sn2)
                undo_assign(v_top, ct, ak1, sn1)
                
                if time.time() > deadline:
                    return None
        
        return None
    
    return backtrack(0, deadline)


def _local_search_reduce(colors, k, adj, edges, deadline):
    """Try to reduce the number of colors by recoloring vertices with incremental checks."""
    n = len(colors)
    current = list(colors)
    current_k = max(current) + 1
    
    # Try to eliminate colors from highest to lowest
    for target_color in range(current_k - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # Find vertices with this color
        verts_to_recolor = [i for i in range(n) if current[i] == target_color]
        if not verts_to_recolor:
            continue
        
        # Try to recolor each vertex to a lower color
        success = True
        new_current = list(current)
        for v in verts_to_recolor:
            recolored = False
            for c in range(target_color):
                # Check if c is valid for v (adjacency)
                valid = True
                for nb in adj[v]:
                    if new_current[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_current[v] = c
                # Fast validation
                if _fast_check(new_current, edges):
                    recolored = True
                    break
                new_current[v] = target_color  # restore
            
            if not recolored:
                success = False
                break
        
        if success:
            current = new_current
            current_k = target_color
    
    return current, current_k


def _search_3_colors(n, adj, edges, deadline):
    """
    Optimized backtracking specifically for 3 colors.
    Returns a list of 50 ints if successful, else None.
    """
    coloring = [-1] * n
    parent = [[i for i in range(n)] for _ in range(3)]
    rank = [[0] * n for _ in range(3)]
    
    def find(p_idx, x):
        p = parent[p_idx]
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(p_idx, x, y):
        p = parent[p_idx]
        r = rank[p_idx]
        rx, ry = find(p_idx, x), find(p_idx, y)
        if rx == ry:
            return False
        if r[rx] < r[ry]:
            rx, ry = ry, rx
        p[ry] = rx
        if r[rx] == r[ry]:
            r[rx] += 1
        return True

    def clone_state():
        return [par[:] for par in parent], [rk[:] for rk in rank]

    def restore_state(state):
        nonlocal parent, rank
        parent, rank = state

    order = sorted(range(n), key=lambda v: -len(adj[v]))
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        forbidden = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                forbidden.add(coloring[nb])
                
        # Symmetry breaking
        # If this is the first vertex, force 0
        # If second, force 0 or 1
        # This reduces search space significantly.
        # We can track max_color_used so far.
        # But simpler: just try 0,1,2.
        
        for c in range(3):
            if c in forbidden:
                continue
            
            affected_pairs = []
            if c == 0:
                affected_pairs = [0, 1]
            elif c == 1:
                affected_pairs = [0, 2]
            else:
                affected_pairs = [1, 2]
            
            snap = clone_state()
            
            valid = True
            for p_idx in affected_pairs:
                other_c = 0
                if p_idx == 0:
                    other_c = 1 if c == 0 else 0
                elif p_idx == 1:
                    other_c = 2 if c == 0 else 0
                else:
                    other_c = 2 if c == 1 else 1
                    
                for nb in adj[v]:
                    if coloring[nb] == other_c:
                        if not union(p_idx, v, nb):
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                coloring[v] = c
                if backtrack(idx + 1):
                    return True
                coloring[v] = -1
            
            restore_state(snap)
            
        return False

    if backtrack(0):
        return coloring
    return None

def construct():
    start = time.time()
    deadline = start + 80
    
    n = N
    edges = EDGES
    
    # Try 3 colors
    result = _search_3_colors(n, adj, edges, deadline)
    if result is not None:
        return [int(c) for c in result]
        
    # Fallback to general backtracking for 4 colors
    deg_order = sorted(range(n), key=lambda v: -len(adj[v]))
    result = _find_coloring(4, edges, n, adj, deadline, deg_order)
    if result is not None:
        return [int(c) for c in result]
        
    return list(range(n))
