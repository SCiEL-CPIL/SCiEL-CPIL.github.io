"""Direct approach: find minimal number of colors such that each pair of colors
induces a forest (no bicolored cycle). Use CSP with MRV + incremental DSU rollback."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
    """Check if coloring is valid: adjacent differ, and no bicolored cycle."""
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
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
        for u, v in elist:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            parent[ru] = rv
    return True

def _find_coloring_with_k_colors(k, edges, n, deadline):
    """CSP backtracking with MRV, forward checking, and incremental DSU per color-pair."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge list for quick access
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    # For each color pair, maintain a union-find. Since k is small, we index pairs.
    # Pair index: (i,j) -> i*k + j for i<j, or use a dict.
    # We'll use a dict mapping (c1,c2) -> {parent: {}, rank: {}}
    
    # Incremental DSU with rollback
    # For each color pair (a,b), we track:
    #   parent: dict vertex -> vertex
    #   rank: dict vertex -> int
    #   history: list of (vertex, old_parent, old_rank_of_root) for rollback
    
    # Actually, let's use a simpler approach:
    # Maintain for each color pair a list of edges and a DSU.
    # When we assign a vertex, we add edges to already-colored neighbors.
    # We check if adding creates a cycle.
    # For rollback, we just need to undo the union operations.
    
    # Let's use a global approach:
    # - coloring[v] for each vertex
    # - For each pair (a,b), maintain parent_b, rank_b dicts
    # - When assigning color c to v, for each colored neighbor u with color d:
    #   pair = (min(c,d), max(c,d))
    #   Try to union v and u in that pair's DSU
    #   If they're already connected -> cycle -> invalid
    #   Record the union for rollback
    
    class PairDSU:
        def __init__(self):
            self.parent = {}
            self.rank = {}
        
        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
                self.rank[x] = 0
            # Path compression not used for rollback; use simple find
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            """Returns (success, info) where info is needed for rollback."""
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry:
                return (False, None)
            # Union by rank
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            # Attach ry to rx
            info = (ry, rx, self.rank[rx])
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return (True, info)
        
        def rollback(self, info):
            ry, rx, old_rank = info
            self.parent[ry] = ry
            self.rank[rx] = old_rank
    
    # State
    coloring = [-1] * n
    # domains[v] = set of available colors for v
    domains = [set(range(k)) for _ in range(n)]
    
    # Pair DSUs
    pair_dsus = {}
    def get_pair_dsu(a, b):
        if a > b:
            a, b = b, a
        if (a, b) not in pair_dsus:
            pair_dsus[(a, b)] = PairDSU()
        return pair_dsus[(a, b)]
    
    # History for rollback
    # Each entry: (vertex, old_color, list_of_union_infos)
    history = []
    
    # Forward checking: when assigning color c to vertex v,
    # for each uncolored neighbor u, remove c from domains[u]
    # Also, for the bicolored cycle constraint, we need to check
    # if assigning c to v would make any future assignment impossible.
    # This is complex, so we'll do a lighter form:
    # Just check that no uncolored neighbor has domain size 0 after removal.
    
    # MRV: pick uncolored vertex with smallest domain
    
    def get_mrv_vertex():
        best_v = -1
        best_size = k + 1
        for v in range(n):
            if coloring[v] == -1:
                sz = len(domains[v])
                if sz == 0:
                    return v  # dead end
                if sz < best_size:
                    best_size = sz
                    best_v = v
        return best_v
    
    def backtrack(depth):
        if time.time() > deadline:
            return False
        if depth == n:
            return True
        
        v = get_mrv_vertex()
        if v == -1:
            return True  # all colored
        
        # Order colors: try colors that appear less in neighbors first (or random)
        # Simple: try in order
        for c in sorted(domains[v]):
            # Check: can we assign c to v?
            # 1. Adjacency: no neighbor already has color c (should be handled by domain)
            # 2. Bicolored cycle: for each colored neighbor u with color d,
            #    check if union(v, u) in pair_dsu(c,d) creates a cycle
            
            # Verify adjacency constraint (should be in domain, but double-check)
            adj_conflict = False
            for nb in adj[v]:
                if coloring[nb] == c:
                    adj_conflict = True
                    break
            if adj_conflict:
                continue
            
            # Check bicolored cycle constraint
            union_infos = []
            cycle_found = False
            for nb in adj[v]:
                if coloring[nb] != -1:
                    d = coloring[nb]
                    if d == c:
                        cycle_found = True
                        break
                    dsu = get_pair_dsu(c, d)
                    success, info = dsu.union(v, nb)
                    if not success:
                        # Rollback previous unions in this iteration
                        for ui in reversed(union_infos):
                            a, b = (c, coloring[adj[v][union_infos.index(ui)]]) if False else (0,0)
                        # Actually let's be more careful
                        for i, ui in enumerate(union_infos):
                            # We need to know which pair each union was for
                            # Let's store (pair, info)
                            pass
                        cycle_found = True
                        break
                    union_infos.append((c, d, info))
            
            if cycle_found:
                # Rollback
                for (a, b, info) in reversed(union_infos):
                    get_pair_dsu(a, b).rollback(info)
                continue
            
            # Assign
            old_color = coloring[v]
            coloring[v] = c
            old_domain_v = set(domains[v])
            domains[v] = set()  # v is now colored
            
            # Forward checking: update domains of uncolored neighbors
            domain_changes = []
            fc_valid = True
            for nb in adj[v]:
                if coloring[nb] == -1:
                    if c in domains[nb]:
                        old_dom = set(domains[nb])
                        domains[nb].discard(c)
                        domain_changes.append((nb, old_dom))
                        if len(domains[nb]) == 0:
                            fc_valid = False
                            break
            
            if not fc_valid:
                # Undo
                for (nb, old_dom) in domain_changes:
                    domains[nb] = old_dom
                coloring[v] = old_color
                domains[v] = old_domain_v
                for (a, b, info) in reversed(union_infos):
                    get_pair_dsu(a, b).rollback(info)
                continue
            
            # Recursive call
            if backtrack(depth + 1):
                return True
            
            # Undo
            for (nb, old_dom) in domain_changes:
                domains[nb] = old_dom
            coloring[v] = old_color
            domains[v] = old_domain_v
            for (a, b, info) in reversed(union_infos):
                get_pair_dsu(a, b).rollback(info)
        
        return False
    
    if backtrack(0):
        return coloring[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Try k = 2, 3, 4, ... until we find a valid coloring
    for k in range(2, 10):
        if time.time() > deadline - 5:
            break
        result = _find_coloring_with_k_colors(k, edges, n, deadline)
        if result is not None:
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Fallback: greedy coloring with many colors (always valid)
    return list(range(n))
# EVOLVE-BLOCK-END
