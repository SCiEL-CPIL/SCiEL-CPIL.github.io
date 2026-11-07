"""Direct approach: find minimal number of colors such that each pair of colors
induces a forest (no bicolored cycle). Use CSP with MRV + incremental DSU rollback."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
    """Check if coloring is valid: adjacent differ, and no bicolored cycle."""
    n = len(coloring)
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
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in elist:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
    return True

def _find_coloring_with_k_colors(k, edges, n, deadline):
    """CSP backtracking with MRV, forward checking, and incremental DSU per color-pair."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Fast DSU with rollback using arrays
    class PairDSU:
        __slots__ = ['parent', 'rank', 'size']
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.size = [1] * n
        
        def find(self, x):
            # No path compression for rollback compatibility
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            """Returns (success, info) where info is needed for rollback."""
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry:
                return (False, None)
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            info = (ry, rx, self.rank[rx], self.size[rx])
            self.parent[ry] = rx
            self.size[rx] += self.size[ry]
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return (True, info)
        
        def rollback(self, info):
            ry, rx, old_rank, old_size = info
            self.parent[ry] = ry
            self.rank[rx] = old_rank
            self.size[rx] = old_size
    
    # State
    coloring = [-1] * n
    # domains[v] as bitmask for speed (k <= 16 should be fine)
    full_mask = (1 << k) - 1
    domains = [full_mask] * n
    domain_sizes = [k] * n
    
    # Pair DSUs - pre-allocate for all pairs
    num_pairs = k * (k - 1) // 2
    pair_dsus = {}
    def get_pair_dsu(a, b):
        if a > b:
            a, b = b, a
        key = a * k + b
        if key not in pair_dsus:
            pair_dsus[key] = PairDSU(n)
        return pair_dsus[key]
    
    # Track colored vertices for faster MRV
    colored_count = [0]
    
    def get_mrv_vertex():
        best_v = -1
        best_size = k + 1
        for v in range(n):
            if coloring[v] == -1:
                sz = domain_sizes[v]
                if sz == 0:
                    return v
                if sz < best_size:
                    best_size = sz
                    best_v = v
                    if sz == 1:
                        break
        return best_v
    
    def popcount(x):
        return bin(x).count('1')
    
    def backtrack(depth):
        if time.time() > deadline:
            return False
        if depth == n:
            return True
        
        v = get_mrv_vertex()
        if v == -1:
            return True
        
        # Get available colors from bitmask
        mask = domains[v]
        if mask == 0:
            return False
        
        # Order colors: try colors used by neighbors first (fail-fast)
        # Build list of available colors
        avail_colors = []
        m = mask
        while m:
            c = (m & (-m)).bit_length() - 1
            avail_colors.append(c)
            m &= m - 1
        
        # Sort by how many neighbors already use this color (more conflicts = try first)
        if len(avail_colors) > 1:
            neighbor_colors = set()
            for nb in adj[v]:
                if coloring[nb] != -1:
                    neighbor_colors.add(coloring[nb])
            avail_colors.sort(key=lambda c: (c in neighbor_colors, c))
        
        for c in avail_colors:
            # Check adjacency constraint
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
                        cycle_found = True
                        break
                    union_infos.append((c, d, info))
            
            if cycle_found:
                for (a, b, info) in reversed(union_infos):
                    get_pair_dsu(a, b).rollback(info)
                continue
            
            # Assign
            coloring[v] = c
            old_mask_v = domains[v]
            domains[v] = 0
            domain_sizes[v] = 0
            colored_count[0] += 1
            
            # Forward checking: update domains of uncolored neighbors
            domain_changes = []
            fc_valid = True
            for nb in adj[v]:
                if coloring[nb] == -1:
                    bit = 1 << c
                    if domains[nb] & bit:
                        old_dom = domains[nb]
                        new_dom = domains[nb] & ~bit
                        domains[nb] = new_dom
                        domain_sizes[nb] = popcount(new_dom)
                        domain_changes.append((nb, old_dom, domain_sizes[nb]))
                        if domain_sizes[nb] == 0:
                            fc_valid = False
                            break
            
            if not fc_valid:
                for (nb, old_dom, _) in domain_changes:
                    domains[nb] = old_dom
                    domain_sizes[nb] = popcount(old_dom)
                coloring[v] = -1
                domains[v] = old_mask_v
                domain_sizes[v] = popcount(old_mask_v)
                colored_count[0] -= 1
                for (a, b, info) in reversed(union_infos):
                    get_pair_dsu(a, b).rollback(info)
                continue
            
            if backtrack(depth + 1):
                return True
            
            # Undo
            for (nb, old_dom, _) in domain_changes:
                domains[nb] = old_dom
                domain_sizes[nb] = popcount(old_dom)
            coloring[v] = -1
            domains[v] = old_mask_v
            domain_sizes[v] = popcount(old_mask_v)
            colored_count[0] -= 1
            for (a, b, info) in reversed(union_infos):
                get_pair_dsu(a, b).rollback(info)
        
        return False
    
    if backtrack(0):
        return coloring[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency for fallback
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try k = 2, 3, 4, ... until we find a valid coloring
    best_result = None
    for k in range(2, 15):
        if time.time() > deadline - 3:
            break
        result = _find_coloring_with_k_colors(k, edges, n, deadline)
        if result is not None:
            if _check_coloring(result, edges):
                best_result = [int(c) for c in result]
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: greedy coloring
    coloring = [-1] * n
    for v in range(n):
        used = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    return [int(c) for c in coloring]
# EVOLVE-BLOCK-END
