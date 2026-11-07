import time
import sys
from collections import defaultdict

# Increase recursion limit just in case
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
    """Verify the coloring is valid."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    
    # Check bicolored cycles using Union-Find for each pair
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

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Sort vertices by degree descending for initial ordering hint
    # But we will use dynamic MRV
    
    # Try k = 3, 4, 5...
    for k in range(3, 15):
        if time.time() > deadline - 5:
            break
            
        # State
        coloring = [-1] * n
        domains = [set(range(k)) for _ in range(n)]
        
        # Union-Find for each pair of colors
        num_pairs = k * (k - 1) // 2
        pair_to_idx = {}
        idx = 0
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                pair_to_idx[(c1, c2)] = idx
                idx += 1
        
        # Lightweight RevUF
        class RevUF:
            __slots__ = ('parent', 'rank', 'stack')
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
                self.stack = []
                
            def find(self, x):
                while self.parent[x] != x:
                    x = self.parent[x]
                return x
            
            def union(self, u, v):
                ru = self.find(u)
                rv = self.find(v)
                if ru == rv:
                    return False
                if self.rank[ru] < self.rank[rv]:
                    ru, rv = rv, ru
                self.stack.append((rv, ru, self.rank[ru]))
                self.parent[rv] = ru
                if self.rank[ru] == self.rank[rv]:
                    self.rank[ru] += 1
                return True
            
            def undo(self):
                if not self.stack:
                    return
                rv, ru, old_rank_ru = self.stack.pop()
                self.parent[rv] = rv
                self.rank[ru] = old_rank_ru

        ufs = [RevUF(n) for _ in range(num_pairs)]
        
        # MRV selection
        def get_mrv_vertex():
            min_size = k + 1
            selected = -1
            for v in range(n):
                if coloring[v] == -1:
                    ds = len(domains[v])
                    if ds == 0:
                        return -1 # Conflict
                    if ds < min_size:
                        min_size = ds
                        selected = v
                        if ds == 1:
                            break
            return selected

        def backtrack():
            if time.time() > deadline - 2:
                return False
            
            v = get_mrv_vertex()
            if v == -1:
                # Check if all colored
                if all(c != -1 for c in coloring):
                    return True
                return False # Conflict (empty domain)
            
            # Try colors
            # Order colors by least constraining value? Or just 0..k-1
            for c in sorted(domains[v]):
                # Check adjacency
                valid = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign
                coloring[v] = c
                domains[v].clear()
                
                snapshots = []
                failed = False
                
                for nb in adj[v]:
                    d = coloring[nb]
                    if d == -1:
                        continue
                    if d == c:
                        continue
                    c_min, c_max = (c, d) if c < d else (d, c)
                    pi = pair_to_idx[(c_min, c_max)]
                    uf = ufs[pi]
                    snap = len(uf.stack)
                    if not uf.union(v, nb):
                        failed = True
                        break
                    snapshots.append((uf, snap))
                
                if not failed:
                    # Forward Checking
                    domain_changes = []
                    for nb in adj[v]:
                        if coloring[nb] == -1:
                            removed = set()
                            for dc in list(domains[nb]):
                                if dc == c:
                                    removed.add(dc)
                                    continue
                                c_min2, c_max2 = (c, dc) if c < dc else (dc, c)
                                pi2 = pair_to_idx[(c_min2, c_max2)]
                                uf2 = ufs[pi2]
                                if uf2.find(v) == uf2.find(nb):
                                    removed.add(dc)
                            if removed:
                                domain_changes.append((nb, removed))
                                domains[nb] -= removed
                                if len(domains[nb]) == 0:
                                    failed = True
                                    break
                            
                    if not failed:
                        if backtrack():
                            return True
                        
                        # Restore domains
                        for nb2, rem2 in domain_changes:
                            domains[nb2] |= rem2
                    
                    # Undo UF
                    for uf_prev, snap_prev in snapshots:
                        while len(uf_prev.stack) > snap_prev:
                            uf_prev.undo()
                
                # Undo coloring
                coloring[v] = -1
                domains[v] = set(range(k))
            
            return False
        
        if backtrack():
            if _check_coloring(coloring, edges):
                return [int(c) for c in coloring]
        
    # Fallback
    coloring = [0] * n
    for v in range(n):
        used = set(coloring[nb] for nb in adj[v])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    return [int(c) for c in coloring]
