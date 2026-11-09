"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

class UF:
    def __init__(self, n):
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def snapshot(self):
        return (self.parent[:], self.rank[:])

    def restore(self, snap):
        self.parent, self.rank = snap[0][:], snap[1][:]

def _solve_k_coloring(k, adj, n, deadline):
    """
    Backtracking solver with MRV heuristic and incremental Union-Find for bicolored forest constraints.
    """
    colors = [-1] * n
    # Pairs of colors to track
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            pairs.append((i, j))
    
    ufs = {}
    for p in pairs:
        ufs[p] = UF(n)

    # Precompute neighbor lists for faster access
    neighbors = [adj[i] for i in range(n)]
    
    def get_mrv_vertex():
        best_v = -1
        min_avail = k + 1
        for v in range(n):
            if colors[v] != -1:
                continue
            # Count available colors
            used = set()
            for u in neighbors[v]:
                if colors[u] != -1:
                    used.add(colors[u])
            avail = k - len(used)
            if avail < min_avail:
                min_avail = avail
                best_v = v
                if avail == 0:
                    return v # Dead end
        return best_v

    def bt():
        if time.time() > deadline:
            return False
        
        v = get_mrv_vertex()
        if v == -1:
            return True # All colored
        
        # Try each color
        for c in range(k):
            # Check proper coloring
            valid = True
            for u in neighbors[v]:
                if colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check bicolored forest constraints
            cycle_found = False
            for c2 in range(k):
                if c2 == c:
                    continue
                uf = ufs[(c, c2) if c < c2 else (c2, c)]
                # v is currently uncolored, so it's not in the UF graph yet.
                # We are adding v.
                # Edges added are (v, u) where u is colored c2.
                # A cycle is formed if v connects to two vertices in the same component.
                
                comps = set()
                for u in neighbors[v]:
                    if colors[u] == c2:
                        root = uf.find(u)
                        if root in comps:
                            cycle_found = True
                            break
                        comps.add(root)
                if cycle_found:
                    break
            
            if cycle_found:
                continue
            
            # Valid assignment. Apply it.
            colors[v] = c
            
            # Take snapshots of UFs for pairs involving c
            snaps = []
            for c2 in range(k):
                if c2 == c:
                    continue
                p = (c, c2) if c < c2 else (c2, c)
                snaps.append((p, ufs[p].snapshot()))
            
            # Apply unions
            for c2 in range(k):
                if c2 == c:
                    continue
                p = (c, c2) if c < c2 else (c2, c)
                uf = ufs[p]
                for u in neighbors[v]:
                    if colors[u] == c2:
                        uf.union(v, u)
            
            if bt():
                return True
            
            # Undo unions
            for p, snap in reversed(snaps):
                ufs[p].restore(snap)
            
            colors[v] = -1
        
        return False

    if bt():
        return colors[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = P['n']
    adj = _build_adjacency()
    
    best_result = None
    best_score = n + 1

    # Try k=3 first
    result = _solve_k_coloring(3, adj, n, deadline)
    if result is not None:
        best_result = result
        best_score = 3
    else:
        # Try k=4
        result = _solve_k_coloring(4, adj, n, deadline)
        if result is not None:
            best_result = result
            best_score = 4
    
    # Fallback if something went wrong
    if best_result is None:
        # Simple greedy
        colors = [0] * n
        for i in range(n):
            used = set()
            for v in adj[i]:
                if v < i:
                    used.add(colors[v])
            for c in range(10):
                if c not in used:
                    colors[i] = c
                    break
            else:
                colors[i] = max(used) + 1 if used else 0
        best_result = colors
        best_score = len(set(colors))

    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
