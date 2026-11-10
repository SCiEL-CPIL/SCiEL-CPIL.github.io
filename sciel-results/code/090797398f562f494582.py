"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        def save(self):
            return (self.parent[:], self.rank[:])
        def restore(self, state):
            self.parent, self.rank = state[0][:], state[1][:]
    
    def is_valid(coloring, k):
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                dsu = DSU(n)
                for u, v in edges:
                    cu, cv = coloring[u], coloring[v]
                    if cu in (c1, c2) and cv in (c1, c2):
                        if not dsu.union(u, v):
                            return False
        return True
    
    order = sorted(range(n), key=lambda i: (-len(adj[i]), i))
    
    best = None
    best_k = n
    
    for k in range(3, min(n+1, 10)):
        if time.time() > deadline:
            break
        if k >= best_k:
            break
        
        coloring = [-1] * n
        assigned = [False] * n
        
        # DSU for each color pair
        dsus = {}
        for c1 in range(k):
            for c2 in range(c1+1, k):
                dsus[(c1,c2)] = DSU(n)
        
        def bt(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            
            v = order[pos]
            
            forbidden = set()
            for nb in adj[v]:
                if assigned[nb]:
                    forbidden.add(coloring[nb])
            
            for c in range(k):
                if c in forbidden:
                    continue
                
                coloring[v] = c
                assigned[v] = True
                
                # Save DSU states for pairs involving c
                present = set(coloring[i] for i in range(n) if assigned[i])
                affected_pairs = []
                for c2 in present:
                    if c2 != c:
                        key = (min(c,c2), max(c,c2))
                        affected_pairs.append(key)
                
                saved = {}
                for key in affected_pairs:
                    saved[key] = dsus[key].save()
                
                ok = True
                for key in affected_pairs:
                    c2 = max(key) if key[0] == c else min(key)
                    dsu = dsus[key]
                    for nb in adj[v]:
                        if not assigned[nb]:
                            continue
                        cnb = coloring[nb]
                        if cnb in key:
                            if not dsu.union(v, nb):
                                ok = False
                                break
                    if not ok:
                        break
                
                if ok:
                    if bt(pos + 1):
                        return True
                
                # Restore DSUs
                for key in affected_pairs:
                    dsus[key].restore(saved[key])
                
                assigned[v] = False
                coloring[v] = -1
            
            return False
        
        if bt(0):
            best = coloring[:]
            best_k = k
            break
    
    if best is None:
        return [int(i) for i in range(n)]
    
    cmap = {}
    nc = 0
    out = []
    for c in best:
        if c not in cmap:
            cmap[c] = nc
            nc += 1
        out.append(cmap[c])
    return [int(x) for x in out]

# EVOLVE-BLOCK-END
