import time
import sys

sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
adj = [[] for _ in range(n)]
for u, v in P['edges']:
    adj[u].append(v)
    adj[v].append(u)

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
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

def solve(k, node_limit=2000000):
    colors = [-1] * n
    pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
    
    def is_valid(v, c, dsus):
        for u in adj[v]:
            if colors[u] == c:
                return False
        
        for d in range(k):
            if d == c:
                continue
            pair = (c, d) if c < d else (d, c)
            dsu = dsus[pair]
            
            neighbors_cd = []
            for u in adj[v]:
                cu = colors[u]
                if cu == c or cu == d:
                    neighbors_cd.append(u)
            
            if len(neighbors_cd) >= 2:
                roots = set()
                for u in neighbors_cd:
                    r = dsu.find(u)
                    if r in roots:
                        return False
                    roots.add(r)
        return True

    def backtrack(idx, dsus):
        if idx == n:
            return True
        
        if node_limit[0] <= 0:
            return False
        node_limit[0] -= 1
        
        v = idx
        # Heuristic: Try colors that are most constrained?
        # For now, 0..k-1
        for c in range(k):
            if is_valid(v, c, dsus):
                colors[v] = c
                
                new_dsus = {}
                for p, dsu in dsus.items():
                    new_dsu = DSU(n)
                    new_dsu.parent = dsu.parent[:]
                    new_dsu.rank = dsu.rank[:]
                    new_dsus[p] = new_dsu
                
                valid_union = True
                for d in range(k):
                    if d == c:
                        continue
                    pair = (c, d) if c < d else (d, c)
                    dsu = new_dsus[pair]
                    
                    for u in adj[v]:
                        cu = colors[u]
                        if cu == c or cu == d:
                            if not dsu.union(v, u):
                                valid_union = False
                                break
                    if not valid_union:
                        break
                
                if valid_union and backtrack(idx + 1, new_dsus):
                    return True
                
        return False

    init_dsus = {p: DSU(n) for p in pairs}
    nl = [node_limit]
    if backtrack(0, init_dsus):
        return colors
    return None

def construct():
    start_time = time.time()
    
    # Try 3 colors
    res3 = solve(3)
    if res3 is not None:
        return res3
    
    # Try 4 colors
    res4 = solve(4)
    if res4 is not None:
        return res4
    
    # Fallback
    return list(range(n))

# EVOLVE-BLOCK-END
