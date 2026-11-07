"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deg = [len(adj[i]) for i in range(n)]
    
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
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry:
                return False
            self.history.append((rx, ry, self.parent[ry], self.rank[rx]))
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def undo(self):
            if not self.history:
                return
            rx, ry, old_parent_ry, old_rank_rx = self.history.pop()
            self.parent[ry] = old_parent_ry
            self.rank[rx] = old_rank_rx
    
    def solve_with_k(k):
        color = [-1] * n
        dsus = {}
        
        def get_dsu(c1, c2):
            key = (min(c1, c2), max(c1, c2))
            if key not in dsus:
                dsus[key] = RollbackDSU(n)
            return dsus[key]
        
        order = sorted(range(n), key=lambda i: (-deg[i], i))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            
            for c in range(k):
                valid = True
                for u in adj[v]:
                    cu = color[u]
                    if cu != -1:
                        if cu == c:
                            valid = False
                            break
                        key = (min(c, cu), max(c, cu))
                        dsu = get_dsu(key)
                        if dsu.find(u) == dsu.find(v):
                            valid = False
                            break
                
                if not valid:
                    continue
                
                color[v] = c
                unions = []
                for u in adj[v]:
                    cu = color[u]
                    if cu != -1 and cu != c:
                        key = (min(c, cu), max(c, cu))
                        dsu = get_dsu(key)
                        if dsu.union(u, v):
                            unions.append(key)
                        else:
                            for k2 in reversed(unions):
                                get_dsu(k2).undo()
                            color[v] = -1
                            valid = False
                            break
                
                if not valid:
                    continue
                
                if backtrack(idx + 1):
                    return True
                
                for key in reversed(unions):
                    get_dsu(key).undo()
                color[v] = -1
            
            return False
        
        return backtrack(0)
    
    for k in range(3, 10):
        if time.time() > deadline:
            break
        if solve_with_k(k):
            return [int(color[i]) for i in range(n)]
    
    return list(range(n))
# EVOLVE-BLOCK-END
