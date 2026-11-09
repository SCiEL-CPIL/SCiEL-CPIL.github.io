"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    class UFWithUndo:
        def __init__(self, n):
            self.n = n
            self.parent = list(range(n))
            self.rank = [0] * n
            self.journal = []
        
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
            self.journal.append((self.parent[ry], self.rank[ry], self.parent[rx], self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def undo(self):
            if not self.journal:
                return
            pr, rr, prx, rrx = self.journal.pop()
            self.parent[pr] = pr
            self.rank[pr] = rr
            self.parent[prx] = prx
            self.rank[prx] = rrx

    def try_coloring(max_colors, seed=42):
        rng = random.Random(seed)
        uf_data = {}
        
        def get_uf(c1, c2):
            key = (min(c1, c2), max(c1, c2))
            if key not in uf_data:
                uf_data[key] = UFWithUndo(n)
            return uf_data[key]
        
        color = [None] * n
        unassigned = list(range(n))
        
        def backtrack():
            if time.time() - start_time > 55:
                return False
            if not unassigned:
                return True
            
            # MRV: Find unassigned vertex with fewest available colors
            best_vertex = -1
            best_available = []
            best_count = max_colors + 1
            
            # We can optimize by not checking all unassigned if we find a 1 or 2 quickly
            for v in unassigned:
                used_by_neighbors = set()
                for u in adj[v]:
                    if color[u] is not None:
                        used_by_neighbors.add(color[u])
                
                available = []
                for c in range(max_colors):
                    if c in used_by_neighbors:
                        continue
                    valid = True
                    for u in adj[v]:
                        if color[u] is not None:
                            cu = color[u]
                            if cu == c:
                                valid = False
                                break
                            key = (min(cu, c), max(cu, c))
                            uf = get_uf(cu, c)
                            if uf.find(u) == uf.find(v):
                                valid = False
                                break
                    if valid:
                        available.append(c)
                
                if len(available) < best_count:
                    best_count = len(available)
                    best_vertex = v
                    best_available = available
                    if best_count <= 1:
                        break
            
            if best_count == 0:
                return False
            
            # Remove best_vertex from unassigned
            unassigned.remove(best_vertex)
            
            rng.shuffle(best_available)
            
            for c in best_available:
                color[best_vertex] = c
                undo_list = []
                valid = True
                for u in adj[best_vertex]:
                    if color[u] is not None:
                        cu = color[u]
                        key = (min(cu, c), max(cu, c))
                        uf = get_uf(cu, c)
                        if not uf.union(u, best_vertex):
                            valid = False
                            break
                        undo_list.append((key, uf))
                
                if valid:
                    if backtrack():
                        return True
                
                for key, uf in reversed(undo_list):
                    uf.undo()
                color[best_vertex] = None
            
            # Restore unassigned
            unassigned.append(best_vertex)
            return False
        
        if backtrack():
            return color
        return None
    
    for max_colors in range(3, 50):
        for seed in range(50):
            if time.time() - start_time > 55:
                break
            result = try_coloring(max_colors, seed)
            if result is not None:
                valid = True
                for u, v in edges:
                    if result[u] == result[v]:
                        valid = False
                        break
                if valid:
                    uf_data = {}
                    for u, v in edges:
                        cu, cv = result[u], result[v]
                        if cu == cv:
                            continue
                        key = (min(cu, cv), max(cu, cv))
                        if key not in uf_data:
                            uf_data[key] = UFWithUndo(n)
                        uf = uf_data[key]
                        if not uf.union(u, v):
                            valid = False
                            break
                    if valid:
                        return result
        if time.time() - start_time > 55:
            break
    
    return list(range(n))

# EVOLVE-BLOCK-END
