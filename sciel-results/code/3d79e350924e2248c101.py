"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    edge_list = [(u, v) for u, v in edges]
    
    class UnionFind:
        def __init__(self, n):
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
    
    def check_validity(coloring, max_colors):
        for u, v in edge_list:
            if coloring[u] == coloring[v]:
                return False
        
        color_pairs = set()
        for u, v in edge_list:
            c1, c2 = coloring[u], coloring[v]
            if c1 != c2:
                if c1 < c2:
                    color_pairs.add((c1, c2))
                else:
                    color_pairs.add((c2, c1))
        
        for c1, c2 in color_pairs:
            uf = UnionFind(n)
            for u, v in edge_list:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not uf.union(u, v):
                        return False
        
        return True
    
    def backtrack_search(max_colors, order, timeout=10.0):
        deadline_local = start_time + timeout
        coloring = [-1] * n
        
        dsu_state = {}
        
        def get_dsu(c1, c2):
            key = (min(c1, c2), max(c1, c2))
            if key not in dsu_state:
                dsu_state[key] = (list(range(n)), [0] * n)
            return dsu_state[key]
        
        def find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(parent, rank, x, y):
            px, py = find(parent, x), find(parent, y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        def backtrack(v_idx):
            if time.time() > deadline_local:
                return False
            
            if v_idx == n:
                return True
            
            v = order[v_idx]
            
            neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
            available = [c for c in range(max_colors) if c not in neighbor_colors]
            
            for c in available:
                coloring[v] = c
                
                conflict = False
                modified_dsu = {}
                
                for u in adj[v]:
                    d = coloring[u]
                    if d != -1 and d != c:
                        key = (min(c, d), max(c, d))
                        if key not in modified_dsu:
                            parent, rank = get_dsu(c, d)
                            modified_dsu[key] = (parent[:], rank[:])
                        
                        parent, rank = get_dsu(c, d)
                        if not union(parent, rank, u, v):
                            conflict = True
                            break
                
                if not conflict:
                    if backtrack(v_idx + 1):
                        return True
                
                coloring[v] = -1
                for key, (saved_parent, saved_rank) in modified_dsu.items():
                    dsu_state[key] = (saved_parent, saved_rank)
            
            return False
        
        if backtrack(0):
            return coloring[:]
        return None
    
    best_coloring = None
    best_count = n + 1
    
    # Try 3 colors first with natural order
    if time.time() < deadline:
        coloring = backtrack_search(3, list(range(n)), timeout=15.0)
        if coloring is not None and check_validity(coloring, 3):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
    
    # Try 3 colors with reverse order
    if best_count > 3 and time.time() < deadline:
        coloring = backtrack_search(3, list(range(n-1, -1, -1)), timeout=10.0)
        if coloring is not None and check_validity(coloring, 3):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
    
    # Try 4 colors with backtracking
    if best_count > 4 and time.time() < deadline:
        coloring = backtrack_search(4, list(range(n)), timeout=15.0)
        if coloring is not None and check_validity(coloring, 4):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
    
    # Try many random restarts for 3 colors
    if best_count > 3 and time.time() < deadline:
        for attempt in range(2000):
            if time.time() > deadline:
                break
            random.seed(attempt * 31 + 7)
            order = list(range(n))
            random.shuffle(order)
            coloring = backtrack_search(3, order, timeout=1.0)
            if coloring is not None and check_validity(coloring, 3):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                if best_count <= 3:
                    break
    
    # Try random greedy for 3 colors
    if best_count > 3 and time.time() < deadline:
        for attempt in range(5000):
            if time.time() > deadline:
                break
            random.seed(100000 + attempt)
            order = list(range(n))
            random.shuffle(order)
            coloring = [-1] * n
            valid = True
            for v in order:
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(3) if c not in neighbor_colors]
                if not available:
                    valid = False
                    break
                coloring[v] = random.choice(available)
            if valid and check_validity(coloring, 3):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                if best_count <= 3:
                    break
    
    # Try 4 colors with random restarts
    if best_count > 4 and time.time() < deadline:
        for attempt in range(2000):
            if time.time() > deadline:
                break
            random.seed(200000 + attempt)
            order = list(range(n))
            random.shuffle(order)
            coloring = [-1] * n
            valid = True
            for v in order:
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(4) if c not in neighbor_colors]
                if not available:
                    valid = False
                    break
                coloring[v] = random.choice(available)
            if valid and check_validity(coloring, 4):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                if best_count <= 4:
                    break
    
    # Fallback: use greedy with more colors
    if best_coloring is None:
        for max_colors in [5, 6, 7, 8]:
            coloring = [-1] * n
            valid = True
            for v in range(n):
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                if not available:
                    valid = False
                    break
                coloring[v] = available[0]
            if valid and check_validity(coloring, max_colors):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                break
    
    # Final fallback: assign unique colors
    if best_coloring is None:
        best_coloring = list(range(n))
        best_count = n
    
    result = [int(c) for c in best_coloring]
    
    for i, c in enumerate(result):
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49
    
    for u, v in edge_list:
        assert result[u] != result[v]
    
    for c1 in range(best_count):
        for c2 in range(c1 + 1, best_count):
            uf = UnionFind(n)
            for u, v in edge_list:
                cu, cv = result[u], result[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    assert uf.union(u, v)
    
    return result
# EVOLVE-BLOCK-END
