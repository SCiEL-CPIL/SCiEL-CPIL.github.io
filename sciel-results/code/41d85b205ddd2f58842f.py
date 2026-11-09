"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def check_coloring(coloring):
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        class UF:
            def __init__(self, n):
                self.p = list(range(n))
            def find(self, x):
                while self.p[x] != x:
                    self.p[x] = self.p[self.p[x]]
                    x = self.p[x]
                return x
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return False
                self.p[py] = px
                return True
        pair_edges = defaultdict(list)
        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if cu > cv:
                cu, cv = cv, cu
            pair_edges[(cu, cv)].append((u, v))
        for (c1, c2), elist in pair_edges.items():
            uf = UF(n)
            for u, v in elist:
                if not uf.union(u, v):
                    return False
        return True
    
    def count_colors(coloring):
        return len(set(coloring))
    
    best_coloring = None
    best_count = n + 1
    start = time.time()
    deadline = start + 85
    
    # Generate and check structured candidates
    checked = set()
    
    def try_coloring(coloring):
        nonlocal best_coloring, best_count
        key = tuple(coloring)
        if key in checked:
            return
        checked.add(key)
        if check_coloring(coloring):
            used = count_colors(coloring)
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
    
    # Periodic modular patterns
    for period in range(2, 51):
        for shift in range(period):
            if time.time() > deadline:
                break
            # mod 3
            try_coloring([((i + shift) % period) % 3 for i in range(n)])
            # mod 4
            try_coloring([((i + shift) % period) % 4 for i in range(n)])
            # mod 5
            try_coloring([((i + shift) % period) % 5 for i in range(n)])
    
    # Linear patterns
    for k in [3, 4, 5, 6]:
        for a in range(1, k):
            for b in range(k):
                if time.time() > deadline:
                    break
                try_coloring([(a * i + b) % k for i in range(n)])
    
    if best_count <= 3:
        result = [int(c) for c in best_coloring]
        for c in result:
            assert isinstance(c, int) and not isinstance(c, bool)
            assert 0 <= c <= 49
        return result
    
    # Local search
    if best_coloring is None:
        best_coloring = [0] * n
        for i in range(n):
            used = set(best_coloring[u] for u in adj[i])
            for c in range(5):
                if c not in used:
                    best_coloring[i] = c
                    break
        best_count = count_colors(best_coloring)
    
    coloring = best_coloring[:]
    random.seed(42)
    
    for iteration in range(300000):
        if time.time() > deadline:
            break
        
        v = random.randint(0, n - 1)
        old_c = coloring[v]
        used = set(coloring[u] for u in adj[v])
        max_c = max(coloring) + 1
        candidates_c = [c for c in range(max_c) if c not in used]
        
        if not candidates_c:
            continue
        
        new_c = random.choice(candidates_c)
        coloring[v] = new_c
        
        if check_coloring(coloring):
            new_count = count_colors(coloring)
            if new_count < best_count:
                best_count = new_count
                best_coloring = coloring[:]
                if best_count == 3:
                    break
        else:
            coloring[v] = old_c
    
    result = [int(c) for c in best_coloring]
    for c in result:
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49
    
    return result
# EVOLVE-BLOCK-END
