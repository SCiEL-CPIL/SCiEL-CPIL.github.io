"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85
    
    def check_valid(coloring, K):
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        for c1 in range(K):
            for c2 in range(c1 + 1, K):
                vc1 = [i for i in range(n) if coloring[i] == c1]
                vc2 = [i for i in range(n) if coloring[i] == c2]
                if not vc1 or not vc2:
                    continue
                set1 = set(vc1)
                set2 = set(vc2)
                all_v = set1 | set2
                if len(all_v) < 3:
                    continue
                parent = {v: v for v in all_v}
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    parent[px] = py
                    return True
                for u in vc1:
                    for v in adj[u]:
                        if v in set2:
                            if not union(u, v):
                                return False
        return True
    
    # Try K from 3 to 10 with random restarts
    for K in range(3, 11):
        if time.time() > deadline:
            break
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            # Random proper coloring via greedy
            coloring = [-1] * n
            order = list(range(n))
            random.shuffle(order)
            
            ok = True
            for v in order:
                used = set(coloring[nb] for nb in adj[v] if coloring[nb] != -1)
                avail = [c for c in range(K) if c not in used]
                if not avail:
                    ok = False
                    break
                coloring[v] = random.choice(avail)
            
            if not ok:
                continue
            
            if check_valid(coloring, K):
                return list(coloring)
            
            # Local search
            for iteration in range(2000):
                if time.time() > deadline:
                    break
                
                # Find a violation
                violated = False
                for c1 in range(K):
                    for c2 in range(c1 + 1, K):
                        vc1 = [i for i in range(n) if coloring[i] == c1]
                        vc2 = [i for i in range(n) if coloring[i] == c2]
                        if not vc1 or not vc2:
                            continue
                        set1 = set(vc1)
                        set2 = set(vc2)
                        all_v = set1 | set2
                        if len(all_v) < 3:
                            continue
                        parent = {v: v for v in all_v}
                        def find(x):
                            while parent[x] != x:
                                parent[x] = parent[parent[x]]
                                x = parent[x]
                            return x
                        def union(x, y):
                            px, py = find(x), find(y)
                            if px == py:
                                return False
                            parent[px] = py
                            return True
                        for u in vc1:
                            for v in adj[u]:
                                if v in set2:
                                    if not union(u, v):
                                        violated = True
                                        # Fix: recolor a vertex
                                        pool = vc1 + vc2
                                        for _ in range(10):
                                            vtr = random.choice(pool)
                                            old_c = coloring[vtr]
                                            new_colors = [c for c in range(K) if c != old_c]
                                            if not new_colors:
                                                continue
                                            nc = random.choice(new_colors)
                                            conflict = False
                                            for nb in adj[vtr]:
                                                if coloring[nb] == nc:
                                                    conflict = True
                                                    break
                                            if not conflict:
                                                coloring[vtr] = nc
                                                break
                                        break
                            if violated:
                                break
                        if violated:
                            break
                    if violated:
                        break
                
                if not violated:
                    break
                
                if check_valid(coloring, K):
                    return list(coloring)
    
    return [i for i in range(n)]

# EVOLVE-BLOCK-END
