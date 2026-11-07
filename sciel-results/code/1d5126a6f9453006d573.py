import time
import random
import sys
from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

edge_list = edges

def get_cost(colors, K):
    # Check properness
    cost = 0
    for u, v in edge_list:
        if colors[u] == colors[v]:
            cost += 1000
    
    if cost > 0:
        return cost
        
    used_colors = set(colors)
    color_list = list(used_colors)
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            a, b = color_list[i], color_list[j]
            uf = UnionFind(N)
            for u, v in edge_list:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not uf.union(u, v):
                        cost += 100
                        break
    return cost

def local_search(K, time_limit):
    start_time = time.time()
    best_cost = float('inf')
    best_coloring = None
    
    # Initial random
    colors = [random.randint(0, K - 1) for _ in range(N)]
    current_cost = get_cost(colors, K)
    
    if current_cost == 0:
        return list(colors)
        
    best_cost = current_cost
    best_coloring = list(colors)
    
    T = 100.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if T < 1.0:
            break
            
        v = random.randint(0, N - 1)
        old_c = colors[v]
        
        # Pick a new color
        new_c = random.randint(0, K - 1)
        if new_c == old_c:
            continue
            
        colors[v] = new_c
        new_cost = get_cost(colors, K)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_coloring = list(colors)
                if current_cost == 0:
                    return best_coloring
        else:
            colors[v] = old_c
            
        T *= cooling
        
    return best_coloring if best_cost == 0 else None

def construct():
    deadline = time.time() + 55.0
    
    # Try K=3
    t_start = time.time()
    res = local_search(3, min(30.0, deadline - time.time() - 5.0))
    if res is not None:
        return res
        
    # Try K=4
    remaining = deadline - time.time()
    if remaining > 5:
        res = local_search(4, min(20.0, remaining - 2.0))
        if res is not None:
            return res
            
    # Fallback: return a known valid 4-coloring if we have one, or generate one
    # We can use the result from local_search(4) even if not 0 cost? No, must be valid.
    # If local_search(4) didn't find 0, we might need to try harder or use a heuristic.
    # But 4 is known to be solvable.
    
    # Let's try a specific structured 4-coloring that might work better than random
    # c(i) = (i // 2) % 4 ?
    # Or just return the best found in K=4 even if cost > 0? No, invalid.
    
    # If we fail to find a valid 4-coloring, we return a valid 5-coloring (identity)
    # But 4 should be findable.
    
    # To be safe, let's run K=4 for a bit longer if K=3 failed
    remaining = deadline - time.time()
    if remaining > 2:
        res = local_search(4, remaining - 1.0)
        if res is not None:
            return res
            
    return list(range(N))
