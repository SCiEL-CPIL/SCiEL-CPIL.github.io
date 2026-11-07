import time
import sys
from collections import deque

sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
adj = [[] for _ in range(n)]
for u, v in P['edges']:
    adj[u].append(v)
    adj[v].append(u)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for safety
    
    def is_valid(v, c, color):
        neighbor_colors = {}
        for u in adj[v]:
            cu = color[u]
            if cu != -1:
                if cu == c:
                    return False
                if cu not in neighbor_colors:
                    neighbor_colors[cu] = []
                neighbor_colors[cu].append(u)
        
        for d, neighs in neighbor_colors.items():
            if len(neighs) > 1:
                target_colors = (c, d)
                # BFS from neighs[0] in subgraph induced by {c, d}
                visited = set()
                queue = deque([neighs[0]])
                visited.add(neighs[0])
                
                found_cycle = False
                while queue:
                    curr = queue.popleft()
                    for nb in adj[curr]:
                        if nb == v:
                            continue
                        if color[nb] == c or color[nb] == d:
                            if nb not in visited:
                                # If we reach another neighbor of v with color d, cycle found
                                # Wait, we are looking for connection between neighs
                                if nb in neighs:
                                    if nb != neighs[0]:
                                        found_cycle = True
                                        break
                                visited.add(nb)
                                queue.append(nb)
                    if found_cycle:
                        break
                
                if found_cycle:
                    return False
        return True

    def backtrack(idx, k, color):
        if time.time() > deadline:
            return None
        if idx == n:
            return color[:]
        
        v = idx
        # Heuristic: try colors that are already used first? Or 0,1,2?
        # For minimization, we are fixing k, so just 0..k-1
        for c in range(k):
            if is_valid(v, c, color):
                color[v] = c
                res = backtrack(idx + 1, k, color)
                if res is not None:
                    return res
                color[v] = -1
        return None

    # Try increasing k
    for k in range(3, 51):
        color = [-1] * n
        # Symmetry breaking: vertex 0 is color 0
        color[0] = 0
        # Quick check for v=0 is trivially true as no neighbors are colored
        res = backtrack(1, k, color)
        if res is not None:
            return [int(x) for x in res]
            
    # Fallback: should not happen for k=n
    return list(range(n))
