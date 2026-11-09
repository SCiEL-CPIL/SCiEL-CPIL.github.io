"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    pred_neighbors = [[] for _ in range(n)]
    for u, v in edges:
        if pos[u] < pos[v]:
            pred_neighbors[v].append(u)
        else:
            pred_neighbors[u].append(v)
    
    for k in range(2, 10):
        if time.time() > deadline:
            break
        result = _backtrack(n, k, order, pos, pred_neighbors, deadline)
        if result is not None:
            return result
    return list(range(n))

def _backtrack(n, k, order, pos, pred_neighbors, deadline):
    color = [-1] * n
    num_pairs = k * (k - 1) // 2
    pair_index = {}
    for a in range(k):
        for b in range(a + 1, k):
            pair_index[(a, b)] = len(pair_index)
    
    parent = [list(range(n)) for _ in range(num_pairs)]
    rank = [[0] * n for _ in range(num_pairs)]
    
    def find(pi, x):
        root = x
        while parent[pi][root] != root:
            root = parent[pi][root]
        while parent[pi][x] != root:
            parent[pi][x], x = root, parent[pi][x]
        return root
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return color[:]
        
        v = order[idx]
        forbidden = set()
        for u in pred_neighbors[v]:
            if color[u] != -1:
                forbidden.add(color[u])
        
        for c in range(k):
            if c in forbidden:
                continue
            
            color[v] = c
            ok = True
            changes = []
            for u in pred_neighbors[v]:
                if color[u] == -1:
                    continue
                d = color[u]
                a, b = (c, d) if c < d else (d, c)
                pi = pair_index[(a, b)]
                rx = find(pi, v)
                ry = find(pi, u)
                if rx == ry:
                    ok = False
                    break
                if rank[pi][rx] < rank[pi][ry]:
                    parent[pi][rx] = ry
                    changes.append((pi, rx, ry, 'left'))
                elif rank[pi][rx] > rank[pi][ry]:
                    parent[pi][ry] = rx
                    changes.append((pi, ry, rx, 'right'))
                else:
                    parent[pi][ry] = rx
                    rank[pi][rx] += 1
                    changes.append((pi, ry, rx, 'equal'))
            
            if ok:
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            
            color[v] = -1
            for pi, x, y, typ in changes:
                if typ == 'left':
                    parent[pi][x] = x
                elif typ == 'right':
                    parent[pi][y] = y
                else:
                    parent[pi][y] = y
                    rank[pi][x] -= 1
    
    return backtrack(0)

# EVOLVE-BLOCK-END
