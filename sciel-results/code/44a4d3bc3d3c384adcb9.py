import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 55
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline:
            break
            
        # State
        colors = [-1] * n
        
        # DSU for each pair of colors
        # pairs[i][j] for i < j
        dsus = {}
        for i in range(k):
            for j in range(i+1, k):
                dsus[(i,j)] = {'parent': list(range(n)), 'rank': [0]*n}
                
        def find(dsu, x):
            p = dsu['parent']
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x
            
        def union(dsu, x, y):
            p = dsu['parent']
            r = dsu['rank']
            px = find(dsu, x)
            py = find(dsu, y)
            if px == py:
                return None
            if r[px] < r[py]:
                px, py = py, px
            p[py] = px
            if r[px] == r[py]:
                r[px] += 1
            return (py, px, r[px] - 1)
            
        def undo(dsu, changes):
            py, px, old_rank = changes
            p = dsu['parent']
            r = dsu['rank']
            p[py] = py
            r[px] = old_rank
            
        def get_dsu(c1, c2):
            if c1 > c2:
                c1, c2 = c2, c1
            return dsus[(c1, c2)]
            
        # Order vertices - use index order for determinism
        order = list(range(n))
        
        # Fix vertex 0 to color 0 to break symmetry
        colors[0] = 0
        
        found = [False]
        
        def backtrack(idx):
            if time.time() > deadline:
                return
            if found[0]:
                return
            if idx == n:
                found[0] = True
                return
                
            v = order[idx]
            
            # Try each color
            for c in range(k):
                # Check if c is valid for v
                valid = True
                changes_list = []
                
                # Check neighbors
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                    if colors[nb] == -1:
                        continue
                    c_nb = colors[nb]
                    if c == c_nb:
                        valid = False
                        break
                    # Get DSU for pair (c, c_nb)
                    c1, c2 = (c, c_nb) if c < c_nb else (c_nb, c)
                    dsu = get_dsu(c1, c2)
                    changes = union(dsu, v, nb)
                    if changes is None:
                        valid = False
                        break
                    changes_list.append((dsu, changes))
                
                if not valid:
                    # Undo all changes made
                    for dsu, changes in changes_list:
                        undo(dsu, changes)
                    continue
                
                # All good, assign color and recurse
                colors[v] = c
                if backtrack(idx + 1):
                    return
                colors[v] = -1
                
                # Undo all changes
                for dsu, changes in changes_list:
                    undo(dsu, changes)
            
        backtrack(1)  # Start from vertex 1 as vertex 0 is fixed
        
        if found[0]:
            return [int(c) for c in colors]
    
    # Fallback: return a valid coloring with 50 colors
    return list(range(n))
