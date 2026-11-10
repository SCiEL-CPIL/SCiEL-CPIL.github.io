"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def has_bicolored_cycle(colors, c1, c2):
        """Check if vertices colored c1 or c2 contain a cycle using union-find."""
        verts = set()
        for i, c in enumerate(colors):
            if c == c1 or c == c2:
                verts.add(i)
        
        parent = {v: v for v in verts}
        rank = {v: 0 for v in verts}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        for u, v in edges:
            if u in verts and v in verts:
                if not union(u, v):
                    return True
        return False
    
    def is_valid(colors):
        """Check if coloring is fully valid."""
        # Check adjacent vertices differ
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Check bicolored forests
        used = set(colors)
        for c1, c2 in combinations(used, 2):
            if has_bicolored_cycle(colors, c1, c2):
                return False
        # Also check single color (should be independent set, no edges within same color)
        # Already covered by adjacent check
        return True
    
    def count_distinct(colors):
        return len(set(colors))
    
    # Try k from 2 upward with backtracking
    for k in range(2, 10):
        if time.time() > deadline:
            break
        
        colors = [-1] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return is_valid(colors)
            
            v = idx
            # Try each color
            for c in range(k):
                # Check adjacent constraint
                valid = True
                for u in adj[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[v] = c
                
                # Early check: for each existing color pair involving c, check no cycle
                # Only check pairs where both colors appear in assigned vertices
                ok = True
                assigned_colors = set(colors[:idx+1])
                for other_c in assigned_colors:
                    if other_c != c:
                        if has_bicolored_cycle(colors, c, other_c):
                            ok = False
                            break
                
                if ok:
                    if backtrack(idx + 1):
                        return True
                
                colors[v] = -1
            
            return False
        
        if backtrack(0):
            return [int(c) for c in colors]
    
    # Fallback: use all distinct colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
