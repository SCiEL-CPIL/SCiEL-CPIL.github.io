"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def check_pair_forest(colors, c1, c2):
        """Check if edges between color c1 and c2 form a forest."""
        parent = list(range(n))
        rank = [0] * n
        
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
            cu, cv = colors[u], colors[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
        return True
    
    def count_violations(colors, k):
        """Count total violations: adjacent same color + bicolored cycles."""
        violations = 0
        # Adjacent same color
        for u, v in edges:
            if colors[u] == colors[v]:
                violations += 1
        # Bicolored cycles
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                if not check_pair_forest(colors, c1, c2):
                    violations += 1
        return violations
    
    def is_valid(colors, k):
        """Check if coloring is fully valid."""
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                if not check_pair_forest(colors, c1, c2):
                    return False
        return True
    
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Try 3 colors with local search
    best_colors = None
    best_num = n  # fallback
    
    for num_colors in [3, 4, 5]:
        if time.time() > deadline:
            break
        if num_colors >= best_num:
            break
        
        # Multiple random restarts
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Random initial coloring
            colors = [random.randint(0, num_colors - 1) for _ in range(n)]
            
            # Local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try changing each vertex's color
                for v in range(n):
                    if time.time() > deadline:
                        break
                    old_color = colors[v]
                    for new_color in range(num_colors):
                        if new_color == old_color:
                            continue
                        # Check if new color is valid for v (no neighbor has same color)
                        valid = True
                        for nb in adj[v]:
                            if colors[nb] == new_color:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        colors[v] = new_color
                        # Check if this creates any bicolored cycle
                        ok = True
                        for c2 in range(num_colors):
                            if c2 == new_color:
                                continue
                            if not check_pair_forest(colors, new_color, c2):
                                ok = False
                                break
                        if ok:
                            improved = True
                            break
                        else:
                            colors[v] = old_color
                    
                    if improved:
                        break
            
            if is_valid(colors, num_colors):
                best_colors = colors[:]
                best_num = num_colors
                break
        
        if best_colors is not None:
            break
    
    if best_colors is None:
        # Fallback: use all different colors
        best_colors = list(range(n))
        best_num = n
    
    return best_colors
# EVOLVE-BLOCK-END
