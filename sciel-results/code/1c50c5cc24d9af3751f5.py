"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Union-Find for bicolored cycle detection
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
                return False  # cycle detected
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True
    
    def is_valid_coloring(coloring, max_colors=None):
        """Check if coloring is valid: adjacent vertices differ, and bicolored subgraphs are forests."""
        # Check adjacent vertices differ
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        if max_colors is not None:
            used = set(coloring)
            if len(used) > max_colors:
                return False
        
        # For each pair of colors, check the induced subgraph is a forest
        color_pairs = set()
        for u, v in edges:
            c1, c2 = coloring[u], coloring[v]
            if c1 != c2:
                if c1 < c2:
                    color_pairs.add((c1, c2))
                else:
                    color_pairs.add((c2, c1))
        
        for c1, c2 in color_pairs:
            uf = UnionFind(n)
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not uf.union(u, v):
                        return False  # bicolored cycle detected
        
        return True
    
    def count_distinct(coloring):
        return len(set(coloring))
    
    # Strategy: Try greedy coloring with random restarts, then local search
    best_coloring = list(range(n))  # fallback: use all 50 colors
    best_count = n
    
    # Orderings to try
    orderings = []
    # Degree-based ordering
    orderings.append(sorted(range(n), key=lambda x: -len(adj[x])))
    # Random orderings
    for i in range(20):
        orderings.append(list(range(n)))
        random.shuffle(orderings[-1])
    
    # Try with increasing max_colors
    for max_colors in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 2:
            break
        
        found = False
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
            
            random.seed(attempt * 7 + max_colors)
            order = list(range(n))
            random.shuffle(order)
            
            coloring = [-1] * n
            valid = True
            
            for v in order:
                used_colors = set(coloring[u] for u in adj[v] if coloring[u] >= 0)
                available = [c for c in range(max_colors) if c not in used_colors]
                
                if not available:
                    valid = False
                    break
                
                # Try each available color and pick the one that minimizes future conflicts
                best_c = None
                best_score = float('inf')
                for c in available:
                    # Score: number of neighbors that would have this color
                    score = sum(1 for u in adj[v] if coloring[u] == c)
                    if score < best_score:
                        best_score = score
                        best_c = c
                
                coloring[v] = best_c
            
            if not valid:
                continue
            
            # Verify with union-find
            if is_valid_coloring(coloring, max_colors):
                used = count_distinct(coloring)
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                
                if used <= max_colors:
                    found = True
                    # If we found with 3 colors, this is likely optimal
                    if max_colors == 3:
                        break
        
        if found and max_colors == 3:
            break
    
    # If we have a solution, try to optimize further with local search
    if best_count > 3:
        for attempt in range(300):
            if time.time() > deadline - 5:
                break
            
            random.seed(1000 + attempt)
            
            # Start from best coloring and try to reduce colors
            coloring = best_coloring[:]
            improved = True
            
            while improved and time.time() < deadline - 5:
                improved = False
                
                # Try to recolor each vertex to a lower color if possible
                for v in range(n):
                    old_c = coloring[v]
                    for new_c in range(old_c):
                        if new_c == old_c:
                            continue
                        
                        # Check if new_c conflicts with neighbors
                        conflict = False
                        for u in adj[v]:
                            if coloring[u] == new_c:
                                conflict = True
                                break
                        
                        if conflict:
                            continue
                        
                        # Try the recoloring
                        coloring[v] = new_c
                        
                        # Check bicolored constraint
                        if is_valid_coloring(coloring):
                            if count_distinct(coloring) < best_count:
                                best_count = count_distinct(coloring)
                                best_coloring = coloring[:]
                            improved = True
                            break
                        else:
                            coloring[v] = old_c
            
            # Also try random perturbations
            coloring = best_coloring[:]
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                
                v = random.randint(0, n - 1)
                old_c = coloring[v]
                new_c = random.randint(0, max(best_count, 4) - 1)
                
                if new_c == old_c:
                    continue
                
                # Check if new_c conflicts with neighbors
                conflict = False
                for u in adj[v]:
                    if coloring[u] == new_c:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                coloring[v] = new_c
                
                if is_valid_coloring(coloring):
                    if count_distinct(coloring) < best_count:
                        best_count = count_distinct(coloring)
                        best_coloring = coloring[:]
                    else:
                        # Keep the perturbation if it doesn't make things worse
                        pass
                else:
                    coloring[v] = old_c
    
    # Final verification
    result = [int(c) for c in best_coloring]
    
    for i, c in enumerate(result):
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49
    
    return result
# EVOLVE-BLOCK-END
