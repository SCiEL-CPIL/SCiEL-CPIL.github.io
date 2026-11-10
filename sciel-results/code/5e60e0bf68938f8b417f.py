"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def is_forest(c1, c2, colors):
        """Check if subgraph induced by colors c1 and c2 is a forest using Union-Find."""
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        for u, v in edges:
            if (colors[u] == c1 or colors[u] == c2) and (colors[v] == c1 or colors[v] == c2):
                if not union(u, v):
                    return False
        return True
    
    def count_violations(colors, k):
        """Count number of bicolored cycle violations."""
        count = 0
        for i in range(k):
            for j in range(i+1, k):
                if not is_forest(i, j, colors):
                    count += 1
        return count
    
    def check_valid(colors, k):
        """Check if coloring is valid (proper + all bicolored subgraphs are forests)."""
        # Check proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        # Check bicolored forests
        return count_violations(colors, k) == 0
    
    # Strategy: Start with a proper 4-coloring, then try to merge colors to get 3 colors
    # First, find a proper 4-coloring using greedy + backtracking
    
    def greedy_color(k, order):
        """Greedy coloring with given vertex order."""
        colors = [-1] * n
        for v in order:
            forbidden = set(colors[nb] for nb in adj[v])
            for c in range(k):
                if c not in forbidden:
                    colors[v] = c
                    break
            if colors[v] == -1:
                return None
        return colors
    
    # Vertex ordering: use degree-based ordering
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Try to find a proper 4-coloring
    colors4 = greedy_color(4, order)
    if colors4 is None:
        # Fallback: try different orderings
        orderings = [
            sorted(range(n), key=lambda x: degrees[x], reverse=True),
            list(range(n)),
            sorted(range(n), key=lambda x: (degrees[x], x), reverse=True)
        ]
        for ord_ in orderings:
            colors4 = greedy_color(4, ord_)
            if colors4 is not None:
                break
    
    if colors4 is None:
        # Last resort: use all distinct colors
        return list(range(n))
    
    # Now try to merge two of the four colors to get a 3-coloring
    # For each pair of colors to merge, check if the result is valid
    
    def try_merge(colors, c1, c2, k):
        """Try to merge colors c1 and c2 into color c1. Return new coloring if valid."""
        new_colors = colors[:]
        for i in range(n):
            if new_colors[i] == c2:
                new_colors[i] = c1
        
        # Renumber colors to be 0..k-2
        used_colors = sorted(set(new_colors))
        color_map = {c: i for i, c in enumerate(used_colors)}
        for i in range(n):
            new_colors[i] = color_map[new_colors[i]]
        
        if check_valid(new_colors, k - 1):
            return new_colors
        return None
    
    # Try all pairs of colors to merge
    for c1 in range(4):
        for c2 in range(c1 + 1, 4):
            merged = try_merge(colors4, c1, c2, 4)
            if merged is not None:
                return [int(c) for c in merged]
    
    # If no merge works directly, try local search to improve the 4-coloring
    # to make a merge possible
    
    def local_search_4(colors, max_iters=50000):
        """Local search to find a 4-coloring that allows a valid merge to 3 colors."""
        curr = colors[:]
        best = curr[:]
        best_score = float('inf')
        
        def score(colors):
            """Score = number of violations after attempting all merges."""
            min_viol = float('inf')
            for c1 in range(4):
                for c2 in range(c1 + 1, 4):
                    new_colors = colors[:]
                    for i in range(n):
                        if new_colors[i] == c2:
                            new_colors[i] = c1
                    used_colors = sorted(set(new_colors))
                    color_map = {c: i for i, c in enumerate(used_colors)}
                    for i in range(n):
                        new_colors[i] = color_map[new_colors[i]]
                    viol = count_violations(new_colors, 3)
                    min_viol = min(min_viol, viol)
            return min_viol
        
        curr_score = score(curr)
        best_score = curr_score
        
        T = 1.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if best_score == 0:
                break
            
            v = random.randint(0, n - 1)
            old_c = curr[v]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Check proper coloring
            if any(curr[nb] == new_c for nb in adj[v]):
                continue
            
            curr[v] = new_c
            new_score = score(curr)
            
            if new_score < best_score:
                best_score = new_score
                best = curr[:]
            
            if new_score < curr_score:
                curr_score = new_score
            else:
                delta = new_score - curr_score
                if random.random() < math.exp(-delta / T):
                    curr_score = new_score
                else:
                    curr[v] = old_c
            
            T *= cooling
        
        return best, best_score
    
    best_colors4, best_score = local_search_4(colors4)
    
    if best_score == 0:
        # Try to merge from the improved coloring
        for c1 in range(4):
            for c2 in range(c1 + 1, 4):
                merged = try_merge(best_colors4, c1, c2, 4)
                if merged is not None:
                    return [int(c) for c in merged]
    
    # If still no 3-coloring, try a direct backtracking search for 3 colors
    # with a time limit
    
    def backtrack_3(idx, colors):
        """Backtracking search for 3-coloring."""
        if time.time() > deadline - 3:
            return False
        
        if idx == n:
            return check_valid(colors, 3)
        
        v = order[idx]
        for c in range(3):
            valid = True
            for nb in adj[v]:
                if colors[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            colors[v] = c
            
            # Check bicolored forest constraint incrementally
            cycle_created = False
            for c_prime in range(3):
                if c_prime == c:
                    continue
                if not is_forest(c, c_prime, colors):
                    cycle_created = True
                    break
            
            if not cycle_created:
                if backtrack_3(idx + 1, colors):
                    return True
            
            colors[v] = -1
        
        return False
    
    colors3 = [-1] * n
    if backtrack_3(0, colors3):
        return [int(c) for c in colors3]
    
    # Return the best 4-coloring found
    # Verify it's valid
    if check_valid(best_colors4, 4):
        return [int(c) for c in best_colors4]
    
    # Fallback: return the initial 4-coloring if valid
    if check_valid(colors4, 4):
        return [int(c) for c in colors4]
    
    # Last resort: use all distinct colors (always valid)
    return list(range(n))

# EVOLVE-BLOCK-END
