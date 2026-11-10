"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Order vertices for backtracking (descending degree)
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    def is_forest(color_set):
        """
        Check if the subgraph induced by vertices with colors in color_set is a forest.
        color_set is a set of color labels.
        We use Union-Find to detect cycles.
        """
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
            if colors[u] in color_set and colors[v] in color_set:
                if not union(u, v):
                    return False
        return True

    def backtrack(k, idx):
        """
        Try to color vertices in 'order' with k colors.
        Returns True if a valid coloring is found.
        """
        if time.time() > deadline:
            return False
            
        if idx == n:
            # Final check: all pairs of colors must induce a forest
            # This is expensive to check at the end if we didn't do it incrementally.
            # But for k=4, we can check all pairs.
            for i in range(k):
                for j in range(i+1, k):
                    if not is_forest({i, j}):
                        return False
            return True
        
        v = order[idx]
        
        for c in range(k):
            # 1. Proper coloring check
            valid = True
            for nb in adj[v]:
                if colors[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # 2. Tentative assignment
            colors[v] = c
            
            # 3. Check if this creates a bicolored cycle with any other color
            # We only need to check pairs involving c.
            # If c is new or c is existing, we check (c, c') for all c' in 0..k-1, c' != c.
            # Optimization: If we maintain the invariant that previous assignments were valid,
            # adding v to color c only affects the forest property of pairs (c, c').
            # We must check if the subgraph induced by {c, c'} is still a forest.
            # Since the previous state was a forest for all pairs, adding v (which is connected
            # to some neighbors in c and c') might create a cycle.
            
            cycle_created = False
            for c_prime in range(k):
                if c_prime == c:
                    continue
                # Check if adding v to color c creates a cycle in subgraph {c, c_prime}
                # This is equivalent to: In the subgraph induced by {c, c_prime} BEFORE adding v,
                # did v have two neighbors in the same connected component?
                # Or simpler: Just run the forest check on the restricted subgraph?
                # The `is_forest` function is O(E). Doing it for every step is O(N^2 * E) which is ~50^2 * 100 = 250k ops.
                # 50 steps * 250k = 12.5M ops. This is very fast.
                
                if not is_forest({c, c_prime}):
                    cycle_created = True
                    break
            
            if not cycle_created:
                if backtrack(k, idx + 1):
                    return True
            
            colors[v] = -1
            
        return False

    # Try 3 colors first
    colors = [-1] * n
    if backtrack(3, 0):
        return [int(c) for c in colors]
        
    # If 3 fails or times out, try 4 colors
    colors = [-1] * n
    if backtrack(4, 0):
        return [int(c) for c in colors]
        
    # Fallback: Greedy 4-coloring (guaranteed proper, might have bicolored cycles)
    # Then try to fix cycles with local search.
    # But backtracking with 4 colors should almost certainly succeed for N=50.
    # If it fails, we have a serious problem.
    
    # Emergency fallback: Random 4-coloring with local search repair
    colors = [random.randint(0, 3) for _ in range(n)]
    
    # Ensure proper coloring first
    for v in range(n):
        forbidden = set(colors[nb] for nb in adj[v])
        for c in range(4):
            if c not in forbidden:
                colors[v] = c
                break
        
    # Local search to remove bicolored cycles
    def count_bicolored_cycles(cols):
        count = 0
        for i in range(4):
            for j in range(i+1, 4):
                if not is_forest({i, j}):
                    count += 1
        return count

    curr_viol = count_bicolored_cycles(colors)
    for _ in range(10000):
        if time.time() > deadline:
            break
        if curr_viol == 0:
            break
            
        v = random.randint(0, n-1)
        old_c = colors[v]
        best_c = old_c
        best_v = curr_viol
        
        for c in range(4):
            if c == old_c:
                continue
            # Check proper
            if any(colors[nb] == c for nb in adj[v]):
                continue
            
            colors[v] = c
            v_new = count_bicolored_cycles(colors)
            if v_new < best_v:
                best_v = v_new
                best_c = c
            colors[v] = old_c
            
        colors[v] = best_c
        curr_viol = best_v
        
    return [int(c) for c in colors]

# EVOLVE-BLOCK-END
