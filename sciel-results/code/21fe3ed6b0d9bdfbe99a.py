"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import random
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
    
    # Vertex ordering by descending degree
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    def check_forest(c1, c2, colors):
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
                if not check_forest(i, j, colors):
                    count += 1
        return count
    
    def backtrack(k, idx, colors, deadline):
        """
        Backtracking search for k-coloring.
        Returns True if valid coloring found.
        """
        if time.time() > deadline:
            return False
            
        if idx == n:
            # Final verification
            for i in range(k):
                for j in range(i+1, k):
                    if not check_forest(i, j, colors):
                        return False
            return True
        
        v = order[idx]
        
        # Try colors in a smart order
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
            
            # 3. Check bicolored forest constraint incrementally
            cycle_created = False
            for c_prime in range(k):
                if c_prime == c:
                    continue
                if not check_forest(c, c_prime, colors):
                    cycle_created = True
                    break
            
            if not cycle_created:
                if backtrack(k, idx + 1, colors, deadline):
                    return True
            
            colors[v] = -1
            
        return False
    
    # Try 3 colors first
    colors = [-1] * n
    if backtrack(3, 0, colors, deadline):
        return [int(c) for c in colors]
    
    # If 3 fails, try 4 colors
    colors = [-1] * n
    if backtrack(4, 0, colors, deadline):
        return [int(c) for c in colors]
    
    # Fallback: Local search with simulated annealing
    if time.time() > deadline - 5:
        # Emergency: return a valid 4-coloring if we can find one quickly
        # Use greedy with conflict resolution
        colors = [0] * n
        for v in range(n):
            forbidden = set(colors[nb] for nb in adj[v])
            for c in range(4):
                if c not in forbidden:
                    colors[v] = c
                    break
        return [int(c) for c in colors]
    
    # Simulated annealing local search
    # Start with a random proper 4-coloring
    colors = [random.randint(0, 3) for _ in range(n)]
    
    # Ensure proper coloring
    for _ in range(100):
        proper = True
        for u, v in edges:
            if colors[u] == colors[v]:
                proper = False
                break
        if proper:
            break
        # Fix a random conflict
        u, v = random.choice(edges)
        if colors[u] == colors[v]:
            forbidden = set(colors[nb] for nb in adj[v])
            for c in range(4):
                if c not in forbidden:
                    colors[v] = c
                    break
    
    curr_viol = count_violations(colors, 4)
    best_colors = colors[:]
    best_viol = curr_viol
    
    # Simulated annealing
    T = 1.0
    cooling = 0.995
    for step in range(50000):
        if time.time() > deadline - 2:
            break
        if curr_viol == 0:
            break
            
        # Pick a random vertex
        v = random.randint(0, n-1)
        old_c = colors[v]
        
        # Try a new color
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
            
        # Check proper coloring
        if any(colors[nb] == new_c for nb in adj[v]):
            continue
            
        # Apply change
        colors[v] = new_c
        new_viol = count_violations(colors, 4)
        
        # Accept or reject
        if new_viol < curr_viol:
            curr_viol = new_viol
            if curr_viol < best_viol:
                best_viol = curr_viol
                best_colors = colors[:]
        else:
            # Metropolis acceptance
            delta = new_viol - curr_viol
            if random.random() < math.exp(-delta / T):
                curr_viol = new_viol
            else:
                colors[v] = old_c
        
        T *= cooling
    
    # Use best solution found
    colors = best_colors
    
    # Final verification
    if count_violations(colors, 4) == 0:
        return [int(c) for c in colors]
    
    # If still invalid, try a different approach:
    # Use a pattern-based coloring
    # Pattern: (i // 7 + i % 7) % 4
    colors = [(i // 7 + i % 7) % 4 for i in range(n)]
    if count_violations(colors, 4) == 0:
        return [int(c) for c in colors]
    
    # Last resort: use all distinct colors (always valid)
    colors = list(range(n))
    return [int(c) for c in colors]

# EVOLVE-BLOCK-END
