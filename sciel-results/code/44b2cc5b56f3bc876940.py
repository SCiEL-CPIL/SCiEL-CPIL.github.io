"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def is_forest(coloring, c1, c2):
        """Check if subgraph induced by colors c1 and c2 is a forest."""
        vertices = [v for v in range(n) if coloring[v] in (c1, c2)]
        if len(vertices) < 3:
            return True
        
        vset = set(vertices)
        # Union-Find on the induced subgraph
        parent = {v: v for v in vertices}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            parent[py] = px
            return True

        for v in vertices:
            for w in adj[v]:
                if w in vset and v < w:
                    if not union(v, w):
                        return False
        return True

    def count_violations(coloring, k):
        """Count adjacent conflicts and bicolored cycles."""
        adj_conflicts = 0
        for u, v in edges:
            if coloring[u] == coloring[v]:
                adj_conflicts += 1
        
        cycle_pairs = 0
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                if not is_forest(coloring, c1, c2):
                    cycle_pairs += 1
        return adj_conflicts, cycle_pairs

    def local_search(k, max_iters=10000):
        """Use local search to find a valid coloring with k colors."""
        # Initialize randomly
        coloring = [random.randint(0, k - 1) for _ in range(n)]
        
        best_coloring = coloring[:]
        best_cost = float('inf')
        
        for i in range(max_iters):
            adj_conf, cycle_conf = count_violations(coloring, k)
            cost = adj_conf * 10 + cycle_conf # Weight adj conflicts higher
            
            if cost == 0:
                return coloring
            
            if cost < best_cost:
                best_cost = cost
                best_coloring = coloring[:]
            
            # Select a vertex to change
            # Prefer vertices involved in adj conflicts
            candidates = []
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    candidates.append(u)
                    candidates.append(v)
            
            if not candidates:
                # No adj conflicts, but cycles exist. Pick random vertex from colors in cycles
                # Simplified: just pick random vertex
                candidates = list(range(n))
            
            v = random.choice(candidates)
            old_c = coloring[v]
            
            # Try all colors, pick best
            best_new_c = old_c
            best_local_cost = cost
            
            for new_c in range(k):
                if new_c == old_c:
                    continue
                coloring[v] = new_c
                adj_conf_new, cycle_conf_new = count_violations(coloring, k)
                local_cost = adj_conf_new * 10 + cycle_conf_new
                
                if local_cost < best_local_cost:
                    best_local_cost = local_cost
                    best_new_c = new_c
            # If no improvement, pick random (kick)
            if best_new_c == old_c:
                best_new_c = random.randint(0, k - 1)
            
            coloring[v] = best_new_c
        
        return best_coloring if best_cost == 0 else None

    # Try k=3, 4, 5, 6
    for k in range(3, 7):
        result = local_search(k, max_iters=5000)
        if result is not None:
            # Verify
            adj_conf, cycle_conf = count_violations(result, k)
            if adj_conf == 0 and cycle_conf == 0:
                return [int(c) for c in result]
        
        # Try a few more times for robustness
        for _ in range(5):
            result = local_search(k, max_iters=5000)
            if result is not None:
                adj_conf, cycle_conf = count_violations(result, k)
                if adj_conf == 0 and cycle_conf == 0:
                    return [int(c) for c in result]
    
    # Fallback: simple greedy or backtracking
    # Just return a valid coloring using many colors if local search fails
    coloring = [0] * n
    for i in range(n):
        used = set(coloring[adj[i]])
        c = 0
        while c in used:
            c += 1
        coloring[i] = c
    return [int(c) for c in coloring]

# EVOLVE-BLOCK-END
