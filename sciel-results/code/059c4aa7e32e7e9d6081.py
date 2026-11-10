"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85  # safety margin
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each pair of colors, track edges in the induced subgraph using union-find
    # We'll use a backtracking approach with constraint propagation
    
    def check_bicolored_cycles(coloring, k):
        """Check if any pair of colors forms a cycle. Returns True if valid (no bicolored cycles)."""
        # For each pair of colors, build the subgraph and check for cycles
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                # Collect vertices with these colors
                vertices = [i for i in range(n) if coloring[i] in (c1, c2)]
                if len(vertices) < 3:
                    continue
                # Build subgraph edges
                vset = set(vertices)
                sub_edges = []
                for u, v in edges:
                    if u in vset and v in vset:
                        sub_edges.append((u, v))
                # Check for cycle using union-find
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
                        return False  # cycle detected
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True
                
                for u, v in sub_edges:
                    if not union(u, v):
                        return False  # cycle found
        return True
    
    def is_valid_partial(coloring, k, assigned_count):
        """Quick check: no adjacent same-color, and check bicolored cycles for assigned vertices."""
        # Check adjacent same color
        for u, v in edges:
            if coloring[u] != -1 and coloring[v] != -1:
                if coloring[u] == coloring[v]:
                    return False
        # Check bicolored cycles only among assigned vertices
        assigned = [i for i in range(n) if coloring[i] != -1]
        aset = set(assigned)
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                vertices = [i for i in assigned if coloring[i] in (c1, c2)]
                if len(vertices) < 3:
                    continue
                vset = set(vertices)
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
                    if u in vset and v in vset:
                        if not union(u, v):
                            return False
        return True
    
    # Try k from 3 upward
    for k in range(3, 15):
        if time.time() > deadline:
            break
        coloring = [-1] * n
        
        # Order vertices by degree (all degree 4 here), but use a heuristic:
        # Try to color in an order that maximizes constraints early
        # Use a simple ordering: by index, but we can try different orderings
        order = list(range(n))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            v = order[idx]
            # Try each color
            for c in range(k):
                # Check if color c is valid for v given current coloring
                valid = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                coloring[v] = c
                if is_valid_partial(coloring, k, idx + 1):
                    if backtrack(idx + 1):
                        return True
                coloring[v] = -1
            return False
        
        # Use a smarter ordering: color vertices in an order that helps constraint propagation
        # Try ordering by "constrainedness" - vertices whose neighbors are already colored
        # For simplicity, use a fixed order but try multiple permutations of the first few
        
        if backtrack(0):
            # Found valid coloring with k colors
            distinct = len(set(coloring))
            return [int(c) for c in coloring]
    
    # Fallback: use more colors
    # Simple greedy with many colors
    coloring = [0] * n
    for i in range(n):
        used = set()
        for nb in adj[i]:
            if nb < i:
                used.add(coloring[nb])
        for c in range(50):
            if c not in used:
                coloring[i] = c
                break
    return [int(c) for c in coloring]
# EVOLVE-BLOCK-END
