"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

# Set recursion limit to handle deep backtracking if needed, though MRV helps
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

# Precompute adjacency list and neighbors
N = P['n']
adj = [[] for _ in range(N)]
for u, v in P['edges']:
    adj[u].append(v)
    adj[v].append(u)

# Precompute cycles of length 3 and 4 to enforce constraints more tightly?
# Actually, the "no bicolored cycle" constraint is global.
# Checking it after assignment is expensive.
# Instead, we can check locally during backtracking.
# A cycle is bicolored if it uses only 2 colors.
# This is hard to check incrementally.
# However, if we ensure that for every edge (u,v), the set of colors used in the connected component of (G \ {u,v}) is not just {c_u, c_v}, it's complex.

# Alternative: Just find a valid coloring and verify at the end.
# Backtracking with MRV is fast for small graphs.
# We will try k=3 first. If it fails within time, try k=4.

def is_valid_coloring(colors, k):
    # Check adjacent vertices differ
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    
    # Check subgraph induced by any two colors is a forest
    # This is equivalent to: no cycle in the graph uses only 2 colors.
    # We can check all cycles? Too slow.
    # We can check: for each pair of colors (c1, c2), build subgraph and check for cycles.
    # Number of color pairs is k*(k-1)/2. For k=4, 6 pairs.
    # Each check is O(V+E).
    used_colors = set(colors)
    for c1 in used_colors:
        for c2 in used_colors:
            if c1 >= c2:
                continue
            # Build subgraph with edges where both endpoints have color c1 or c2
            # And at least one has c1 and the other c2? No, induced subgraph includes all vertices with c1 or c2.
            # Edges in induced subgraph: (u,v) such that colors[u] in {c1,c2} and colors[v] in {c1,c2}.
            # We need to check if this subgraph has a cycle.
            # Use Union-Find to detect cycles.
            parent = list(range(N))
            rank = [0] * N
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False # Cycle found
                if rank[rx] < rank[ry]:
                    parent[rx] = ry
                elif rank[rx] > rank[ry]:
                    parent[ry] = rx
                else:
                    parent[ry] = rx
                    rank[rx] += 1
                return True
            
            has_cycle = False
            for u, v in P['edges']:
                cu, cv = colors[u], colors[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if not union(u, v):
                        has_cycle = True
                        break
            if has_cycle:
                return False
    return True

def solve(k):
    colors = [-1] * N
    # Order vertices by degree? All are 4.
    # Maybe just 0..N-1
    # Use MRV: pick uncolored vertex with fewest valid colors
    
    # Precompute constraints?
    # We'll use a simple recursive backtracking
    
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds per k attempt
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == N:
            return True
        
        # Find next uncolored vertex
        # Since we assign in order, we can just check idx
        # But MRV is better. Let's stick to order for simplicity first, 
        # or pre-sort vertices.
        # Actually, for 50 nodes, simple order might be slow if constraints are tight.
        # Let's try to find the uncolored vertex with the most constraints.
        
        # For efficiency in this specific graph, let's just iterate 0 to N-1
        # But if we backtrack, we need to manage state.
        
        # Let's use a standard backtracking with forward checking.
        
        # Find unassigned vertex
        # We maintain a list of unassigned vertices?
        # Simpler: iterate through all vertices, find the first unassigned one.
        # This is O(N) per step. Total O(N^2) overhead. Fine for N=50.
        
        unassigned = -1
        min_options = k + 1
        best_vertex = -1
        
        for v in range(N):
            if colors[v] != -1:
                continue
            # Count valid colors
            valid_colors = []
            for c in range(k):
                if c not in [colors[u] for u in adj[v] if colors[u] != -1]:
                    valid_colors.append(c)
            if len(valid_colors) < min_options:
                min_options = len(valid_colors)
                best_vertex = v
                if min_options == 0:
                    return False # Dead end
                if min_options == 1:
                    break # Can't get better than 1
        
        if best_vertex == -1:
            return False # Should not happen if idx < N and some are unassigned
            
        v = best_vertex
        
        # Try all valid colors
        # Sort colors? Maybe try colors that appear less in neighbors?
        neighbor_colors = set(colors[u] for u in adj[v] if colors[u] != -1)
        
        for c in range(k):
            if c in neighbor_colors:
                continue
            
            colors[v] = c
            
            # Forward check: ensure neighbors have options
            consistent = True
            for u in adj[v]:
                if colors[u] != -1:
                    continue
                # Check if u has any valid color
                u_neighbor_colors = set(colors[x] for x in adj[u] if colors[x] != -1)
                if len([cc for cc in range(k) if cc not in u_neighbor_colors]) == 0:
                    consistent = False
                    break
            
            if consistent:
                if backtrack(idx + 1):
                    return True
            
            colors[v] = -1
        
        return False

    # Optimization: Symmetry breaking.
    # Vertex 0 must be color 0.
    colors[0] = 0
    # Vertex 1 must be color 1 (if k>1).
    if k > 1:
        colors[1] = 1
    # Vertex 2?
    # This reduces branching factor significantly.
    
    if backtrack(0):
        return colors
    return None

def construct():
    # Try k=3
    sol = solve(3)
    if sol:
        if is_valid_coloring(sol, 3):
            return sol
    
    # Try k=4
    sol = solve(4)
    if sol:
        if is_valid_coloring(sol, 4):
            return sol
    
    # Fallback: k=5
    sol = solve(5)
    if sol:
        if is_valid_coloring(sol, 5):
            return sol
            
    # Fallback: identity
    return list(range(N))

# EVOLVE-BLOCK-END
