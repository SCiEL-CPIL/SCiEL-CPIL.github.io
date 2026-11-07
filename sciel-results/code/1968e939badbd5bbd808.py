"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(colors, edges):
    """Check if for every pair of colors, the induced subgraph is a forest."""
    n = len(colors)
    color_map = defaultdict(list)
    for v in range(n):
        color_map[colors[v]].append(v)
    
    colors_used = set(colors)
    for i, c1 in enumerate(colors_used):
        for c2 in colors_used:
            if c1 >= c2:
                continue
            # Build subgraph with only vertices of color c1 or c2
            verts = [v for v in range(n) if colors[v] in (c1, c2)]
            if len(verts) < 3:
                continue
            vert_set = set(verts)
            # Build adjacency in this subgraph
            adj = defaultdict(list)
            for u, v in edges:
                if u in vert_set and v in vert_set:
                    adj[u].append(v)
                    adj[v].append(u)
            # Check for cycles using union-find
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            for u in verts:
                find(u)
            for u, v in edges:
                if u in vert_set and v in vert_set:
                    if not union(u, v):
                        return False
    return True

def _validate(colors, edges):
    """Validate coloring: adjacent differ, no bicolored cycles."""
    n = len(colors)
    # Check adjacent vertices differ
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # Check bicolored forests
    if not _check_bicolored_forest(colors, edges):
        return False
    return True

def _find_all_cycles(adj, n):
    """Find all simple cycles in the graph."""
    cycles = []
    visited = [False] * n
    
    def dfs(start, current, path, visited_local):
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 3:
                # Found a cycle
                # Normalize the cycle representation
                # Find the minimum vertex in the cycle
                min_vertex = min(path)
                min_idx = path.index(min_vertex)
                # Rotate so min_vertex is first
                normalized = path[min_idx:] + path[:min_idx]
                # Check for duplicates
                if normalized not in cycles:
                    cycles.append(normalized)
            elif neighbor > start and not visited_local[neighbor]:
                # Only visit neighbors > start to avoid finding same cycle multiple times
                # and to ensure we find each cycle exactly once (with min vertex as start)
                visited_local[neighbor] = True
                path.append(neighbor)
                dfs(start, neighbor, path, visited_local)
                path.pop()
                visited_local[neighbor] = False

    for start in range(n):
        visited_local = [False] * n
        visited_local[start] = True
        dfs(start, start, [start], visited_local)
        
    return cycles

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Precompute all simple cycles
    cycles = _find_all_cycles(adj, n)
    
    # Try increasing number of colors
    for k in range(3, 8):
        # MILP Formulation
        # Variables:
        # x[v, c] for v in 0..n-1, c in 0..k-1
        # y[u, v, c1, c2] for each edge (u,v) and pair (c1, c2) with c1 < c2
        
        num_x = n * k
        edge_list = edges
        num_edges = len(edge_list)
        
        # Map edge to index
        edge_idx = {}
        for i, (u, v) in enumerate(edge_list):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i
            
        # Number of color pairs
        color_pairs = [(c1, c2) for c1 in range(k) for c2 in range(c1 + 1, k)]
        num_pairs = len(color_pairs)
        
        # Total variables: num_x + num_edges * num_pairs
        # We need to map (type, index) to variable index
        # Let's just create a dictionary for variable indices
        
        var_index = {}
        var_count = 0
        
        # x variables
        for v in range(n):
            for c in range(k):
                var_index[('x', v, c)] = var_count
                var_count += 1
                
        # y variables
        for i, (u, v) in enumerate(edge_list):
            for c1, c2 in color_pairs:
                var_index[('y', i, c1, c2)] = var_count
                var_count += 1
                
        m = var_count
        
        c_obj = np.zeros(m)
        
        A_rows = []
        b_lower = []
        b_upper = []
        
        # 1. Each vertex gets exactly one color
        for v in range(n):
            row = np.zeros(m)
            for c in range(k):
                row[var_index[('x', v, c)]] = 1
            A_rows.append(row)
            b_lower.append(1)
            b_upper.append(1)
            
        # 2. Adjacent vertices have different colors
        for u, v in edges:
            for c in range(k):
                row = np.zeros(m)
                row[var_index[('x', u, c)]] = 1
                row[var_index[('x', v, c)]] = 1
                A_rows.append(row)
                b_lower.append(0)
                b_upper.append(1)
                
        # 3. Linking constraints for y variables
        # y[i, c1, c2] <= x[u, c1] + x[u, c2]
        # y[i, c1, c2] <= x[v, c1] + x[v, c2]
        # y[i, c1, c2] >= x[u, c1] + x[u, c2] + x[v, c1] + x[v, c2] - 2
        
        for i, (u, v) in enumerate(edge_list):
            for c1, c2 in color_pairs:
                idx_y = var_index[('y', i, c1, c2)]
                
                # y <= x_u_c1 + x_u_c2
                row = np.zeros(m)
                row[idx_y] = 1
                row[var_index[('x', u, c1)]] = -1
                row[var_index[('x', u, c2)]] = -1
                A_rows.append(row)
                b_lower.append(0)
                b_upper.append(0)
                
                # y <= x_v_c1 + x_v_c2
                row = np.zeros(m)
                row[idx_y] = 1
                row[var_index[('x', v, c1)]] = -1
                row[var_index[('x', v, c2)]] = -1
                A_rows.append(row)
                b_lower.append(0)
                b_upper.append(0)
                
                # y >= x_u_c1 + x_u_c2 + x_v_c1 + x_v_c2 - 2
                # => -y + x_u_c1 + x_u_c2 + x_v_c1 + x_v_c2 <= 2
                row = np.zeros(m)
                row[idx_y] = -1
                row[var_index[('x', u, c1)]] = 1
                row[var_index[('x', u, c2)]] = 1
                row[var_index[('x', v, c1)]] = 1
                row[var_index[('x', v, c2)]] = 1
                A_rows.append(row)
                b_lower.append(0)
                b_upper.append(2)
                
        # 4. Forest constraints
        # For each cycle C and each pair (c1, c2):
        # sum_{(u,v) in C} y[i, c1, c2] <= len(C) - 1
        
        for cycle in cycles:
            # Get edges in the cycle
            # Cycle is a list of vertices [v0, v1, ..., v_{L-1}]
            # Edges are (v0,v1), (v1,v2), ..., (v_{L-1}, v0)
            cycle_edges = []
            L = len(cycle)
            for j in range(L):
                u = cycle[j]
                v = cycle[(j + 1) % L]
                # Find edge index
                # The edge might be stored as (min, max) or just (u,v) in edge_list
                # We stored edge_idx for both (u,v) and (v,u)
                if (u, v) in edge_idx:
                    cycle_edges.append(edge_idx[(u, v)])
                else:
                    # Should not happen if graph is consistent
                    pass
            
            for c1, c2 in color_pairs:
                row = np.zeros(m)
                for i in cycle_edges:
                    row[var_index[('y', i, c1, c2)]] = 1
                A_rows.append(row)
                b_lower.append(0)
                b_upper.append(L - 1)
                
        A = np.array(A_rows) if A_rows else np.zeros((1, m))
        b_lower = np.array(b_lower) if b_lower else np.zeros(1)
        b_upper = np.array(b_upper) if b_upper else np.zeros(1)
        
        integrality = np.ones(m)
        bounds = Bounds(lb=0, ub=1)
        
        try:
            res = milp(c=c_obj, constraints=LinearConstraint(A, b_lower, b_upper),
                       integrality=integrality, bounds=bounds)
            if res.status == 0:
                x = res.x
                colors = [-1] * n
                for v in range(n):
                    for c in range(k):
                        if x[var_index[('x', v, c)]] > 0.5:
                            colors[v] = c
                            break
                if _validate(colors, edges):
                    return [int(c) for c in colors]
        except Exception:
            pass
            
    # Fallback: use n colors
    return list(range(n))
# EVOLVE-BLOCK-END
