import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_valid(coloring):
    """Check if coloring is valid: adjacent differ, and no bicolored cycles."""
    n = P['n']
    edges = P['edges']
    
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    
    colors_used = set(coloring)
    
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            
            parent = list(range(n))
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[rx] = ry
                return False
            
            has_cycle = False
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if union(u, v):
                        has_cycle = True
                        break
            
            if has_cycle:
                return False
    
    return True

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _try_structural_coloring(max_colors):
    """
    Try a greedy structural coloring based on the ladder-like structure.
    """
    n = P['n']
    adj = _build_adj()
    edges = P['edges']
    
    patterns = []
    
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append((col + row) % k)
        patterns.append(coloring)
    
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append((col * 2 + row) % k)
        patterns.append(coloring)
    
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append((col + 2 * row) % k)
        patterns.append(coloring)
    
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append(row % k)
        patterns.append(coloring)
    
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append(col % k)
        patterns.append(coloring)
    
    for coloring in patterns:
        if _check_valid(coloring):
            used = set(coloring)
            if len(used) <= max_colors:
                return coloring
    
    return None

def _milp_solve(k, deadline):
    """
    Solve MILP for exactly k colors with bicolored-cycle-free constraint.
    
    Variables:
      x[v, c] = 1 if vertex v gets color c
      y[e, c1, c2] = 1 if edge e is in the bicolored subgraph for colors c1, c2
    
    Constraints:
      1. Each vertex gets exactly one color: sum_c x[v,c] = 1
      2. Adjacent vertices differ: x[u,c] + x[v,c] <= 1 for each edge (u,v), color c
      3. No bicolored cycles: for each pair (c1,c2) and each cycle in the graph,
         the sum of y over edges in the cycle <= cycle_length - 1
         (This is the standard cycle-inequality approach for forest constraints)
    
    For the forest constraint on bicolored subgraphs, we use the fact that a graph
    on m vertices is a forest iff it has at most m - 1 edges. So for each color pair
    (c1, c2), the number of edges with both endpoints colored c1 or c2 must be <= 
    (number of vertices colored c1 or c2) - 1.
    
    We use scipy.optimize.milp with HiGHS solver.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np
    
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Variable indexing:
    # x[v, c] for v in 0..n-1, c in 0..k-1: n*k variables
    # Total variables: n*k
    
    nvars = n * k
    
    def x_idx(v, c):
        return v * k + c
    
    # Objective: minimize sum of x[v,c] (should be n for valid solutions, but we want to
    # encourage compactness). Actually, we just need feasibility. Set c=0 for all.
    c_obj = np.zeros(nvars)
    
    # Build constraint matrix
    # We'll build rows incrementally
    
    A_rows = []
    lb_list = []
    ub_list = []
    
    # Constraint 1: For each vertex v, sum_c x[v,c] = 1
    for v in range(n):
        row = np.zeros(nvars)
        for c in range(k):
            row[x_idx(v, c)] = 1.0
        A_rows.append(row)
        lb_list.append(1.0)
        ub_list.append(1.0)
    
    # Constraint 2: For each edge (u,v) and color c, x[u,c] + x[v,c] <= 1
    for u, v in edges:
        for c in range(k):
            row = np.zeros(nvars)
            row[x_idx(u, c)] = 1.0
            row[x_idx(v, c)] = 1.0
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(1.0)
    
    # Constraint 3: Bicolored forest constraint.
    # For each pair (c1, c2) with c1 < c2:
    #   Let S_{c1,c2} = {v : x[v,c1]=1 or x[v,c2]=1}
    #   Let E_{c1,c2} = {(u,v) in edges : u,v in S_{c1,c2}}
    #   We need |E_{c1,c2}| <= |S_{c1,c2}| - 1
    #
    # This is nonlinear. Linearization:
    # Introduce y[e, c1, c2] = 1 if edge e is "active" for color pair (c1,c2),
    # i.e., both endpoints are colored c1 or c2.
    # y[e, c1, c2] >= x[u, c1] + x[v, c1] - 1
    # y[e, c1, c2] >= x[u, c2] + x[v, c2] - 1
    # y[e, c1, c2] >= x[u, c1] + x[v, c2] - 1
    # y[e, c1, c2] >= x[u, c2] + x[v, c1] - 1
    # y[e, c1, c2] <= x[u, c1] + x[v, c1]
    # y[e, c1, c2] <= x[u, c2] + x[v, c2]
    # y[e, c1, c2] <= x[u, c1] + x[v, c2]
    # y[e, c1, c2] <= x[u, c2] + x[v, c1]
    #
    # Then: sum_e y[e, c1, c2] <= sum_v (x[v,c1] + x[v,c2]) - 1
    #
    # But wait, this counts vertices that could be either color. The issue is that
    # a vertex colored c1 contributes to S_{c1,c2} once, not twice.
    # 
    # Better: introduce z[v, c1, c2] = 1 if vertex v is in S_{c1,c2}
    # z[v, c1, c2] >= x[v, c1]
    # z[v, c1, c2] >= x[v, c2]
    # z[v, c1, c2] <= x[v, c1] + x[v, c2]
    #
    # Then: sum_e y[e, c1, c2] <= sum_v z[v, c1, c2] - 1
    #
    # But we also need y to correctly identify edges where BOTH endpoints are in S.
    # Actually, an edge is in E_{c1,c2} iff both endpoints are in S_{c1,c2}.
    # y[e, c1, c2] = 1 iff (x[u,c1]+x[u,c2]) >= 1 AND (x[v,c1]+x[v,c2]) >= 1
    #
    # So: y[e, c1, c2] >= (x[u,c1]+x[u,c2]) + (x[v,c1]+x[v,c2]) - 1
    #     y[e, c1, c2] <= x[u,c1] + x[u,c2]
    #     y[e, c1, c2] <= x[v,c1] + x[v,c2]
    #
    # And z[v, c1, c2] >= x[v,c1], z[v,c1,c2] >= x[v,c2], z[v,c1,c2] <= x[v,c1]+x[v,c2]
    #
    # Forest constraint: sum_e y[e,c1,c2] <= sum_v z[v,c1,c2] - 1
    #
    # Number of additional variables: m * k*(k-1)/2 for y, n * k*(k-1)/2 for z
    # For n=50, m~100, k=3: y has 100*3=300, z has 50*3=150. Total ~450 extra.
    # For k=4: y has 100*6=600, z has 50*6=300. Total ~900 extra.
    # For k=5: y has 100*10=1000, z has 50*10=500. Total ~1500 extra.
    
    # Let's compute total variables
    num_pairs = k * (k - 1) // 2
    nvars_y = m * num_pairs
    nvars_z = n * num_pairs
    total_vars = nvars + nvars_y + nvars_z
    
    if total_vars > 5000:
        return None  # Too large for practical solving
    
    def y_idx(e, c1, c2):
        # e in 0..m-1, pair index for (c1,c2) where c1 < c2
        pair_idx = c1 * (k - 1) - c1 * (c1 + 1) // 2 + (c2 - c1 - 1)
        # Actually let me use a simpler mapping
        pairs = []
        for a in range(k):
            for b in range(a + 1, k):
                pairs.append((a, b))
        pi = pairs.index((c1, c2))
        return nvars + e * num_pairs + pi
    
    def z_idx(v, c1, c2):
        pairs = []
        for a in range(k):
            for b in range(a + 1, k):
                pairs.append((a, b))
        pi = pairs.index((c1, c2))
        return nvars + nvars_y + v * num_pairs + pi
    
    # Precompute pair list
    pairs = []
    for a in range(k):
        for b in range(a + 1, k):
            pairs.append((a, b))
    
    def y_idx2(e, pi):
        return nvars + e * num_pairs + pi
    
    def z_idx2(v, pi):
        return nvars + nvars_y + v * num_pairs + pi
    
    # Reset A_rows for full problem
    A_rows = []
    lb_list = []
    ub_list = []
    
    # Constraint 1: For each vertex v, sum_c x[v,c] = 1
    for v in range(n):
        row = np.zeros(total_vars)
        for c in range(k):
            row[x_idx(v, c)] = 1.0
        A_rows.append(row)
        lb_list.append(1.0)
        ub_list.append(1.0)
    
    # Constraint 2: For each edge (u,v) and color c, x[u,c] + x[v,c] <= 1
    for u, v in edges:
        for c in range(k):
            row = np.zeros(total_vars)
            row[x_idx(u, c)] = 1.0
            row[x_idx(v, c)] = 1.0
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(1.0)
    
    # Constraints for y and z variables
    for pi, (c1, c2) in enumerate(pairs):
        # z[v, c1, c2] >= x[v, c1]
        for v in range(n):
            row = np.zeros(total_vars)
            row[z_idx2(v, pi)] = 1.0
            row[x_idx(v, c1)] = -1.0
            A_rows.append(row)
            lb_list.append(0.0)
            ub_list.append(np.inf)
        
        # z[v, c1, c2] >= x[v, c2]
        for v in range(n):
            row = np.zeros(total_vars)
            row[z_idx2(v, pi)] = 1.0
            row[x_idx(v, c2)] = -1.0
            A_rows.append(row)
            lb_list.append(0.0)
            ub_list.append(np.inf)
        
        # z[v, c1, c2] <= x[v, c1] + x[v, c2]
        for v in range(n):
            row = np.zeros(total_vars)
            row[z_idx2(v, pi)] = 1.0
            row[x_idx(v, c1)] = -1.0
            row[x_idx(v, c2)] = -1.0
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(0.0)
        
        # y[e, c1, c2] >= (x[u,c1]+x[u,c2]) + (x[v,c1]+x[v,c2]) - 1
        for e, (u, v) in enumerate(edges):
            row = np.zeros(total_vars)
            row[y_idx2(e, pi)] = 1.0
            row[x_idx(u, c1)] = -1.0
            row[x_idx(u, c2)] = -1.0
            row[x_idx(v, c1)] = -1.0
            row[x_idx(v, c2)] = -1.0
            A_rows.append(row)
            lb_list.append(-1.0)
            ub_list.append(np.inf)
        
        # y[e, c1, c2] <= x[u,c1] + x[u,c2]
        for e, (u, v) in enumerate(edges):
            row = np.zeros(total_vars)
            row[y_idx2(e, pi)] = 1.0
            row[x_idx(u, c1)] = -1.0
            row[x_idx(u, c2)] = -1.0
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(0.0)
        
        # y[e, c1, c2] <= x[v,c1] + x[v,c2]
        for e, (u, v) in enumerate(edges):
            row = np.zeros(total_vars)
            row[y_idx2(e, pi)] = 1.0
            row[x_idx(v, c1)] = -1.0
            row[x_idx(v, c2)] = -1.0
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(0.0)
        
        # Forest constraint: sum_e y[e, c1, c2] <= sum_v z[v, c1, c2] - 1
        row = np.zeros(total_vars)
        for e in range(m):
            row[y_idx2(e, pi)] = 1.0
        for v in range(n):
            row[z_idx2(v, pi)] = -1.0
        A_rows.append(row)
        lb_list.append(-np.inf)
        ub_list.append(-1.0)
    
    # Build constraint matrix
    num_constraints = len(A_rows)
    if num_constraints == 0:
        return None
    
    A_matrix = np.array(A_rows, dtype=np.float64)
    lb = np.array(lb_list, dtype=np.float64)
    ub = np.array(ub_list, dtype=np.float64)
    
    # Variable bounds: all binary [0, 1]
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    
    # Integrality: all binary
    integrality = np.ones(total_vars)
    
    # Objective: minimize 0 (feasibility problem)
    c_obj = np.zeros(total_vars)
    
    constraint = LinearConstraint(A_matrix, lb, ub)
    
    try:
        result = milp(
            c=c_obj,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds
        )
    except Exception:
        return None
    
    if not result.success:
        return None
    
    x_vals = result.x
    
    # Extract coloring
    coloring = []
    for v in range(n):
        for c in range(k):
            if x_vals[x_idx(v, c)] > 0.5:
                coloring.append(c)
                break
        else:
            return None
    
    # Verify
    if _check_valid(coloring):
        return coloring
    
    return None

def _find_coloring_csp(max_colors, deadline):
    """CSP-based search with MRV heuristic and fast incremental bicolored-cycle check."""
    n = P['n']
    adj = _build_adj()
    edges = P['edges']
    
    coloring = [-1] * n
    
    def check_bicolored_fast(vertex, color):
        used_colors = set()
        for v in range(n):
            if coloring[v] != -1:
                used_colors.add(coloring[v])
        
        for c2 in used_colors:
            if c2 == color:
                continue
            
            parent = {}
            
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[rx] = ry
                return False
            
            has_cycle = False
            for u, v in edges:
                cu = coloring[u] if u != vertex else color
                cv = coloring[v] if v != vertex else color
                if (cu == color or cu == c2) and (cv == color or cv == c2):
                    if union(u, v):
                        has_cycle = True
                        break
            
            if has_cycle:
                return True
        
        return False
    
    deg = [len(adj[i]) for i in range(n)]
    
    def get_mrv_vertex():
        best_v = -1
        best_count = n + 1
        for v in range(n):
            if coloring[v] != -1:
                continue
            neighbor_colors = set()
            for nb in adj[v]:
                if coloring[nb] != -1:
                    neighbor_colors.add(coloring[nb])
            avail = max_colors - len(neighbor_colors)
            if avail < best_count:
                best_count = avail
                best_v = v
                if avail <= 1:
                    break
        return best_v
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        v = get_mrv_vertex()
        if v == -1:
            return True
        
        neighbor_colors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                neighbor_colors.add(coloring[nb])
        
        color_order = list(range(max_colors))
        color_order.sort(key=lambda c: (c in neighbor_colors, c))
        
        for color in color_order:
            if color in neighbor_colors:
                continue
            
            coloring[v] = color
            
            if not check_bicolored_fast(v, color):
                if backtrack():
                    return True
            
            coloring[v] = -1
        
        return False
    
    if backtrack():
        return coloring
    return None

def construct():
    deadline = time.time() + 85
    
    # First try structural colorings (very fast)
    for num_colors in range(3, 15):
        if time.time() > deadline - 10:
            break
        coloring = _try_structural_coloring(num_colors)
        if coloring is not None:
            if _check_valid(coloring):
                return [int(c) for c in coloring]
    
    # Try MILP for small k
    for num_colors in range(3, 8):
        if time.time() > deadline - 30:
            break
        coloring = _milp_solve(num_colors, deadline)
        if coloring is not None:
            if _check_valid(coloring):
                return [int(c) for c in coloring]
    
    # Then try CSP search
    for num_colors in range(3, 15):
        if time.time() > deadline - 5:
            break
        coloring = _find_coloring_csp(num_colors, deadline)
        if coloring is not None:
            if _check_valid(coloring):
                return [int(c) for c in coloring]
    
    # Fallback: identity coloring
    return [int(i) for i in range(P['n'])]

if __name__ == '__main__':
    result = construct()
    print(result)
    print(len(set(result)))
