import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _is_not_3_colorable(adj):
    """
    Checks if the graph represented by adj (list of ints) is NOT 3-colorable.
    Returns True if not 3-colorable, False if 3-colorable.
    Also returns the coloring if it is 3-colorable.
    """
    n = 16
    color = [-1] * n
    
    # Order vertices by degree descending for better pruning
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    
    def backtrack(idx):
        if idx == n:
            return True, color[:]
        v = order[idx]
        # Determine available colors
        neighbor_colors = set()
        mask = adj[v]
        for u in range(n):
            if (mask >> u) & 1:
                if color[u] != -1:
                    neighbor_colors.add(color[u])
        
        for c in range(3):
            if c not in neighbor_colors:
                color[v] = c
                found, col = backtrack(idx + 1)
                if found:
                    return True, col
                color[v] = -1
        return False, None

    # Symmetry breaking: fix the highest degree vertex to color 0
    color[order[0]] = 0
    is_not_3c, col = backtrack(1)
    return is_not_3c, col

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110 # 10s safety margin
    
    # Generate all possible edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))
    m = len(edges)
    
    # Map edge index to (u, v)
    edge_idx = {e: i for i, e in enumerate(edges)}
    
    # Generate all triangles
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triangles.append((i, j, k))
                
    # ILP Setup
    # Variables x[0..m-1] in [0, 1]
    
    # Objective: Maximize sum(x) -> Minimize -sum(x)
    c = np.full(m, -1.0)
    
    # Bounds
    bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
    
    # Constraints: Triangle-free
    # For each triangle (u, v, w), x_uv + x_vw + x_wu <= 2
    A_tri = np.zeros((len(triangles), m))
    for t_idx, (u, v, w) in enumerate(triangles):
        e1 = edge_idx[(u, v)]
        e2 = edge_idx[(v, w)]
        e3 = edge_idx[(u, w)]
        A_tri[t_idx, e1] = 1
        A_tri[t_idx, e2] = 1
        A_tri[t_idx, e3] = 1
    
    lb_tri = np.full(len(triangles), -np.inf)
    ub_tri = np.full(len(triangles), 2.0)
    
    constraints = [
        LinearConstraint(A_tri, lb_tri, ub_tri)
    ]
    
    best_edges_list = []
    best_count = 0
    
    # Initial solution: Clebsch graph
    S = [1, 2, 4, 8, 15]
    init_adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if (u ^ v) in S:
                init_adj[u] |= (1 << v)
                init_adj[v] |= (1 << u)
                e_idx = edge_idx[(u, v)]
                # We will use this to seed the ILP or just as a fallback
                best_edges_list.append([u, v])
                best_count += 1
                
    # Check if Clebsch is not 3-colorable (it is)
    is_not_3c, _ = _is_not_3_colorable(init_adj)
    if is_not_3c:
        best_count = len(best_edges_list)
    else:
        # Fallback to empty if Clebsch fails (should not happen)
        best_edges_list = []
        best_count = 0

    # Cutting Plane Loop
    while time.time() < deadline:
        # Solve ILP
        try:
            res = milp(c=c, constraints=constraints, integrality=np.ones(m), bounds=bounds)
        except Exception:
            break
            
        if not res.success:
            break
            
        x = res.x
        # Extract edges
        current_edges = []
        current_adj = [0] * n
        for i, (u, v) in enumerate(edges):
            if x[i] > 0.5:
                current_edges.append([u, v])
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
        # Check 3-colorability
        is_not_3c, coloring = _is_not_3_colorable(current_adj)
        
        if is_not_3c:
            # Found a valid solution
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges_list = current_edges[:]
            # If this is the optimal triangle-free graph and it is not 3-colorable,
            # it is the global optimum for the constrained problem.
            # However, there might be other triangle-free graphs with same edge count?
            # Or maybe we can stop if we hit the Mantel bound?
            # For now, we continue to see if we can find more? No, ILP maximizes edges.
            # So this is the max triangle-free graph. If it's not 3-colorable, it's the best.
            # We can break.
            break
        else:
            # It is 3-colorable. Add a cut.
            # The cut forbids this specific coloring.
            # Coloring: list of 16 colors (0, 1, 2)
            # S0, S1, S2 are sets of vertices with color 0, 1, 2
            # Constraint: Sum of x_uv for (u,v) in S0 + Sum for S1 + Sum for S2 >= 1
            
            S0 = [v for v in range(n) if coloring[v] == 0]
            S1 = [v for v in range(n) if coloring[v] == 1]
            S2 = [v for v in range(n) if coloring[v] == 2]
            
            row = np.zeros(m)
            for u in S0:
                for v in S0:
                    if u < v:
                        row[edge_idx[(u, v)]] = 1
            for u in S1:
                for v in S1:
                    if u < v:
                        row[edge_idx[(u, v)]] = 1
            for u in S2:
                for v in S2:
                    if u < v:
                        row[edge_idx[(u, v)]] = 1
                        
            # If no edges are possible in any color class (e.g. singleton sets),
            # the constraint is 0 >= 1, which is infeasible.
            # This means the graph structure forces this coloring?
            # No, it means we cannot add an edge within these classes.
            # If the sum of coefficients is 0, the constraint is 0 >= 1, infeasible.
            # This implies no triangle-free graph can avoid this coloring?
            # Actually, if S_i has size 1, there are no edges inside.
            # If all S_i have size 1, impossible for N=16.
            # If some S_i have size >= 2, there are potential edges.
            
            if np.sum(row) == 0:
                # Infeasible cut? This shouldn't happen for N=16 unless the coloring classes are singletons.
                # If it happens, break.
                break
                
            new_constraint = LinearConstraint(row[np.newaxis, :], 1.0, np.inf)
            constraints.append(new_constraint)
            
    return best_edges_list
