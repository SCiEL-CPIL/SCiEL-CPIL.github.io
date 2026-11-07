import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    n = 15
    vertices = list(range(n))
    all_edges = list(combinations(vertices, 2))
    m = len(all_edges)
    
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Generate all valid 3-paths s-u-v-t
    paths = []
    for u in range(n):
        for v in range(u + 1, n):
            for s in range(n):
                if s == u or s == v:
                    continue
                for t in range(n):
                    if t == u or t == v or t == s:
                        continue
                    e1 = (min(s, u), max(s, u))
                    e2 = (min(u, v), max(u, v))
                    e3 = (min(v, t), max(v, t))
                    
                    idx1 = edge_to_idx[e1]
                    idx2 = edge_to_idx[e2]
                    idx3 = edge_to_idx[e3]
                    
                    paths.append((s, u, v, t, idx1, idx2, idx3))
    
    p_count = len(paths)
    total_vars = m + n + 2 * p_count
    
    c = np.zeros(total_vars)
    for i in range(m):
        c[i] = -1.0
        
    M = 14
    total_ineq = 3 * p_count
    A = np.zeros((total_ineq + n, total_vars))
    lb = np.zeros(total_ineq + n)
    ub = np.zeros(total_ineq + n)
    
    # 1. Degree constraints (Equality)
    for v in range(n):
        row = v
        A[row, m + v] = -1.0
        for u in range(n):
            if u == v: continue
            e = (min(u, v), max(u, v))
            idx = edge_to_idx[e]
            A[row, idx] = 1.0
        lb[row] = 0.0
        ub[row] = 0.0
        
    # 2. Path Logic: x1 + x2 + x3 - z - w <= 2
    for k, (s, u, v, t, i1, i2, i3) in enumerate(paths):
        row = n + k
        A[row, i1] = 1.0
        A[row, i2] = 1.0
        A[row, i3] = 1.0
        z_idx = m + n + k
        w_idx = m + n + p_count + k
        A[row, z_idx] = -1.0
        A[row, w_idx] = -1.0
        lb[row] = -np.inf
        ub[row] = 2.0
        
    # 3. Big-M Constraints
    for k, (s, u, v, t, i1, i2, i3) in enumerate(paths):
        row_z = n + p_count + k
        row_w = n + 2 * p_count + k
        
        A[row_z, m + s] = 1.0
        A[row_z, m + t] = -1.0
        z_idx = m + n + k
        A[row_z, z_idx] = M
        lb[row_z] = -np.inf
        ub[row_z] = M - 1
        
        A[row_w, m + s] = -1.0
        A[row_w, m + t] = 1.0
        w_idx = m + n + p_count + k
        A[row_w, w_idx] = M
        lb[row_w] = -np.inf
        ub[row_w] = M - 1
        
    integrality = np.zeros(total_vars)
    integrality[:m] = 1
    integrality[m:m+n] = 1
    integrality[m+n:] = 1
    
    bounds = Bounds(
        lb=np.zeros(total_vars),
        ub=np.ones(total_vars)
    )
    bounds.ub[m:m+n] = 14
    
    start_time = time.time()
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': 50, 'mip_rel_gap': 0.0}
        )
    except Exception as e:
        edges = [[0, i] for i in range(1, n)]
        return edges

    if not res.success or res.x is None:
        edges = [[0, i] for i in range(1, n)]
        return edges
        
    x = res.x[:m]
    edges = []
    for i, e in enumerate(all_edges):
        if x[i] > 0.5:
            edges.append([e[0], e[1]])
            
    return edges
# EVOLVE-BLOCK-END
