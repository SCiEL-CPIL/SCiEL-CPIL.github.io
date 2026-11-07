import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
            
    candidate_edges = []
    edge_index = {}
    for e in all_edges:
        if e not in base_edges:
            edge_index[e] = len(candidate_edges)
            candidate_edges.append(e)
            
    m = len(candidate_edges)
    
    c = np.zeros(m + n * n)
    c[:m] = 1.0
    
    A_rows = []
    lb = []
    ub = []
    
    # 1. Triangle-free
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                row = np.zeros(m + n * n)
                count = 0
                for e in [(u, v), (v, w), (u, w)]:
                    e_norm = (min(e[0], e[1]), max(e[0], e[1]))
                    if e_norm in base_edges:
                        count += 1
                    elif e_norm in edge_index:
                        row[edge_index[e_norm]] = 1.0
                if count > 2:
                    return []
                A_rows.append(row)
                lb.append(-np.inf)
                ub.append(2 - count)
                
    # 2. Distance constraints
    for a in range(n):
        for b in range(n):
            d_idx = a * n + b
            if a == b:
                row = np.zeros(m + n * n)
                row[d_idx] = 1.0
                A_rows.append(row)
                lb.append(0)
                ub.append(0)
            else:
                # d_ab <= 4
                row = np.zeros(m + n * n)
                row[d_idx] = 1.0
                A_rows.append(row)
                lb.append(-np.inf)
                ub.append(4)
                
                # d_ab >= 1
                row = np.zeros(m + n * n)
                row[d_idx] = 1.0
                A_rows.append(row)
                lb.append(1)
                ub.append(np.inf)
                
                # Symmetry d_ab = d_ba for a < b
                if a < b:
                    row = np.zeros(m + n * n)
                    row[d_idx] = 1.0
                    row[b * n + a] = -1.0
                    A_rows.append(row)
                    lb.append(0)
                    ub.append(0)
                
                # Triangle inequality: d_ab <= d_au + y_au
                for u in range(n):
                    if u == b:
                        continue
                    d_au_idx = a * n + u
                    row = np.zeros(m + n * n)
                    row[d_idx] = 1.0
                    row[d_au_idx] = -1.0
                    e_norm = (min(a, u), max(a, u))
                    base_val = 0
                    if e_norm in base_edges:
                        base_val = 1
                    elif e_norm in edge_index:
                        row[edge_index[e_norm]] = -1.0
                    A_rows.append(row)
                    lb.append(-np.inf)
                    ub.append(base_val)
                    
    A = np.array(A_rows) if A_rows else np.empty((0, m + n * n))
    constraints = LinearConstraint(A, lb, ub)
    
    integrality = np.zeros(m + n * n)
    integrality[:m] = 1
    
    lbounds = np.zeros(m + n * n)
    ubounds = np.ones(m + n * n)
    for a in range(n):
        for b in range(n):
            d_idx = a * n + b
            if a == b:
                lbounds[d_idx] = 0
                ubounds[d_idx] = 0
            else:
                lbounds[d_idx] = 1
                ubounds[d_idx] = 4
                
    bounds = Bounds(lbounds, ubounds)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={'time_limit': 80})
        if res.status == 0:
            x_vals = res.x[:m]
            added_edges = []
            for i, val in enumerate(x_vals):
                if val > 0.5:
                    u, v = candidate_edges[i]
                    added_edges.append([int(u), int(v)])
            return added_edges
    except Exception:
        pass
        
    return []
