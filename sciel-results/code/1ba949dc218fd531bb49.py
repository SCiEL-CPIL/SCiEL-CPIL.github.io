"""Rewritten program: maximize edge-disjoint Hamiltonian cycles on the 21-vertex host."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = [tuple(e) for e in P['edges']]
    m = len(edges)
    
    # Max possible cycles
    max_k = m // n
    if max_k == 0:
        return []
        
    K = max_k
    
    # Variables layout:
    # z_k (K)
    # x_{e,k} (M*K)
    # y_{uv,k} (2*M*K)
    # w_{v,k} (N*K)
    
    n_z = K
    n_x = m * K
    n_y = 2 * m * K
    n_w = n * K
    nvar = n_z + n_x + n_y + n_w
    
    # Indices
    idx_z = lambda k: k
    idx_x = lambda e, k: n_z + k * m + e
    
    # Build arc indices
    arc_indices = {}
    arc_counter = 0
    for e, (a, b) in enumerate(edges):
        arc_indices[(a, b)] = arc_counter
        arc_counter += 1
        arc_indices[(b, a)] = arc_counter
        arc_counter += 1
        
    idx_y = lambda u, v, k: n_z + n_x + k * (2 * m) + arc_indices[(u, v)]
    idx_w = lambda v, k: n_z + n_x + n_y + k * n + v
    
    # Objective: Maximize sum z_k
    c = np.zeros(nvar)
    for k in range(K):
        c[idx_z(k)] = -1.0
        
    # Integrality
    integrality = np.zeros(nvar)
    integrality[:n_z] = 1
    integrality[n_z:n_z+n_x] = 1
    integrality[n_z+n_x:n_z+n_x+n_y] = 1
    integrality[n_z+n_x+n_y:] = 1
    
    lb = np.zeros(nvar)
    ub = np.ones(nvar)
    # w bounds
    for k in range(K):
        for v in range(n):
            ub[idx_w(v, k)] = n
            
    cons = []
    
    # 1. Edge disjointness: sum_k x_{e,k} <= 1
    for e in range(m):
        row = np.zeros(nvar)
        for k in range(K):
            row[idx_x(e, k)] = 1.0
        cons.append(LinearConstraint(row, 0, 1))
        
    # 2. Degree constraints: sum_{e in v} x_{e,k} = 2 * z_k
    for k in range(K):
        for v in range(n):
            row = np.zeros(nvar)
            for e, (a, b) in enumerate(edges):
                if a == v or b == v:
                    row[idx_x(e, k)] = 1.0
            row[idx_z(k)] = -2.0
            cons.append(LinearConstraint(row, 0, 0))
            
    # 3. Link x and y: y_{uv,k} + y_{vu,k} = x_{uv,k}
    for k in range(K):
        for e, (a, b) in enumerate(edges):
            row = np.zeros(nvar)
            row[idx_y(a, b, k)] = 1.0
            row[idx_y(b, a, k)] = 1.0
            row[idx_x(e, k)] = -1.0
            cons.append(LinearConstraint(row, 0, 0))
            
    # 4. MTZ constraints
    # w_{u,k} - w_{v,k} + 1 <= (n-1)(1 - y_{uv,k})
    # w_{u,k} - w_{v,k} + (n-1)y_{uv,k} <= n-2
    for k in range(K):
        # w_0,k = z_k
        row = np.zeros(nvar)
        row[idx_w(0, k)] = 1.0
        row[idx_z(k)] = -1.0
        cons.append(LinearConstraint(row, 0, 0))
        
        for e, (a, b) in enumerate(edges):
            # Arc a->b
            row = np.zeros(nvar)
            row[idx_w(a, k)] = 1.0
            row[idx_w(b, k)] = -1.0
            row[idx_y(a, b, k)] = n - 1
            cons.append(LinearConstraint(row, -np.inf, n - 2))
            
            # Arc b->a
            row = np.zeros(nvar)
            row[idx_w(b, k)] = 1.0
            row[idx_w(a, k)] = -1.0
            row[idx_y(b, a, k)] = n - 1
            cons.append(LinearConstraint(row, -np.inf, n - 2))
            
    res = milp(c=c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub))
    
    cycles = []
    if res.success and res.x is not None:
        for k in range(K):
            if res.x[idx_z(k)] > 0.5:
                # Reconstruct cycle k
                adjc = [set() for _ in range(n)]
                for e, (a, b) in enumerate(edges):
                    if res.x[idx_x(e, k)] > 0.5:
                        adjc[a].add(b)
                        adjc[b].add(a)
                # Walk
                start = 0
                cyc = [start]
                prev = -1
                cur = start
                for _ in range(n):
                    nxts = [u for u in adjc[cur] if u != prev]
                    if not nxts:
                        break
                    prev, cur = cur, nxts[0]
                    if cur == start:
                        break
                    cyc.append(cur)
                if len(cyc) == n:
                    cycles.append(cyc)
                    
    if not cycles:
        return []
        
    return [tuple(int(v) for v in c) for c in cycles]

# EVOLVE-BLOCK-END
