"""Optimized K2,3-free graph construction on 20 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    
    # Map edge (u, v) with u < v to an index
    edges = []
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = idx
            edges.append((u, v))
            idx += 1
            
    m = len(edges) # 190
    
    # Variables:
    # 0 .. m-1: x_{uv} (edge indicators)
    # m .. m + 3*m - 1: y_{uvw} (common neighbor indicators)
    # For each pair (u, v) and each w != u, v, we have a y variable.
    # Let's index y variables systematically.
    # y_idx[(u, v, w)] where u < v and w not in {u, v}
    
    y_vars = []
    y_index = {}
    y_idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w == u or w == v:
                    continue
                y_index[(u, v, w)] = y_idx
                y_vars.append((u, v, w))
                y_idx += 1
                
    total_vars = m + len(y_vars)
    
    # Objective: Maximize sum of x_{uv}
    # In scipy milp, we minimize. So minimize -sum(x_{uv}).
    c = np.zeros(total_vars)
    for i in range(m):
        c[i] = -1.0
        
    # Bounds: 0 <= x, y <= 1
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    bounds = Bounds(lb, ub)
    
    # Constraints:
    # 1. y_{uvw} <= x_{uv}
    # 2. y_{uvw} <= x_{vw}
    # 3. y_{uvw} >= x_{uv} + x_{vw} - 1  =>  -x_{uv} - x_{vw} + y_{uvw} >= -1
    # 4. sum_{w} y_{uvw} <= 2 for all u < v
    
    # We will build the A matrix and b vector.
    # A @ var <= b
    # Number of constraints:
    # 3 * len(y_vars) + m
    num_constraints = 3 * len(y_vars) + m
    
    A = np.zeros((num_constraints, total_vars))
    b = np.zeros(num_constraints)
    
    row = 0
    for k, (u, v, w) in enumerate(y_vars):
        y_var_idx = m + k
        uv_idx = edge_index[(u, v)]
        # Note: if w > v, edge is (v, w). If w < v, edge is (w, v).
        # We need to get the correct edge index for (v, w).
        if v < w:
            vw_idx = edge_index[(v, w)]
        else:
            vw_idx = edge_index[(w, v)]
            
        # Constraint 1: y_{uvw} - x_{uv} <= 0
        A[row, y_var_idx] = 1.0
        A[row, uv_idx] = -1.0
        b[row] = 0.0
        row += 1
        
        # Constraint 2: y_{uvw} - x_{vw} <= 0
        A[row, y_var_idx] = 1.0
        A[row, vw_idx] = -1.0
        b[row] = 0.0
        row += 1
        
        # Constraint 3: -x_{uv} - x_{vw} + y_{uvw} >= -1  =>  x_{uv} + x_{vw} - y_{uvw} <= 1
        A[row, uv_idx] = 1.0
        A[row, vw_idx] = 1.0
        A[row, y_var_idx] = -1.0
        b[row] = 1.0
        row += 1
        
    # Constraints 4: sum_{w} y_{uvw} <= 2 for each pair (u, v)
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w == u or w == v:
                    continue
                y_var_idx = m + y_index[(u, v, w)]
                A[row, y_var_idx] = 1.0
            b[row] = 2.0
            row += 1
            
    constraints = LinearConstraint(A, lb=np.full(num_constraints, -np.inf), ub=b)
    
    # Integrality: all variables are binary
    integrality = np.ones(total_vars)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 130, "mip_rel_gap": 0.01})
        
        if res.success or res.status == 8: # 8 is "solution found, but no optimality proof" (HiGHS)
            x = res.x[:m]
            # Round to nearest integer
            x = np.round(x).astype(int)
            
            # Verify and construct edges
            valid_edges = []
            for i, (u, v) in enumerate(edges):
                if x[i] == 1:
                    valid_edges.append([u, v])
            
            # Verify K2,3-free property
            adj = [0] * n
            for u, v in valid_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            def is_k23_free(adj):
                for u in range(n):
                    for v in range(u + 1, n):
                        if bin(adj[u] & adj[v]).count('1') >= 3:
                            return False
                return True
                
            if is_k23_free(adj):
                return valid_edges
            else:
                # Fallback to heuristic if MILP returns invalid (shouldn't happen if constraints are correct)
                pass
                
    except Exception as e:
        print(f"MILP failed: {e}", file=sys.stderr)
        
    # Fallback heuristic if MILP fails or times out without solution
    # Reuse logic from previous version but faster
    def popcount(x):
        return bin(x).count('1')
        
    def can_add_edge(adj, u, v):
        if u == v or ((adj[u] >> v) & 1):
            return False
        if popcount(adj[u] & adj[v]) >= 3:
            return False
        for y in range(n):
            if y == u or y == v: continue
            if (adj[u] >> y) & 1:
                if popcount(adj[v] & adj[y]) >= 2:
                    return False
        for x in range(n):
            if x == u or x == v: continue
            if (adj[v] >> x) & 1:
                if popcount(adj[u] & adj[x]) >= 2:
                    return False
        return True
        
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    adj = [0] * n
    rng = random.Random(42)
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if can_add_edge(adj, u, v):
            add_edge(adj, u, v)
            
    # Try to improve
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    changed = True
                    break
            if changed: break
            
    valid_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                valid_edges.append([u, v])
                
    return valid_edges

# EVOLVE-BLOCK-END
