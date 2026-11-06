import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    edge_indices = []
    for u in range(n):
        for v in range(u + 1, n):
            edge_indices.append((u, v))
    m_edges = len(edge_indices)
    
    edge_to_idx = {e: i for i, e in enumerate(edge_indices)}
    
    # Variables:
    # 0 .. m_edges-1: x_e (edge existence)
    # m_edges .. m_edges + z_count - 1: z_{i,j,k}
    
    # Count z variables
    z_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k != i and k != j:
                    z_count += 1
                    
    total_vars = m_edges + z_count
    
    c = np.zeros(total_vars)
    for i in range(m_edges):
        c[i] = -1.0 # Maximize edges => Minimize -sum
        
    # Constraints
    # 1. z <= x_ik
    # 2. z <= x_jk
    # 3. z >= x_ik + x_jk - 1  => x_ik + x_jk - z <= 1
    # 4. sum_k z_{i,j,k} <= 2
    
    num_cons = z_count * 3 + (n * (n - 1) // 2)
    A = np.zeros((num_cons, total_vars))
    b = np.zeros(num_cons)
    
    con_idx = 0
    z_idx = m_edges
    
    for i in range(n):
        for j in range(i + 1, n):
            # Track z indices for this pair to add sum constraint
            z_start = z_idx
            for k in range(n):
                if k == i or k == j:
                    continue
                    
                u1, v1 = sorted((i, k))
                u2, v2 = sorted((j, k))
                idx_ik = edge_to_idx[(u1, v1)]
                idx_jk = edge_to_idx[(u2, v2)]
                idx_z = z_idx
                z_idx += 1
                
                # C1: x_ik - z >= 0  => -x_ik + z <= 0
                A[con_idx, idx_ik] = -1.0
                A[con_idx, idx_z] = 1.0
                b[con_idx] = 0.0
                con_idx += 1
                
                # C2: x_jk - z >= 0  => -x_jk + z <= 0
                A[con_idx, idx_jk] = -1.0
                A[con_idx, idx_z] = 1.0
                b[con_idx] = 0.0
                con_idx += 1
                
                # C3: x_ik + x_jk - z <= 1
                A[con_idx, idx_ik] = 1.0
                A[con_idx, idx_jk] = 1.0
                A[con_idx, idx_z] = -1.0
                b[con_idx] = 1.0
                con_idx += 1
                
            # C4: sum z <= 2
            # z indices are from z_start to z_idx - 1 (18 of them)
            for zz in range(18):
                A[con_idx, z_start + zz] = 1.0
            b[con_idx] = 2.0
            con_idx += 1
            
    bounds = Bounds(lb=0.0, ub=1.0)
    integrality = np.ones(total_vars)
    constraints = LinearConstraint(A, ub=b)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, method='highs')
        if res.status == 0:
            x_sol = res.x[:m_edges]
            edges = []
            for i, (u, v) in enumerate(edge_indices):
                if x_sol[i] > 0.5:
                    edges.append([u, v])
            return edges
    except Exception:
        pass
    
    # Fallback: Simple greedy (just to avoid crash, though MILP should work)
    # This fallback is a minimal valid graph (empty or few edges)
    # Actually, let's return a valid small graph if MILP fails
    # Empty graph is valid
    return []
