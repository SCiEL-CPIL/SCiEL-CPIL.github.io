import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS
SCHOOLBOOK = N * M * P  # 72

def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        for a in range(N * M):
            for b in range(M * P):
                acc = [0.0] * (P * N)
                for c in range(rank):
                    u = U[a][c]
                    if u == 0:
                        continue
                    v = V[b][c]
                    if v == 0:
                        continue
                    uv = u * v
                    if uv == 0:
                        continue
                    wc = W[c]
                    for k in range(P * N):
                        acc[k] += uv * wc[k]
                i, j = divmod(a, M)
                j2, k2 = divmod(b, P)
                for kk in range(P * N):
                    expected = 1.0 if (kk % N == i and kk // N == j2) else 0.0
                    if abs(acc[kk] - expected) > tol:
                        return False
        return True
    except Exception:
        return False

def _generate_candidates():
    """
    Generate a dictionary of candidate rank-1 terms.
    Each candidate is a tuple (u_vec, v_vec, w_vec) where u_vec has length N*M, 
    v_vec has length M*P, w_vec has length P*N.
    We include:
    1. Schoolbook terms: u_{i,j}, v_{j,k}, w_{k,i}
    2. Strassen-like terms for all 2x2 sub-blocks.
    """
    candidates = []
    
    # 1. Schoolbook terms
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = np.zeros(N*M)
                v = np.zeros(M*P)
                w = np.zeros(P*N)
                u[i*M + j] = 1
                v[j*P + k] = 1
                w[k*N + i] = 1
                candidates.append((u, v, w))
    
    # 2. Strassen-like terms for all 2x2 sub-blocks
    # Strassen for 2x2 * 2x2 -> 2x2
    # There are 7 terms.
    # We map local indices (0..3) to global indices.
    
    # Define Strassen patterns for 2x2.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Local indices for A: 0:a, 1:b, 2:c, 3:d
    # Local indices for B: 0:e, 1:f, 2:g, 3:h
    # Local indices for C: 0:c00, 1:c01, 2:c10, 3:c11
    
    # M1
    u1 = np.array([1, 0, 0, 1])
    v1 = np.array([1, 0, 0, 1])
    w1 = np.array([1, -1, -1, 1])
    
    # M2
    u2 = np.array([0, 0, 1, 1])
    v2 = np.array([1, 0, 0, 0])
    w2 = np.array([0, -1, 1, 0])
    
    # M3
    u3 = np.array([1, 0, 0, 0])
    v3 = np.array([0, 1, 0, -1])
    w3 = np.array([0, 1, 0, 1])
    
    # M4
    u4 = np.array([0, 0, 0, 1])
    v4 = np.array([0, 0, 1, -1])
    w4 = np.array([1, 0, 1, 0])
    
    # M5
    u5 = np.array([1, 1, 0, 0])
    v5 = np.array([0, 0, 0, 1])
    w5 = np.array([-1, 1, 0, 0])
    
    # M6
    u6 = np.array([-1, 0, 1, 0])
    v6 = np.array([1, 1, 0, 0])
    w6 = np.array([0, 0, 0, 1])
    
    # M7
    u7 = np.array([0, 1, 0, -1])
    v7 = np.array([0, 0, 1, 1])
    w7 = np.array([1, 0, -1, 0])
    
    strassen_terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Iterate over all possible 2x2 sub-block placements
    for i0 in range(N-1):
        for j0 in range(M-1):
            for k0 in range(P-1):
                # Map local indices to global indices
                # A: (i0..i0+1, j0..j0+1) -> U indices
                # B: (j0..j0+1, k0..k0+1) -> V indices
                # C: (k0..k0+1, i0..i0+1) -> W indices
                
                for u_loc, v_loc, w_loc in strassen_terms:
                    u_global = np.zeros(N*M)
                    v_global = np.zeros(M*P)
                    w_global = np.zeros(P*N)
                    
                    # Fill U
                    for dl in range(4):
                        val = u_loc[dl]
                        if val != 0:
                            li, lj = divmod(dl, 2) # local i, j
                            gi = i0 + li
                            gj = j0 + lj
                            u_global[gi*M + gj] += val
                            
                    # Fill V
                    for dl in range(4):
                        val = v_loc[dl]
                        if val != 0:
                            lj, lk = divmod(dl, 2) # local j, k
                            gj = j0 + lj
                            gk = k0 + lk
                            v_global[gj*P + gk] += val
                            
                    # Fill W
                    for dl in range(4):
                        val = w_loc[dl]
                        if val != 0:
                            lk, li = divmod(dl, 2) # local k, i
                            gk = k0 + lk
                            gi = i0 + li
                            w_global[gk*N + gi] += val
                            
                    candidates.append((u_global, v_global, w_global))
                    
    return candidates

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # 1. Generate candidates
    candidates = _generate_candidates()
    num_candidates = len(candidates)
    
    # 2. Formulate BIP
    # We want to minimize sum x_i subject to A x = b
    # where A is a matrix of size (N*M*M*P) x num_candidates
    # b is the target tensor (flattened)
    
    # The tensor T_{a,b,c} = sum_i x_i * U_{a,i} * V_{b,i} * W_{c,i}
    # This is not linear in x if we treat U, V, W as variables.
    # But here, U, V, W are fixed for each candidate i.
    # So T_{a,b,c} = sum_i x_i * (U_i[a] * V_i[b] * W_i[c])
    # This IS linear in x_i.
    
    # Construct A matrix
    # Rows indexed by (a, b, c)
    # Cols indexed by candidate i
    
    # Size: (9*24*24) = 5184 rows, num_candidates cols
    # num_candidates is roughly 72 + 7 * (2*2*7) = 72 + 196 = 268
    
    A = np.zeros((N*M*M*P, num_candidates))
    
    for i, (u, v, w) in enumerate(candidates):
        # For each (a, b, c), A[a, b, c, i] = u[a] * v[b] * w[c]
        # We can do this efficiently with outer products
        # But 5184 * 268 is small, so loop is fine.
        for a in range(N*M):
            ua = u[a]
            if ua == 0: continue
            for b in range(M*P):
                vb = v[b]
                if vb == 0: continue
                uv = ua * vb
                if uv == 0: continue
                for c in range(P*N):
                    A[a * (M*P) + b * P + c, i] = uv * w[c]
                    
    # Target tensor b
    b = np.zeros(N*M*M*P)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i*M + j
                bb = j*P + k
                c = k*N + i
                b[a * (M*P) + bb * P + c] = 1.0
                
    # Solve BIP
    # Minimize x^T 1
    # Subject to A x = b
    # x >= 0 (and ideally integer, but let's try LP first to get lower bound, then round?)
    # Actually, the task requires exact decomposition with integer/half-integer entries.
    # The BIP ensures x are 0/1 if we enforce integrality.
    # SciPy linprog does not support integer constraints directly.
    # We can use `milp` from scipy.optimize if available (SciPy >= 1.7)
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        # Fallback to schoolbook if milp not available
        rank = SCHOOLBOOK
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        column = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][column] = 1
                    V[j * P + k][column] = 1
                    W[k * N + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    # milp formulation
    c = np.ones(num_candidates)
    constraints = [LinearConstraint(A, b, b)]
    integrality = np.ones(num_candidates) # binary variables
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        x = res.x
        # Filter to selected candidates
        selected = np.where(x > 0.5)[0]
        rank = len(selected)
        
        if rank == 0:
            # Fallback
            rank = SCHOOLBOOK
            U = [[0] * rank for _ in range(N * M)]
            V = [[0] * rank for _ in range(M * P)]
            W = [[0] * rank for _ in range(P * N)]
            column = 0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        U[i * M + j][column] = 1
                        V[j * P + k][column] = 1
                        W[k * N + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
            
        U = np.zeros((N*M, rank))
        V = np.zeros((M*P, rank))
        W = np.zeros((P*N, rank))
        
        for new_idx, old_idx in enumerate(selected):
            u, v, w = candidates[old_idx]
            U[:, new_idx] = u
            V[:, new_idx] = v
            W[:, new_idx] = w
            
        # Convert to lists for return
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook if BIP fails or verification fails
    rank = SCHOOLBOOK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][column] = 1
                V[j * P + k][column] = 1
                W[k * N + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
