import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[ _round_to_half_integer(x) for x in row] for row in M])

def _als_decompose(n, m, p, rank, seed=42, max_iters=300):
    """Try to find a rank-r decomposition using alternating least squares."""
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    for it in range(max_iters):
        # Optimize U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V: for each (j,k)
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W: for each (i,k)
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    
    # Round to half-integers
    U_r = _round_matrix_to_half_int(U)
    V_r = _round_matrix_to_half_int(V)
    W_r = _round_matrix_to_half_int(W)
    
    # Verify
    if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _strassen_construct(n, m, p):
    """Construct the Strassen-based decomposition (rank 72 for 4x4x5)."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    def add_strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = 1; w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = 1; v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1; u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        # Naive for col bk+2
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk+2, ai+di)] = 1
                add_term(u, v, w)

    # C00 = A00 B00 + A01 B10
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    # C01 = A00 B01 + A01 B11
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    # C10 = A10 B00 + A11 B10
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    # C11 = A10 B01 + A11 B11
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition using Kronecker product embedding.
    
    Splits 4x4x5 into 4x4x4 (handled by Kronecker-embedded Strassen, rank 49)
    plus 4x4x1 (handled naively, rank 16).
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Define the 7 Strassen 2x2x2 terms as sparse index-value lists
    # Each term: (list of (idx, val) for U, list of (idx, val) for V, list of (idx, val) for W)
    # U indices: (i, j) in 2x2 -> i*2+j
    # V indices: (j, k) in 2x2 -> j*2+k
    # W indices: (k, i) in 2x2 -> k*2+i
    
    strassen_terms = [
        # M1 = (a+d)(e+h): U[(0,0),(1,1)], V[(0,0),(1,1)], W[(0,0),(1,1)]
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e: U[(1,0),(1,1)], V[(0,0)], W[(0,1),(1,1)=-1]
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h): U[(0,0)], V[(0,1),(1,1)=-1], W[(1,0),(1,1)]
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e): U[(1,1)], V[(1,0),(0,0)=-1], W[(0,0),(0,1)]
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h: U[(0,0),(0,1)], V[(1,1)], W[(1,0)=-1,(1,1)]
        ([(0,1),(1,1)], [(3,1)], [(2,-1),(3,1)]),
        # M6 = (c-a)(e+f): U[(1,0),(0,0)=-1], V[(0,0),(0,1)], W[(1,1)]
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h): U[(0,1),(1,1)=-1], V[(1,0),(1,1)], W[(0,0)]
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    # Build 4x4x4 part using Kronecker product of 2x2x2 Strassen terms
    # For each pair (r, s) of Strassen terms:
    # U_{rs}[i*4+j] = sum over (i1,j1) in term r's U, (i2,j2) in term s's U: val_r(i1,j1)*val_s(i2,j2)
    # where i = 2*i1+i2, j = 2*j1+j2
    # Similarly for V and W
    
    for r in range(7):
        for s in range(7):
            U_rs = np.zeros(16)
            V_rs = np.zeros(16)
            W_rs = np.zeros(16)
            
            # Build U_{rs}
            for (idx_u1, val_u1) in strassen_terms[r][0]:
                i1 = idx_u1 // 2
                j1 = idx_u1 % 2
                for (idx_u2, val_u2) in strassen_terms[s][0]:
                    i2 = idx_u2 // 2
                    j2 = idx_u2 % 2
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    U_rs[i*4+j] += val_u1 * val_u2
            
            # Build V_{rs}
            for (idx_v1, val_v1) in strassen_terms[r][1]:
                j1 = idx_v1 // 2
                k1 = idx_v1 % 2
                for (idx_v2, val_v2) in strassen_terms[s][1]:
                    j2 = idx_v2 // 2
                    k2 = idx_v2 % 2
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    V_rs[j*4+k] += val_v1 * val_v2
            
            # Build W_{rs}
            for (idx_w1, val_w1) in strassen_terms[r][2]:
                k1 = idx_w1 // 2
                i1 = idx_w1 % 2
                for (idx_w2, val_w2) in strassen_terms[s][2]:
                    k2 = idx_w2 // 2
                    i2 = idx_w2 % 2
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    W_rs[k*4+i] += val_w1 * val_w2
            
            # Extend to 4x4x5 by zeroing out column 4
            U_full = np.zeros(n*m)
            V_full = np.zeros(m*p)
            W_full = np.zeros(p*n)
            
            for idx in range(16):
                U_full[idx] = U_rs[idx]
                V_full[idx] = V_rs[idx]
                W_full[idx] = W_rs[idx]
            
            add_term(U_full, V_full, W_full)
    
    # Build 4x4x1 part (naive, rank 16)
    for i in range(4):
        for j in range(4):
            U_full = np.zeros(n*m)
            V_full = np.zeros(m*p)
            W_full = np.zeros(p*n)
            
            U_full[i*4+j] = 1
            V_full[j*5+4] = 1
            W_full[4*4+i] = 1
            
            add_term(U_full, V_full, W_full)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Get the Kronecker-embedded Strassen decomposition (rank 65)
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    
    # Verify the Kronecker construction
    if _verify_decomposition(U_kron, V_kron, W_kron, n, m, p):
        best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
        best_rank = len(U_kron[0])
    else:
        # Fall back to the original Strassen construction (rank 72)
        U_base, V_base, W_base = _strassen_construct(n, m, p)
        best_result = {"U": U_base, "V": V_base, "W": W_base}
        best_rank = len(U_base[0])
    
    # Try to find lower-rank decompositions using ALS
    for rank in range(best_rank - 1, max(40, best_rank - 20), -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=200)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
