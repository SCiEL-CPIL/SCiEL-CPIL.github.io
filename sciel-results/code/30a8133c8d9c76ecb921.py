"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b, col] = 1
                V[b*p + c, col] = 1
                W[c*n + a, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _strassen_4x4():
    """
    4x4x4 via recursive Strassen: rank 49.
    View as 2x2 blocks of 2x2. Apply Strassen at block level (7 terms),
    each term is a 2x2x2 multiplication (rank 7). Total 49.
    """
    n, m, p = 4, 4, 4
    U2, V2, W2 = _strassen_2x2()
    
    # Strassen coefficients
    # X[k][a][b]: coefficient of A_block[a][b] in the k-th Strassen X-factor
    # Y[k][b][c]: coefficient of B_block[b][c] in the k-th Strassen Y-factor
    X_coeff = [
        [[1,0],[0,1]],   # M1: (A11+A22)
        [[0,0],[1,1]],   # M2: (A21+A22)
        [[1,0],[0,0]],   # M3: A11
        [[0,0],[0,1]],   # M4: A22
        [[1,1],[0,0]],   # M5: (A11+A12)
        [[-1,0],[1,0]],  # M6: (A21-A11)
        [[0,1],[0,-1]],  # M7: (A12-A22)
    ]
    Y_coeff = [
        [[1,0],[0,1]],   # M1: (B11+B22)
        [[1,0],[0,0]],   # M2: B11
        [[0,1],[0,-1]],  # M3: (B12-B22)
        [[-1,0],[1,0]],  # M4: (B21-B11)
        [[0,0],[0,1]],   # M5: B22
        [[1,1],[0,0]],   # M6: (B11+B12)
        [[0,0],[1,1]],   # M7: (B21+B22)
    ]
    
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    
    col = 0
    for k in range(7):
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    xc = X_coeff[k][a][b]
                    yc = Y_coeff[k][b][c]
                    if xc == 0 or yc == 0:
                        continue
                    for t in range(7):
                        # U factor
                        for i in range(2):
                            for j in range(2):
                                val = U2[i*2+j, t]
                                if val != 0:
                                    U[(2*a+i)*4 + (2*b+j), col] += xc * val
                        # V factor
                        for j in range(2):
                            for k2 in range(2):
                                val = V2[j*2+k2, t]
                                if val != 0:
                                    V[(2*b+j)*4 + (2*c+k2), col] += yc * val
                        # W factor
                        for k2 in range(2):
                            for i in range(2):
                                val = W2[k2*2+i, t]
                                if val != 0:
                                    W[(2*c+k2)*4 + (2*a+i), col] += val
                        col += 1
    
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    return U, V, W

def _embed_decomposition(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                         n_total, m_total, p_total,
                         offset_a, offset_b, offset_c):
    rank = U_sub.shape[1]
    U_global = np.zeros((n_total * m_total, rank), dtype=np.int64)
    V_global = np.zeros((m_total * p_total, rank), dtype=np.int64)
    W_global = np.zeros((p_total * n_total, rank), dtype=np.int64)
    
    for a_sub in range(n_sub):
        for b_sub in range(m_sub):
            idx_u_sub = a_sub * m_sub + b_sub
            idx_u_global = (offset_a + a_sub) * m_total + (offset_b + b_sub)
            U_global[idx_u_global] = U_sub[idx_u_sub]
            
    for b_sub in range(m_sub):
        for c_sub in range(p_sub):
            idx_v_sub = b_sub * p_sub + c_sub
            idx_v_global = (offset_b + b_sub) * p_total + (offset_c + c_sub)
            V_global[idx_v_global] = V_sub[idx_v_sub]
            
    for c_sub in range(p_sub):
        for a_sub in range(n_sub):
            idx_w_sub = c_sub * n_sub + a_sub
            idx_w_global = (offset_c + c_sub) * n_total + (offset_a + a_sub)
            W_global[idx_w_global] = W_sub[idx_w_sub]
            
    return U_global, V_global, W_global

def _compose_decompositions(decomps, n, m, p):
    if not decomps:
        return np.zeros((n*m, 0)), np.zeros((m*p, 0)), np.zeros((p*n, 0))
    U_total = np.hstack([d[0] for d in decomps])
    V_total = np.hstack([d[1] for d in decomps])
    W_total = np.hstack([d[2] for d in decomps])
    return U_total, V_total, W_total

def _build_5x5_decomposition():
    n, m, p = 5, 5, 5
    decomps = []
    
    # C[0,0] = A[0,0]*B[0,0]: 1x1x1, rank 1
    U00, V00, W00 = _schoolbook(1, 1, 1)
    U00, V00, W00 = _embed_decomposition(U00, V00, W00, 1, 1, 1, n, m, p, 0, 0, 0)
    decomps.append((U00, V00, W00))
    
    # C[0,1:5] = A[0,0]*B[0,1:5] + A[0,1:5]*B[1:5,1:5]
    # Part 1: 1x1 * 1x4 -> rank 4
    U01, V01, W01 = _schoolbook(1, 1, 4)
    U01, V01, W01 = _embed_decomposition(U01, V01, W01, 1, 1, 4, n, m, p, 0, 0, 1)
    decomps.append((U01, V01, W01))
    
    # Part 2: 1x4 * 4x4 -> rank 16
    U01m, V01m, W01m = _schoolbook(1, 4, 4)
    U01m, V01m, W01m = _embed_decomposition(U01m, V01m, W01m, 1, 4, 4, n, m, p, 0, 1, 1)
    decomps.append((U01m, V01m, W01m))
    
    # C[1:5,0] = A[1:5,0]*B[0,0] + A[1:5,1:5]*B[1:5,0]
    # Part 1: 4x1 * 1x1 -> rank 4
    U10, V10, W10 = _schoolbook(4, 1, 1)
    U10, V10, W10 = _embed_decomposition(U10, V10, W10, 4, 1, 1, n, m, p, 1, 0, 0)
    decomps.append((U10, V10, W10))
    
    # Part 2: 4x4 * 4x1 -> rank 16
    U10m, V10m, W10m = _schoolbook(4, 4, 1)
    U10m, V10m, W10m = _embed_decomposition(U10m, V10m, W10m, 4, 4, 1, n, m, p, 1, 1, 0)
    decomps.append((U10m, V10m, W10m))
    
    # C[1:5,1:5] = A[1:5,0]*B[0,1:5] + A[1:5,1:5]*B[1:5,1:5]
    # Part 1: 4x1 * 1x4 -> rank 4
    U11c, V11c, W11c = _schoolbook(4, 1, 4)
    U11c, V11c, W11c = _embed_decomposition(U11c, V11c, W11c, 4, 1, 4, n, m, p, 1, 0, 1)
    decomps.append((U11c, V11c, W11c))
    
    # Part 2: 4x4 * 4x4 -> rank 49 (Strassen)
    U44, V44, W44 = _strassen_4x4()
    U11, V11, W11 = _embed_decomposition(U44, V44, W44, 4, 4, 4, n, m, p, 1, 1, 1)
    decomps.append((U11, V11, W11))
    
    U_total, V_total, W_total = _compose_decompositions(decomps, n, m, p)
    return U_total.tolist(), V_total.tolist(), W_total.tolist()

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    U, V, W = _build_5x5_decomposition()
    
    if _verify(U, V, W, n, m, p, len(U[0])):
        return {"U": U, "V": V, "W": W}
    else:
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
