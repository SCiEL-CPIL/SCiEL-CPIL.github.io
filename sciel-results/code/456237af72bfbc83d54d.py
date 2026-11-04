import numpy as np
import time

def _build_2x2x2_strassen():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Shapes: (4, 7)
    """
    # Standard Strassen for 2x2x2
    # U indices: A00, A01, A10, A11 (flattened row-major: 0,1,2,3)
    # V indices: B00, B01, B10, B11 (flattened row-major: 0,1,2,3)
    # W indices: C00, C01, C10, C11 (flattened row-major: 0,1,2,3)
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 1], # A00
        [0, 1, 0, 0, 1, 0, -1], # A01
        [0, 0, 1, 0, 0, 1, 0],  # A10
        [1, 0, 0, 1, 0, 0, -1]  # A11
    ], dtype=np.float64)
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],  # B00
        [1, 0, 1, -1, 0, 1, 0], # B01
        [1, 0, -1, 1, 1, 0, 1], # B10
        [1, 0, 0, 0, 1, 0, 1]   # B11
    ], dtype=np.float64)
    
    W = np.array([
        [1, 0, 0, -1, 0, 1, -1], # C00
        [0, 0, 1, 1, 0, 0, 0],   # C01
        [0, 1, 1, 0, 0, 1, 0],   # C10
        [1, -1, 0, 0, 1, 0, 1]   # C11
    ], dtype=np.float64)
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build rank-49 decomposition for 4x4x4 using Strassen at block level.
    """
    U2, V2, W2 = _build_2x2x2_strassen()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for r in range(7):
        u_blocks = U2[:, r]
        v_blocks = V2[:, r]
        w_blocks = W2[:, r]
        
        for t in range(7):
            col = r * 7 + t
            u_elem = U2[:, t]
            v_elem = V2[:, t]
            w_elem = W2[:, t]
            
            for bi in range(4):
                coeff_u = u_blocks[bi]
                if coeff_u == 0: continue
                br = bi // 2
                bc = bi % 2
                for ei in range(4):
                    coeff_ue = u_elem[ei]
                    if coeff_ue == 0: continue
                    er = ei // 2
                    ec = ei % 2
                    gr = br * 2 + er
                    gc = bc * 2 + ec
                    idx = gr * 4 + gc
                    U[idx, col] += coeff_u * coeff_ue
            
            for bi in range(4):
                coeff_v = v_blocks[bi]
                if coeff_v == 0: continue
                br = bi // 2
                bc = bi % 2
                for ei in range(4):
                    coeff_ve = v_elem[ei]
                    if coeff_ve == 0: continue
                    er = ei // 2
                    ec = ei % 2
                    gr = br * 2 + er
                    gc = bc * 2 + ec
                    idx = gr * 4 + gc
                    V[idx, col] += coeff_v * coeff_ve
            
            for bi in range(4):
                coeff_w = w_blocks[bi]
                if coeff_w == 0: continue
                br = bi // 2
                bc = bi % 2
                for ei in range(4):
                    coeff_we = w_elem[ei]
                    if coeff_we == 0: continue
                    er = ei // 2
                    ec = ei % 2
                    gr = br * 2 + er
                    gc = bc * 2 + ec
                    idx = gr * 4 + gc
                    W[idx, col] += coeff_w * coeff_we
                    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify T = sum u_r v_r w_r
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U[i*m + j]
                v_vec = V[j*p + k]
                w_vec = W[k*n + i]
                val = np.dot(u_vec, v_vec * w_vec)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_4x4x8_rank64():
    """
    Build a rank-64 decomposition for 4x4x8 by exploiting shared A factors.
    
    Strategy: Split the 8-dimensional output into 4 groups of 2.
    For each group of 2 output columns, we compute C[:, 2g:2g+2] = A * B[:, 2g:2g+2].
    
    Instead of 4 independent 4x4x2 problems (each rank 16, total 64),
    we try to share the A-side factors across groups.
    
    Actually, let's use a different approach: treat 4x4x8 as two 4x4x4 blocks
    but with shared U factors. Since both blocks use the same A, we can
    potentially reduce rank by combining.
    
    For now, implement a clean rank-64 baseline and then try to improve.
    """
    n, m, p = 4, 4, 8
    
    # Schoolbook for 4x4x8 has rank 128
    # Two 4x4x4 Strassen blocks give rank 98
    # Let's try rank 64: 4 groups of 4x4x2
    
    # For 4x4x2: C[i,k] = sum_j A[i,j] * B[j,k] for k=0,1
    # This is equivalent to computing two 4x4x1 products.
    # Rank of 4x4x1 is 16 (one per output element).
    # So 4x4x2 has rank at most 32, but actually it's 16*2=32 if independent.
    # Wait, no. For 4x4x2, the tensor is:
    # T[i,j,k] = delta[i,k] for k in {0,1} where the "multiplication" is A[i,j]*B[j,k]
    # Actually for p=2, T[i,j,k] = 1 if we're looking at the bilinear form.
    # The rank of n x m x p matrix mult tensor for p=1 is n*m.
    # For p=2, it's at most 2*n*m but can be less.
    
    # Let me just use the straightforward approach: 4 copies of 4x4x2
    # Each 4x4x2 can be done with rank 16 (schoolbook: for each i,k, sum over j)
    # Total rank = 4 * 16 = 64
    
    U = np.zeros((16, 64))
    V = np.zeros((32, 64))
    W = np.zeros((32, 64))
    
    col = 0
    for g in range(4):  # 4 groups of 2 output columns
        for i in range(4):  # 4 rows
            for k in range(2):  # 2 columns in this group
                k_global = g * 2 + k
                for j in range(4):  # 4 cols of A / rows of B
                    # U index: i*m + j = i*4 + j
                    # V index: j*p + k_global = j*8 + k_global
                    # W index: k_global*n + i = k_global*4 + i
                    U[i*4 + j, col] = 1.0
                    V[j*8 + k_global, col] = 1.0
                    W[k_global*4 + i, col] = 1.0
                    col += 1
    
    return U, V, W

def construct():
    """
    Construct a low-rank decomposition for 4x4x8.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 4, 4, 8
    
    # Try rank-98 Strassen-based approach first
    U44, V44, W44 = _build_4x4x4_strassen()
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First 4x4x4 block (output cols 0-3)
    U[:, 0:49] = U44
    V[0:16, 0:49] = V44
    W[0:16, 0:49] = W44
    
    # Second 4x4x4 block (output cols 4-7)
    U[:, 49:98] = U44
    V[16:32, 49:98] = V44
    W[16:32, 49:98] = W44
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_sb = np.zeros((n*m, rank))
        V_sb = np.zeros((m*p, rank))
        W_sb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
