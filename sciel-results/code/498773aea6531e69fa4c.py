import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _khatri_rao(A, B):
    """Compute Khatri-Rao product of two matrices A and B.
    A is (n, r), B is (m, r). Result is (n*m, r)."""
    n, r = A.shape
    m, _ = B.shape
    return np.einsum('ir,jr->ijr', A, B).reshape(n*m, r)

def _tensor_unfold(T, mode):
    """Unfold tensor T (shape n,m,p) along mode.
    mode 0: (n, m*p)
    mode 1: (m, n*p)
    mode 2: (p, n*m)
    """
    n, m, p = T.shape
    if mode == 0:
        return T.reshape(n, m*p)
    elif mode == 1:
        return np.transpose(T, (1, 0, 2)).reshape(m, n*p)
    elif mode == 2:
        return np.transpose(T, (2, 0, 1)).reshape(p, n*m)

def _solve_rank(n, m, p, rank, time_limit=10.0):
    """Find a decomposition of rank `rank` for M_{n,m} x M_{m,p}."""
    # Build the target tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0 # Wait, this is not the multiplication tensor.
                
    # Correct tensor construction:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is sum_{i,j,k} E_{i,j} \otimes F_{j,k} \otimes G_{i,k}
    # So T[i, j, k] is not 1.
    # The tensor components are T_{i,j,k} where the index of the second factor is j,
    # and the first is i, second is j, third is i,k?
    # No, the tensor is defined on the space of A, B, C.
    # A has indices i,j. B has indices j,k. C has indices i,k.
    # The bilinear map is (A,B) -> C.
    # The tensor is \sum_{i,j,k} e_i \otimes f_j \otimes g_{i,k}? No.
    # It is \sum_{i,j,k} E_{i,j} \otimes F_{j,k} \otimes G_{i,k}.
    # So the component at (i,j, j', k, i', k') is 1 if j=j', i=i', k=k'.
    # In the unfolded form:
    # T_{(3)} has rows indexed by (i,k) and columns by (i,j, j,k).
    # T_{(3)}[(i,k), (i',j', j'',k'')] = 1 if i=i', k=k'', j'=j''.
    
    # Let's construct T_unfolded directly.
    # Mode 2 (W) unfolding: rows (i,k), cols (i,j) x (j,k).
    # Size: (n*p) x (n*m * m*p)
    
    T2 = np.zeros((n*p, n*m * m*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Row index for (i,k)
                row = k*n + i
                # Col index for (i,j) and (j,k)
                col_A = i*m + j
                col_B = j*p + k
                col = col_A * (m*p) + col_B
                T2[row, col] = 1.0
                
    # Initialize U, V, W
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    start_time = time.time()
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Solve for W
        M_UV = _khatri_rao(U, V) # Shape (n*m * m*p, rank)
        # T2 is (n*p, n*m*m*p). We need to solve W * M_UV^T = T2?
        # T = sum w_r \otimes (u_r \otimes v_r)
        # T2 = W * (U \odot V)^T
        # So W = T2 * (U \odot V) / ((U \odot V)^T (U \odot V))
        try:
            W, _, _, _ = np.linalg.lstsq(M_UV, T2.T, rcond=None)
            W = W.T
        except:
            pass
            
        # Solve for U
        # T1 = U * (V \odot W)^T
        T1 = _tensor_unfold(T, 0) # Wait, T is not built.
        # We only have T2. We can build T1 and T2 from T2?
        # Or just build T1 and T2 separately.
        
        # Let's build T1 and T2 explicitly to be safe.
        T1 = np.zeros((n*m, m*p * p*n))
        T2 = np.zeros((p*n, n*m * m*p))
        T3 = np.zeros((m*p, n*m * p*n))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row1 = i*m + j
                    col1 = (j*p + k) * (p*n) + (k*n + i)
                    T1[row1, col1] = 1.0
                    
                    row2 = k*n + i
                    col2 = (i*m + j) * (m*p) + (j*p + k)
                    T2[row2, col2] = 1.0
                    
                    row3 = j*p + k
                    col3 = (i*m + j) * (p*n) + (k*n + i)
                    T3[row3, col3] = 1.0

        # Solve U
        M_VW = _khatri_rao(V, W)
        try:
            U, _, _, _ = np.linalg.lstsq(M_VW, T1.T, rcond=None)
            U = U.T
        except:
            pass
            
        # Solve V
        M_UW = _khatri_rao(U, W)
        try:
            V, _, _, _ = np.linalg.lstsq(M_UW, T3.T, rcond=None)
            V = V.T
        except:
            pass
            
    # Snap to integers
    U_int = np.rint(U).astype(int)
    V_int = np.rint(V).astype(int)
    W_int = np.rint(W).astype(int)
    
    # Verify
    # Reconstruct T2
    M_UV_int = _khatri_rao(U_int, V_int)
    T2_recon = W_int @ M_UV_int.T
    if np.allclose(T2, T2_recon, atol=1e-5):
        return U_int, V_int, W_int
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0
    
    # Try to find rank 13 for 2x3x3
    U_233, V_233, W_233 = None, None, None
    if time.time() < DEADLINE - 20:
        U_233, V_233, W_233 = _solve_rank(2, 3, 3, 13, time_limit=15.0)
        
    if U_233 is not None:
        # Build 2x3x10 using 3 blocks of 3 and 1 block of 1
        # Rank 13 + 13 + 13 + 6 = 45
        U_total = []
        V_total = []
        W_total = []
        
        # Block 1 (cols 0-2 of B, 0-2 of C)
        U_total.append(U_233)
        V_total.append(V_233)
        W_233_block1 = np.zeros((2*3, 13))
        W_233_block1[:6, :] = W_233
        W_total.append(W_233_block1)
        
        # Block 2 (cols 3-5 of B, 3-5 of C)
        U_total.append(U_233)
        # V needs to be shifted?
        # V index is j*p + k.
        # For block 1, k=0,1,2.
        # For block 2, k=3,4,5.
        # So V_233 has rows for k=0,1,2.
        # We need to map them to k=3,4,5.
        V_233_block2 = np.zeros((3*10, 13))
        # V_233 shape (9, 13). Rows j*3+k.
        for j in range(3):
            for k in range(3):
                orig_row = j*3 + k
                new_row = j*10 + (k+3)
                V_233_block2[new_row, :] = V_233[orig_row, :]
        V_total.append(V_233_block2)
        
        W_233_block2 = np.zeros((2*3, 13))
        # W index k*n+i.
        # Block 1: k=0,1,2. Rows 0-5.
        # Block 2: k=3,4,5. Rows 6-11.
        W_233_block2[6:12, :] = W_233
        W_total.append(W_233_block2)
        
        # Block 3 (cols 6-8 of B, 6-8 of C)
        U_total.append(U_233)
        V_233_block3 = np.zeros((3*10, 13))
        for j in range(3):
            for k in range(3):
                orig_row = j*3 + k
                new_row = j*10 + (k+6)
                V_233_block3[new_row, :] = V_233[orig_row, :]
        V_total.append(V_233_block3)
        
        W_233_block3 = np.zeros((2*3, 13))
        W_233_block3[12:18, :] = W_233
        W_total.append(W_233_block3)
        
        # Block 4 (col 9 of B, 9 of C)
        # Rank 6 for 2x3x1
        U_231 = np.zeros((6, 6))
        V_231 = np.zeros((30, 6))
        W_231 = np.zeros((20, 6))
        for j in range(3):
            for i in range(2):
                idx = j*2 + i
                U_231[i*3+j, idx] = 1.0
                V_231[j*10+9, idx] = 1.0
                W_231[9*2+i, idx] = 1.0
        U_total.append(U_231)
        V_total.append(V_231)
        W_total.append(W_231)
        
        U_final = np.hstack(U_total)
        V_final = np.hstack(V_total)
        W_final = np.hstack(W_total)
        
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        # Fallback to naive
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    c = i * m + j
                    d = j * p + k
                    e = k * n + i
                    idx = j * (n * p) + i * p + k
                    U[c, idx] = 1.0
                    V[d, idx] = 1.0
                    W[e, idx] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
