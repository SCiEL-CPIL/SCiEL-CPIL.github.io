import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=5, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        
        for c in range(rank):
            # factor_A[i] = sum_j A[i,j] * U[i*m+j, c]
            # U is (n*m, rank). Row i*m+j corresponds to A[i,j]
            # A is (n, m).
            # We can reshape A to (n*m, 1) and U[:, c] is (n*m, 1)
            # factor_A[i] is not a simple dot product of A[i,:] and U[i*m:(i+1)*m, c]?
            # Yes, factor_A[i] = dot(A[i, :], U[i*m:(i+1)*m, c])
            
            # Vectorized computation for all i
            # A_flat = A.flatten() -> shape (n*m,)
            # U_col = U_np[:, c] -> shape (n*m,)
            # product = A_flat * U_col -> shape (n*m,)
            # We need to sum over j for each i.
            # Reshape product to (n, m) and sum axis=1.
            
            A_flat = A.flatten()
            prod_A = A_flat * U_np[:, c]
            factor_A = prod_A.reshape(n, m).sum(axis=1)
            
            # factor_B[k] = sum_j V[j*p+k, c] * B[j,k]
            # B is (m, p).
            # V is (m*p, rank).
            # For fixed k, V indices are j*p + k for j in 0..m-1.
            # This is not contiguous in V if p > 1?
            # V row index is j*p + k.
            # If we flatten B to (m*p, 1), row j*p+k is B[j,k].
            # So factor_B[k] = dot(B_flat[V_indices], V_np[V_indices, c])?
            # No, factor_B[k] = sum_{j=0}^{m-1} B[j,k] * V[j*p+k, c]
            
            B_flat = B.flatten() # shape (m*p), order C: B[0,0], B[0,1]... B[0,p-1], B[1,0]...
            # V index for B[j,k] is j*p + k.
            # So B_flat[j*p+k] is B[j,k].
            # V_col = V_np[:, c]
            # prod_B = B_flat * V_col
            # factor_B[k] = sum_{j} prod_B[j*p+k]
            # Reshape prod_B to (m, p) and sum axis=0.
            
            prod_B = B_flat * V_np[:, c]
            factor_B = prod_B.reshape(m, p).sum(axis=0)
            
            # C[i,k] += factor_A[i] * factor_B[k] * W[k*n+i, c]
            # W is (p*n, rank). Index k*n + i.
            # factor_A is (n,), factor_B is (p,).
            # outer = np.outer(factor_A, factor_B) -> shape (n, p)
            # We need to multiply element (i,k) by W[k*n+i, c].
            # W_flat = W_np[:, c] -> shape (p*n)
            # Reshape W_flat to (p, n) -> W_mat[k, i] = W[k*n+i]
            # C_computed += outer * W_mat.T ?
            # outer[i, k] * W_mat[k, i]
            # W_mat = W_flat.reshape(p, n)
            # term = outer * W_mat.T  # W_mat.T is (n, p), element (i,k) is W_mat[k,i]
            
            W_mat = W_np[:, c].reshape(p, n)
            C_computed += np.outer(factor_A, factor_B) * W_mat.T
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try Strassen rank-56 decomposition
    try:
        U = np.zeros((n*m, 56))
        V = np.zeros((m*p, 56))
        W = np.zeros((p*n, 56))
        
        # Strassen coefficients for 2x2x2
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U coefficients (A part)
        # M1: A11, A22 -> [1, 0, 0, 1] (flat A: A11, A12, A21, A22)
        # M2: A21, A22 -> [0, 0, 1, 1]
        # M3: A11 -> [1, 0, 0, 0]
        # M4: A22 -> [0, 0, 0, 1]
        # M5: A11, A12 -> [1, 1, 0, 0]
        # M6: A21, A11 -> [-1, 0, 1, 0] (A21-A11)
        # M7: A12, A22 -> [0, 1, -1, 0] (A12-A22)
        
        u_strassen = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, -1, 0]
        ]
        
        # V coefficients (B part)
        # M1: B11, B22 -> [1, 0, 0, 1] (flat B: B11, B12, B21, B22)
        # M2: B11 -> [1, 0, 0, 0]
        # M3: B12, B22 -> [0, 1, 0, -1]
        # M4: B21, B11 -> [-1, 0, 1, 0]
        # M5: B22 -> [0, 0, 0, 1]
        # M6: B11, B12 -> [1, 1, 0, 0]
        # M7: B21, B22 -> [0, 0, 1, 1]
        
        v_strassen = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        # W coefficients (C part)
        # C11 = M1 - M4 + M6 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M5 + M7
        
        # Flat C: C11, C12, C21, C22
        # M1: C11, C22 -> [1, 0, 0, 1]
        # M2: C21 -> [0, 0, 1, 0]
        # M3: C12, C22 -> [0, 1, 0, -1]
        # M4: C11, C21 -> [1, 0, 1, 0]
        # M5: C12, C22 -> [0, 1, 0, 1]
        # M6: C11 -> [1, 0, 0, 0]
        # M7: C11, C22 -> [-1, 0, 0, 1]
        
        w_strassen = [
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, -1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 0],
            [-1, 0, 0, 1]
        ]
        
        col_offset = 0
        # We have 4 independent 2x2x2 blocks for A_L B_T
        # and 4 for A_R B_B
        # Total 8 blocks.
        
        for block_idx in range(8):
            is_left = block_idx < 4
            k_pair = block_idx % 4
            
            for s in range(7):
                if is_left:
                    # A indices: i in 0..1, j in 0..1 (cols 0..1 of A)
                    # A row i, col j. Index i*4 + j.
                    for i in range(2):
                        for j in range(2):
                            U[i*4 + j, col_offset] = u_strassen[s][i*2 + j]
                    # B indices: j in 0..1 (rows 0..1 of B), k in k_pair*2 .. k_pair*2+1
                    # B row j, col k. Index j*8 + k.
                    for j in range(2):
                        for k in range(2):
                            V[j*8 + k_pair*2 + k, col_offset] = v_strassen[s][j*2 + k]
                    # C indices: i in 0..1, k in k_pair*2 .. k_pair*2+1
                    # C row i, col k. Index k*2 + i.
                    for i in range(2):
                        for k in range(2):
                            W[(k_pair*2 + k)*2 + i, col_offset] = w_strassen[s][i*2 + k]
                else:
                    # A indices: i in 0..1, j in 2..3 (cols 2..3 of A)
                    # A row i, col j. Index i*4 + j.
                    for i in range(2):
                        for j in range(2):
                            U[i*4 + (2 + j), col_offset] = u_strassen[s][i*2 + j]
                    # B indices: j in 2..3 (rows 2..3 of B), k in k_pair*2 .. k_pair*2+1
                    # B row j, col k. Index j*8 + k.
                    for j in range(2):
                        for k in range(2):
                            V[(2 + j)*8 + k_pair*2 + k, col_offset] = v_strassen[s][j*2 + k]
                    # C indices: i in 0..1, k in k_pair*2 .. k_pair*2+1
                    # C row i, col k. Index k*2 + i.
                    for i in range(2):
                        for k in range(2):
                            W[(k_pair*2 + k)*2 + i, col_offset] = w_strassen[s][i*2 + k]
                
                col_offset += 1
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, trials=5, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook
            return {"U": U_school, "V": V_school, "W": W_school}
            
    except Exception:
        return {"U": U_school, "V": V_school, "W": W_school}
