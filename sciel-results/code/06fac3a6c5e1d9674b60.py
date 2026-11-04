import time
import numpy as np

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2: 7 terms per block
    # Block at (i0, j0, k0) covers:
    #   A rows i0..i0+1, cols j0..j0+1
    #   B rows j0..j0+1, cols k0..k0+1
    #   C rows i0..i0+1, cols k0..k0+1
    #
    # Within a block, local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global: I=i0+i, J=j0+j, K=k0+k
    #
    # U index: I*m + J
    # V index: J*p + K
    # W index: K*n + I
    
    def add_strassen_block(i0, j0, k0, offset):
        # Term 1: M1 = (A00+A11)(B00+B11)
        col = offset
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        
        # Term 2: M2 = (A10+A11)(B00)
        col = offset + 1
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        
        # Term 3: M3 = (A00)(B10-B11)
        col = offset + 2
        U[i0*m + j0, col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + i0, col] += 1
        
        # Term 4: M4 = (A11)(B01-B00)
        col = offset + 3
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] -= 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        
        # Term 5: M5 = (A00+A01)(B11)
        col = offset + 4
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        col = offset + 5
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        col = offset + 6
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
    
    # 6 blocks: j0 in {0, 2}, k0 in {0, 2, 4}
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]
    
    for idx, (i0, j0, k0) in enumerate(blocks):
        add_strassen_block(i0, j0, k0, idx * 7)
    
    # Verify the decomposition
    def is_valid(U_mat, V_mat, W_mat, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U_mat[u_idx, :], V_mat[v_idx, :] * W_mat[w_idx, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not is_valid(U, V, W):
        # Fallback: schoolbook rank 48
        rank = 48
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m + j, r] = 1
                    V[j*p + k, r] = 1
                    W[k*n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
