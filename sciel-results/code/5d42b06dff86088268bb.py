import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_DEADLINE = time.time() + 140

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # k=0
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    # k=1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    # k=2
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    # k=3
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    # k=4
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1; W[0,4] = -1
    # k=5
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    # k=6
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """Return U, V, W for 4x4x4 Strassen decomposition."""
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C blocks: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    
    # T0 = (A00+A11)(B00+B11)
    # A' = A00+A11, B' = B00+B11
    # A' indices: 0, 3
    # B' indices: 0, 3
    # C' = C00+C11
    # C' indices: 0, 3
    
    # T1 = (A10+A11)B00
    # A' = A10+A11, B' = B00
    # A' indices: 2, 3
    # B' indices: 0
    # C' = C10-C11
    # C' indices: 2, 3
    
    # T2 = A00(B01-B11)
    # A' = A00, B' = B01-B11
    # A' indices: 0
    # B' indices: 1, 3
    # C' = C01+C11
    # C' indices: 1, 3
    
    # T3 = A11(B10-B00)
    # A' = A11, B' = B10-B00
    # A' indices: 3
    # B' indices: 2, 0
    # C' = C00+C10
    # C' indices: 0, 2
    
    # T4 = (A00+A01)B11
    # A' = A00+A01, B' = B11
    # A' indices: 0, 1
    # B' indices: 3
    # C' = C01-C00
    # C' indices: 1, 0
    
    # T5 = (A10-A00)(B00+B01)
    # A' = A10-A00, B' = B00+B01
    # A' indices: 2, 0
    # B' indices: 0, 1
    # C' = C11
    # C' indices: 3
    
    # T6 = (A01-A11)(B10+B11)
    # A' = A01-A11, B' = B10+B11
    # A' indices: 1, 3
    # B' indices: 2, 3
    # C' = C00
    # C' indices: 0
    
    for k in range(7):
        col_offset = k * 7
        # Get U2, V2, W2 for this k
        Uk = U2[:, k]
        Vk = V2[:, k]
        Wk = W2[:, k]
        
        # Map A' indices to A block indices
        if k == 0:
            a_idx = [0, 3]
            b_idx = [0, 3]
            c_idx = [0, 3]
        elif k == 1:
            a_idx = [2, 3]
            b_idx = [0]
            c_idx = [2, 3]
        elif k == 2:
            a_idx = [0]
            b_idx = [1, 3]
            c_idx = [1, 3]
        elif k == 3:
            a_idx = [3]
            b_idx = [2, 0]
            c_idx = [0, 2]
        elif k == 4:
            a_idx = [0, 1]
            b_idx = [3]
            c_idx = [1, 0]
        elif k == 5:
            a_idx = [2, 0]
            b_idx = [0, 1]
            c_idx = [3]
        elif k == 6:
            a_idx = [1, 3]
            b_idx = [2, 3]
            c_idx = [0]
            
        # Fill U
        for i in range(len(a_idx)):
            a_block = a_idx[i]
            a_row = a_block * 4 # 2x2 block, so 4 rows
            for r in range(4):
                U[a_row + r, col_offset + i] = Uk[r]
                
        # Fill V
        for i in range(len(b_idx)):
            b_block = b_idx[i]
            b_row = b_block * 4
            for r in range(4):
                V[b_row + r, col_offset + i] = Vk[r]
                
        # Fill W
        for i in range(len(c_idx)):
            c_block = c_idx[i]
            c_row = c_block * 4
            for r in range(4):
                W[c_row + r, col_offset + i] = Wk[r]
                
    return U, V, W

def _schoolbook_4x4x5():
    """Return U, V, W for 4x4x5 schoolbook decomposition."""
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U1, V1, W1 = _strassen_4x4x4()
    U2, V2, W2 = _schoolbook_4x4x5()
    
    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_s = np.zeros((n * m, rank))
        V_s = np.zeros((m * p, rank))
        W_s = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

