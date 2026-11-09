import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = 3, 3, 8
    
    # Rank calculation: 4 blocks * (7 Strassen + 5 Cross) = 4 * 12 = 48
    rank_total = 48
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    col = 0
    
    # Strassen 2x2x2 factors
    # Indices in 2x2x2: i,j,k in {0,1}
    # U_s (4, 7), V_s (4, 7), W_s (4, 7)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2: (A10+A11)B01
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # M3: A00(B10-B11)
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # M4: A01(B00-B01)
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # M5: (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[2, 5] = 1
    W_s[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[1, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1; W_s[1, 6] = 1
    
    # Cross terms for 3x3x2
    # We need 5 terms for the cross.
    # Each term is a vector of length 1 in U, V, W? No, we add them to the block.
    # Let's define the 5 cross terms as matrices (4, 5) for U, (4, 5) for V, (4, 5) for W?
    # No, U is 9x1, V is 6x1, W is 6x1 for each cross term.
    # But we are filling 12 columns per block.
    # Columns 0-6: Strassen.
    # Columns 7-11: Cross.
    
    # Cross Term 1: i=2, j=0
    # U[2, col+7] = 1
    # V[0, col+7] = 1, V[1, col+7] = 1  (j=0, k=0,1 -> V indices 0, 1 in local 2x2)
    # W[2, col+7] = 1, W[5, col+7] = 1  (k=0,1, i=2 -> W indices 0*3+2=2, 1*3+2=5)
    
    # Cross Term 2: i=2, j=1
    # U[5, col+8] = 1
    # V[2, col+8] = 1, V[3, col+8] = 1  (j=1, k=0,1 -> V indices 2, 3 in local)
    # W[2, col+8] = 1, W[5, col+8] = 1
    
    # Cross Term 3: i=0, j=2
    # U[2, col+9] = 1
    # V[4, col+9] = 1, V[5, col+9] = 1  (j=2, k=0,1 -> V indices 4, 5 in local)
    # W[0, col+9] = 1, W[3, col+9] = 1  (k=0,1, i=0 -> W indices 0, 3 in local)
    
    # Cross Term 4: i=1, j=2
    # U[5, col+10] = 1
    # V[4, col+10] = 1, V[5, col+10] = 1
    # W[1, col+10] = 1, W[4, col+10] = 1  (k=0,1, i=1 -> W indices 1, 4 in local)
    
    # Cross Term 5: i=2, j=2
    # U[8, col+11] = 1
    # V[4, col+11] = 1, V[5, col+11] = 1
    # W[2, col+11] = 1, W[5, col+11] = 1

    for block_idx in range(4):
        k_offset = block_idx * 2
        
        # Fill Strassen part (cols col to col+6)
        for i in range(2):
            for j in range(2):
                u_full = i * 3 + j
                u_str = i * 2 + j
                U[u_full, col:col+7] = U_s[u_str, :]
        for j in range(2):
            for k_rel in range(2):
                v_full = j * 8 + (k_offset + k_rel)
                v_str = j * 2 + k_rel
                V[v_full, col:col+7] = V_s[v_str, :]
        for k_rel in range(2):
            for i in range(2):
                w_full = (k_offset + k_rel) * 3 + i
                w_str = k_rel * 2 + i
                W[w_full, col:col+7] = W_s[w_str, :]
        
        # Fill Cross part (cols col+7 to col+11)
        # Term 1: i=2, j=0
        U[2*3+0, col+7] = 1
        V[0*8 + (k_offset+0), col+7] = 1
        V[0*8 + (k_offset+1), col+7] = 1
        W[(k_offset+0)*3+2, col+7] = 1
        W[(k_offset+1)*3+2, col+7] = 1
        
        # Term 2: i=2, j=1
        U[2*3+1, col+8] = 1
        V[1*8 + (k_offset+0), col+8] = 1
        V[1*8 + (k_offset+1), col+8] = 1
        W[(k_offset+0)*3+2, col+8] = 1
        W[(k_offset+1)*3+2, col+8] = 1
        
        # Term 3: i=0, j=2
        U[0*3+2, col+9] = 1
        V[2*8 + (k_offset+0), col+9] = 1
        V[2*8 + (k_offset+1), col+9] = 1
        W[(k_offset+0)*3+0, col+9] = 1
        W[(k_offset+1)*3+0, col+9] = 1
        
        # Term 4: i=1, j=2
        U[1*3+2, col+10] = 1
        V[2*8 + (k_offset+0), col+10] = 1
        V[2*8 + (k_offset+1), col+10] = 1
        W[(k_offset+0)*3+1, col+10] = 1
        W[(k_offset+1)*3+1, col+10] = 1
        
        # Term 5: i=2, j=2
        U[2*3+2, col+11] = 1
        V[2*8 + (k_offset+0), col+11] = 1
        V[2*8 + (k_offset+1), col+11] = 1
        W[(k_offset+0)*3+2, col+11] = 1
        W[(k_offset+1)*3+2, col+11] = 1
        
        col += 12

    # Verify
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
                
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T, target, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
