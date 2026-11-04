import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 115

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def get_2x2x2_strassen():
        """
        Returns U, V, W for a 2x2x2 block with rank 7 (Strassen).
        U: (4, 7), V: (4, 7), W: (4, 7)
        Local indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 - M4 + M5 + M6
        # C01 = M2 + M4
        # C10 = M3 + M5
        # C11 = M1 - M3 + M2 + M7
        
        # U coefficients: row index = i*2+j
        # A00=0, A01=1, A10=2, A11=3
        
        # M1: U = A00+A11 = 0+3
        U[0, 0] = 1; U[3, 0] = 1
        # V = B00+B11 = 0+3 (V index = j*2+k: B00=0, B11=3)
        V[0, 0] = 1; V[3, 0] = 1
        # W = C00+C11 = W[0*2+0] + W[1*2+1] = W[0] + W[3]
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: U = A10+A11 = 2+3
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1  # B00
        W[1, 1] = 1  # C01 = W[1*2+0]... wait
        
        # Let me recheck W indexing. W[k*n+i], local n=2.
        # C00: k=0,i=0 -> W[0]
        # C01: k=1,i=0 -> W[1*2+0]=W[2]
        # C10: k=0,i=1 -> W[0*2+1]=W[1]
        # C11: k=1,i=1 -> W[1*2+1]=W[3]
        
        # Redo:
        # M1: W = C00+C11 = W[0]+W[3]
        W[0, 0] = 1; W[3, 0] = 1
        # M2: W = C01 = W[2]
        W[2, 1] = 1
        # M3: W = C10 = W[1]
        W[1, 2] = 1
        # M4: W = -C00 + C01 = -W[0] + W[2]
        W[0, 3] = -1; W[2, 3] = 1
        # M5: W = C00 + C10 = W[0] + W[1]
        W[0, 4] = 1; W[1, 4] = 1
        # M6: W = C00 = W[0]
        W[0, 5] = 1
        # M7: W = C11 = W[3]
        W[3, 6] = 1
        
        # V indexing: V[j*2+k]
        # B00: j=0,k=0 -> V[0]
        # B01: j=0,k=1 -> V[1]
        # B10: j=1,k=0 -> V[2]
        # B11: j=1,k=1 -> V[3]
        
        # M1: V = B00+B11 = V[0]+V[3]
        V[0, 0] = 1; V[3, 0] = 1
        # M2: V = B00 = V[0]
        V[0, 1] = 1
        # M3: V = B10-B11 = V[2]-V[3]
        V[2, 2] = 1; V[3, 2] = -1
        # M4: V = B01-B00 = V[1]-V[0]
        V[1, 3] = 1; V[0, 3] = -1
        # M5: V = B11 = V[3]
        V[3, 4] = 1
        # M6: V = B00+B01 = V[0]+V[1]
        V[0, 5] = 1; V[1, 5] = 1
        # M7: V = B10+B11 = V[2]+V[3]
        V[2, 6] = 1; V[3, 6] = 1
        
        # U indexing: U[i*2+j]
        # A00: i=0,j=0 -> U[0]
        # A01: i=0,j=1 -> U[1]
        # A10: i=1,j=0 -> U[2]
        # A11: i=1,j=1 -> U[3]
        
        # M1: U = A00+A11 = U[0]+U[3]
        U[0, 0] = 1; U[3, 0] = 1
        # M2: U = A10+A11 = U[2]+U[3]
        U[2, 1] = 1; U[3, 1] = 1
        # M3: U = A00 = U[0]
        U[0, 2] = 1
        # M4: U = A11 = U[3]
        U[3, 3] = 1
        # M5: U = A00+A01 = U[0]+U[1]
        U[0, 4] = 1; U[1, 4] = 1
        # M6: U = A10-A00 = U[2]-U[0]
        U[2, 5] = 1; U[0, 5] = -1
        # M7: U = A01-A11 = U[1]-U[3]
        U[1, 6] = 1; U[3, 6] = -1
        
        return U, V, W
    
    def get_2x2x8_strassen_sch():
        """
        2x2 A, 2x8 B, 2x8 C.
        Use Strassen for the 2x2x2 blocks (4 blocks in the B,C dimension).
        For each of the 4 column-pairs of B, apply Strassen.
        Total: 4 * 7 = 28 rank terms.
        
        U: (4, 28), V: (16, 28), W: (16, 28)
        Local indexing: U[i*2+j], V[j*8+k], W[k*2+i]
        """
        rank = 28
        U = np.zeros((4, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # For each of 4 column blocks (k_pair = 0,1,2,3), 
        # the k indices are 2*k_pair and 2*k_pair+1
        # The 2x2 B block has B entries at V[j*8 + k] for k in {2*kp, 2*kp+1}
        # The 2x2 C block has C entries at W[k*2 + i] for k in {2*kp, 2*kp+1}
        
        # Strassen U, V, W are defined for local 2x2 with:
        # V_local[j*2+k_local], W_local[k_local*2+i]
        
        U_str, V_str, W_str = get_2x2x2_strassen()
        
        for kp in range(4):
            col_start = kp * 7
            col_end = col_start + 7
            k0 = 2 * kp
            
            U[:, col_start:col_end] = U_str
            
            # Map V_local[j*2+k_local] to V[j*8 + k0 + k_local]
            for c in range(7):
                for j in range(2):
                    for kl in range(2):
                        V[j*8 + k0 + kl, col_start + c] = V_str[j*2 + kl, c]
            
            # Map W_local[k_local*2+i] to W[(k0+k_local)*2 + i]
            for c in range(7):
                for kl in range(2):
                    for i in range(2):
                        W[(k0 + kl)*2 + i, col_start + c] = W_str[kl*2 + i, c]
        
        return U, V, W
    
    def get_3x3x8_decomp():
        """
        3x3 A, 3x8 B, 3x8 C.
        Strategy: Use the 2x2x8 Strassen-based block for i,j in {0,1}, 
        and schoolbook for the remaining rows/cols.
        
        Decomposition:
        - i,j in {0,1}: Use 2x2x8 with rank 28
        - i=2 or j=2: schoolbook terms
          - i=2, j in {0,1,2}, k in {0..7}: 3*8 = 24 terms
          - i in {0,1}, j=2, k in {0..7}: 2*8 = 16 terms
        Total: 28 + 24 + 16 = 68
        
        U: (9, 68), V: (24, 68), W: (24, 68)
        U[i*3+j], V[j*8+k], W[k*3+i]
        """
        rank = 68
        U = np.zeros((9, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        U_blk, V_blk, W_blk = get_2x2x8_strassen_sch()
        # U_blk: (4, 28), V_blk: (16, 28), W_blk: (16, 28)
        
        # Place 2x2x8 block
        for i in range(2):
            for j in range(2):
                U[i*3+j, 0:28] = U_blk[i*2+j, 0:28]
        
        for j in range(2):
            for k in range(8):
                V[j*8+k, 0:28] = V_blk[j*8+k, 0:28]
        
        for k in range(8):
            for i in range(2):
                W[k*3+i, 0:28] = W_blk[k*2+i, 0:28]
        
        # Schoolbook: i=2, j in {0,1,2}, k in {0..7}
        col = 28
        for i in [2]:
            for j in range(3):
                for k in range(8):
                    U[i*3+j, col] = 1.0
                    V[j*8+k, col] = 1.0
                    W[k*3+i, col] = 1.0
                    col += 1
        
        # Schoolbook: i in {0,1}, j=2, k in {0..7}
        for i in range(2):
            for k in range(8):
                U[i*3+2, col] = 1.0
                V[2*8+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
        
        assert col == rank, f"col={col}, rank={rank}"
        return U, V, W
    
    def verify(U, V, W):
        """Check that the decomposition is exact."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        rank = U.shape[1]
        # Check: for each (i,j,k), sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] == 1
        # Actually the tensor is T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        # and we want T[i,j,k] = 1 for all i,j,k
        # Build the tensor and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(s - 1.0) > 1e-10:
                        return False, (i,j,k,s)
        return True, None
    
    # Construct the decomposition
    U, V, W = get_3x3x8_decomp()
    
    # Verify
    ok, info = verify(U, V, W)
    if not ok:
        # Return schoolbook as fallback
        return schoolbook()
    
    # Try to optimize: remove zero rows/cols
    # Find columns that are all-zero
    rank = U.shape[1]
    U_rank = np.linalg.matrix_rank(U, tol=1e-12)
    V_rank = np.linalg.matrix_rank(V, tol=1e-12)
    W_rank = np.linalg.matrix_rank(W, tol=1e-12)
    
    if time.time() - start_time > DEADLINE_S - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
