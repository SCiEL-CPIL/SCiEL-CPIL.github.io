"""Rewritten program: use known low-rank decompositions for sub-blocks."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def check_deadline():
        return time.time() > deadline

    # Schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
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
        return U, V, W, rank

    # Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2x2():
        # 7 multiplications for 2x2x2
        # U has 4 entries (2x2), V has 4 entries (2x2), W has 4 entries (2x2)
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[1, 0] = 1  # B12  -- wait, B is 2x2 so indices are B11,B12,B21,B22
        # Let me re-index: A is 2x2 with entries A[0,0],A[0,1],A[1,0],A[1,1]
        # B is 2x2 with entries B[0,0],B[0,1],B[1,0],B[1,1]
        # C is 2x2 with entries C[0,0],C[0,1],C[1,0],C[1,1]
        # U[i*m+j] where m=2, so U[0]=A11, U[1]=A12, U[2]=A21, U[3]=A22
        # V[j*p+k] where p=2, so V[0]=B11, V[1]=B12, V[2]=B21, V[3]=B22
        # W[k*n+i] where n=2, so W[0]=C11, W[1]=C12, W[2]=C21, W[3]=C22
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # C11 += M1, C22 += M1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C21 += M2
        W[3, 1] = -1  # C22 -= M2
        
        # M3 = A11(B21-B22)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C12 += M3
        W[3, 2] = 1  # C22 += M3
        
        # M4 = A22(B12-B11)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # C11 += M4
        W[2, 3] = 1  # C21 += M4
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1  # C11 -= M5
        W[1, 4] = 1   # C12 += M5
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1   # C22 += M6
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1   # C11 += M7
        
        return U, V, W, 7

    # Verify a decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Build full 3x5x5 using block decomposition: split 5 = 2 + 3
    # Blocks: (3x2x2), (3x3x2), (3x2x3), (3x3x3)
    # For each block, we can use schoolbook or Strassen-based methods
    
    # For 3x3x3, we can use a known rank-23 decomposition (Strassen-like)
    # But let's try to use the block approach with Strassen where possible
    
    # Strategy: 
    # 3x2x2: schoolbook rank 12
    # 3x3x2: schoolbook rank 18  
    # 3x2x3: schoolbook rank 18
    # 3x3x3: try rank 23 (Strassen for 3x3x3)
    # Total: 12 + 18 + 18 + 23 = 71
    
    # For 3x3x3 rank 23, we need a specific decomposition
    # Let me use a known approach: split 3 = 2 + 1
    # 3x3x3 = (2+1)x(2+1)x(2+1) gives 8 sub-blocks
    # But that's complex. Let me try a different approach.
    
    # Actually, let me try to find a 3x3x3 rank 23 decomposition using ALS
    def find_3x3x3_rank23():
        """Try to find a rank-23 decomposition for 3x3x3."""
        n_sub, m_sub, p_sub = 3, 3, 3
        target_rank = 23
        max_iters = 200
        n_restarts = 10
        
        for attempt in range(n_restarts):
            if check_deadline():
                break
            
            np.random.seed(100 + attempt * 7)
            U = np.random.randn(n_sub * m_sub, target_rank) * 0.3
            V = np.random.randn(m_sub * p_sub, target_rank) * 0.3
            W = np.random.randn(p_sub * n_sub, target_rank) * 0.3
            
            for it in range(max_iters):
                if check_deadline():
                    break
                
                # Update U
                A = np.zeros((n_sub * m_sub, target_rank))
                for i in range(n_sub):
                    for j in range(m_sub):
                        for c in range(target_rank):
                            s = 0.0
                            for k in range(p_sub):
                                s += V[j * p_sub + k, c] * W[k * n_sub + i, c]
                            A[i * m_sub + j, c] = s
                for r in range(n_sub * m_sub):
                    if np.linalg.norm(A[r, :]) > 1e-10:
                        U[r, :] = np.linalg.lstsq(A[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                
                # Update V
                B = np.zeros((m_sub * p_sub, target_rank))
                for j in range(m_sub):
                    for k in range(p_sub):
                        for c in range(target_rank):
                            s = 0.0
                            for i in range(n_sub):
                                s += U[i * m_sub + j, c] * W[k * n_sub + i, c]
                            B[j * p_sub + k, c] = s
                for r in range(m_sub * p_sub):
                    if np.linalg.norm(B[r, :]) > 1e-10:
                        V[r, :] = np.linalg.lstsq(B[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                
                # Update W
                C = np.zeros((p_sub * n_sub, target_rank))
                for k in range(p_sub):
                    for i in range(n_sub):
                        for c in range(target_rank):
                            s = 0.0
                            for j in range(m_sub):
                                s += U[i * m_sub + j, c] * V[j * p_sub + k, c]
                            C[k * n_sub + i, c] = s
                for r in range(p_sub * n_sub):
                    if np.linalg.norm(C[r, :]) > 1e-10:
                        W[r, :] = np.linalg.lstsq(C[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                
                # Check error
                err = 0.0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            val = np.sum(U[i * m_sub + j, :] * V[j * p_sub + k, :] * W[k * n_sub + i, :])
                            err += (val - 1) ** 2
                
                if err < 1e-8:
                    return U, V, W, target_rank
        
        return None, None, None, None

    # Try to find 3x3x3 rank 23
    U4, V4, W4, r4 = find_3x3x3_rank23()
    
    if r4 is not None and r4 < 27:
        # Build full decomposition with 3x3x3 at rank r4
        # 3x2x2: schoolbook rank 12
        U1, V1, W1, r1 = schoolbook(3, 2, 2)
        # 3x3x2: schoolbook rank 18
        U2, V2, W2, r2 = schoolbook(3, 3, 2)
        # 3x2x3: schoolbook rank 18
        U3, V3, W3, r3 = schoolbook(3, 2, 3)
        
        total_rank = r1 + r2 + r3 + r4
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        col = 0
        # Block 1: 3x2x2 (i:0-2, j:0-1, k:0-1)
        for i in range(3):
            for j in range(2):
                for c in range(r1):
                    U_full[i * m + j, col + c] = U1[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                for c in range(r1):
                    V_full[j * p + k, col + c] = V1[j * 2 + k, c]
        for k in range(2):
            for i in range(3):
                for c in range(r1):
                    W_full[k * n + i, col + c] = W1[k * 3 + i, c]
        col += r1
        
        # Block 2: 3x3x2 (i:0-2, j:2-4, k:0-1)
        for i in range(3):
            for j in range(3):
                for c in range(r2):
                    U_full[i * m + (j + 2), col + c] = U2[i * 3 + j, c]
        for j in range(3):
            for k in range(2):
                for c in range(r2):
                    V_full[(j + 2) * p + k, col + c] = V2[j * 2 + k, c]
        for k in range(2):
            for i in range(3):
                for c in range(r2):
                    W_full[k * n + i, col + c] = W2[k * 3 + i, c]
        col += r2
        
        # Block 3: 3x2x3 (i:0-2, j:0-1, k:2-4)
        for i in range(3):
            for j in range(2):
                for c in range(r3):
                    U_full[i * m + j, col + c] = U3[i * 2 + j, c]
        for j in range(2):
            for k in range(3):
                for c in range(r3):
                    V_full[j * p + (k + 2), col + c] = V3[j * 3 + k, c]
        for k in range(3):
            for i in range(3):
                for c in range(r3):
                    W_full[(k + 2) * n + i, col + c] = W3[k * 3 + i, c]
        col += r3
        
        # Block 4: 3x3x3 (i:0-2, j:2-4, k:2-4)
        for i in range(3):
            for j in range(3):
                for c in range(r4):
                    U_full[i * m + (j + 2), col + c] = U4[i * 3 + j, c]
        for j in range(3):
            for k in range(3):
                for c in range(r4):
                    V_full[(j + 2) * p + (k + 2), col + c] = V4[j * 3 + k, c]
        for k in range(3):
            for i in range(3):
                for c in range(r4):
                    W_full[(k + 2) * n + i, col + c] = W4[k * 3 + i, c]
        col += r4
        
        # Verify
        if verify(U_full, V_full, W_full, n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: schoolbook
    U, V, W, _ = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
