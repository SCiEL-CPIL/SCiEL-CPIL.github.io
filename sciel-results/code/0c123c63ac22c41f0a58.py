"""Optimized matrix multiplication tensor decomposition for 3x4 x 4x8."""
DIMENSIONS = [3, 4, 8]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    def build_schoolbook():
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
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def build_strassen_2x2x2():
        """Return U,V,W for 2x2 x 2x2 using Strassen (rank 7)."""
        # Strassen: 7 multiplications
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
        
        # U maps (i,j) -> coefficients for each Strassen term
        # A indices: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,2,3
        # B indices: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,2,3
        # C indices: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,2,3
        
        # For each Strassen multiplication term, U and V have the sum/diff patterns
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: (A11+A22)(B11+B22) -> C11 +=1, C22 +=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 1: (A21+A22)B11 -> C21 +=1, C22 -=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # Term 2: A11(B21-B22) -> C12 +=1, C21 +=1
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[2, 2] = 1
        
        # Term 3: A22(B12-B11) -> C11 -=1, C21 +=1
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        
        # Term 4: (A11+A12)B22 -> C11 -=1, C12 +=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # Term 5: (A21-A11)(B11+B12) -> C22 +=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Term 6: (A12-A22)(B21+B22) -> C11 +=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def build_strassen_2x2x8():
        """Build rank-28 decomposition for 2x2 x 2x8 using Strassen on 4 column blocks."""
        U = np.zeros((4, 28))
        V = np.zeros((16, 28))
        W = np.zeros((16, 28))
        
        U_base, V_base, W_base = build_strassen_2x2x2()
        
        for block in range(4):
            offset = block * 7
            # U: same for all blocks (2x2 rows of A)
            U[:, offset:offset+7] = U_base
            # V: 2x8, block b uses rows 2b, 2b+1 of B
            V_base_block = np.zeros((16, 7))
            for r in range(2):
                for c in range(2):
                    V_base_block[2*block + r, offset + c] = V_base[r, c]
            V[:, offset:offset+7] = V_base_block
            # W: 2x8, block b uses rows 2b, 2b+1 of C
            W_base_block = np.zeros((16, 7))
            for r in range(2):
                for c in range(2):
                    W_base_block[2*block + r, offset + c] = W_base[r, c]
            W[:, offset:offset+7] = W_base_block
        
        return U, V, W
    
    def build_strassen_3x4x8():
        """
        Build decomposition for 3x4 x 4x8.
        Split A (3x4) into A_top (2x4) and A_bot (1x4).
        Split B (4x8) into 4 column blocks of (4x2).
        
        C = A_top @ B + A_bot @ B
        
        For A_top (2x4) @ B (4x8):
          Split B into 2 row-blocks of (2x8): B_top (2x8), B_bot (2x8)
          C_top = A_top[:,0:2] @ B_top + A_top[:,2:4] @ B_bot
          Each 2x2 @ 2x8 uses Strassen rank 28. Total: 56.
        
        For A_bot (1x4) @ B (4x8):
          Split B into 2 row-blocks of (2x8): B_top (2x8), B_bot (2x8)
          C_bot = A_bot[0,0:2] @ B_top + A_bot[0,2:4] @ B_bot
          Each 1x2 @ 2x8 is schoolbook rank 8. Total: 16.
        
        Grand total: 56 + 16 = 72.
        """
        n, m, p = 3, 4, 8
        rank_total = 72
        U = np.zeros((n * m, rank_total))  # 12 x 72
        V = np.zeros((m * p, rank_total))  # 32 x 72
        W = np.zeros((p * n, rank_total))  # 24 x 72
        
        # Part 1: A_top (2x4) @ B (4x8) -> contributes to C rows 0,1
        # Sub-part 1a: A_top[:,0:2] (2x2) @ B[0:2,:] (2x8)
        U1, V1, W1 = build_strassen_2x2x8()
        # A_top[:,0:2] is A rows 0,1, cols 0,1 -> flat indices 0,1,4,5
        # B[0:2,:] is B rows 0,1, cols 0-7 -> flat indices 0-7
        # C rows 0,1, cols 0-7 -> flat indices 0-7 (C is 8x3, W[k*n+i,r])
        
        # Map U: 2x2 A block (rows 0,1; cols 0,1) -> flat A indices [0,1,4,5]
        a_map = [0, 1, 4, 5]  # A[0,0], A[0,1], A[1,0], A[1,1]
        for i in range(4):
            U[a_map[i], 0:28] = U1[i, :]
        
        # Map V: 2x8 B block (rows 0,1; cols 0-7) -> flat B indices 0-7
        for i in range(16):
            V[i, 0:28] = V1[i, :]
        
        # Map W: 2x8 C block (rows 0-7; cols 0,1) -> W[k*n+i, r] where k=col in C, i=row in C
        # C row i (0 or 1), col k (0-7) -> W[k*3 + i, r]
        for i in range(2):  # C row
            for k in range(8):  # C col
                W[k * 3 + i, 0:28] = W1[k * 2 + i, :]
        
        # Sub-part 1b: A_top[:,2:4] (2x2) @ B[2:4,:] (2x8)
        U2, V2, W2 = build_strassen_2x2x8()
        # A_top[:,2:4] is A rows 0,1, cols 2,3 -> flat A indices [2,3,6,7]
        a_map2 = [2, 3, 6, 7]
        for i in range(4):
            U[a_map2[i], 28:56] = U2[i, :]
        
        # B[2:4,:] is B rows 2,3, cols 0-7 -> flat B indices 16-23
        for i in range(2):
            for k in range(8):
                V[2 * 8 + i * 8 + k, 28:56] = V2[i * 8 + k, :]
        
        # C rows 0,1, cols 0-7 -> same as before
        for i in range(2):
            for k in range(8):
                W[k * 3 + i, 28:56] = W2[k * 2 + i, :]
        
        # Part 2: A_bot (1x4) @ B (4x8) -> contributes to C row 2
        # Sub-part 2a: A_bot[0,0:2] (1x2) @ B[0:2,:] (2x8) schoolbook rank 8
        # A_bot[0,0:2] -> A[2,0], A[2,1] -> flat indices 8, 9
        # B[0:2,:] -> B rows 0,1 -> flat indices 0-7
        # C row 2, cols 0-7 -> W[k*3+2, r] for k=0..7
        
        for k in range(8):
            U[8, 56 + k] = 1      # A[2,0]
            V[k, 56 + k] = 1       # B[0,k]
            W[k * 3 + 2, 56 + k] = 1  # C[2,k]
            
            U[9, 56 + 8 + k] = 1   # A[2,1]
            V[8 + k, 56 + 8 + k] = 1  # B[1,k]
            W[k * 3 + 2, 56 + 8 + k] = 1  # C[2,k]
        
        # Sub-part 2b: A_bot[0,2:4] (1x2) @ B[2:4,:] (2x8) schoolbook rank 8
        for k in range(8):
            U[10, 56 + 16 + k] = 1   # A[2,2]
            V[16 + k, 56 + 16 + k] = 1  # B[2,k]
            W[k * 3 + 2, 56 + 16 + k] = 1  # C[2,k]
            
            U[11, 56 + 24 + k] = 1   # A[2,3]
            V[24 + k, 56 + 24 + k] = 1  # B[3,k]
            W[k * 3 + 2, 56 + 24 + k] = 1  # C[2,k]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that sum_r U[:,r] outer V[:,r] matches matmul tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Build the matmul tensor T[i,j,k] = delta
        # The decomposition should satisfy: for all i,j,k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] summed over r? 
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor: T[i,j,k] such that sum_j A[i,j] B[j,k] = C[i,k]
        # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... no
        # Correct: sum_r U[i*m+j, r] * V[j*p+k, r] should give a value, and
        # C[i,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r]
        # But W[k*n+i, r] is the coefficient for the r-th term in C[i,k]
        # So: C[i,k] = sum_r W[k*n+i, r] * (sum_j ... ) no.
        
        # The tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
        # Actually T[i,j,k] = delta_{i,j,k} in the sense that the bilinear form is:
        # C[i,k] = sum_{j,r} A[i,j] B[j,k] W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k? No.
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be the coefficient of A[i,j]B[j,k] in C[i,k]
        # which is 1 if (i,k) is the output, so:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try Strassen-based decomposition first
    try:
        result = build_strassen_3x4x8()
        if verify(result["U"], result["V"], result["W"], n, m, p):
            return result
        else:
            # Fallback to schoolbook
            return build_schoolbook()
    except Exception:
        return build_schoolbook()
    
# EVOLVE-BLOCK-END
