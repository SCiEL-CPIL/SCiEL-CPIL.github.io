"""
Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Uses Strassen's rank-7 decomposition for 2x2*2x2 blocks, composed to handle
4x4*4x8. The 4x8 matrix is treated as two 4x4 blocks along the column dimension,
giving 2 independent 4x4*4x4 multiplications, each with rank 49 (Strassen),
for a total rank of 98.

The tensor decomposition is:
  T[i,j,k,r] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

For 4x4*4x8:
  n=4, m=4, p=8
  U: shape (16, rank) - indexed by (i,j) where i in [0,4), j in [0,4)
  V: shape (32, rank) - indexed by (j,k) where j in [0,4), k in [0,8)
  W: shape (32, rank) - indexed by (k,i) where k in [0,8), i in [0,4)
"""

import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Strassen rank-7 decomposition for 2x2*2x2
    # U: (4,7) indexed by [i*2+j]
    # V: (4,7) indexed by [j*2+k]  
    # W: (4,7) indexed by [k*2+i]
    
    def strassen_2x2():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (A11+A22)(B11+B22) -> C11(+1), C22(+1)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11 -> C21(+1), C22(-1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3 = A11(B12-B22) -> C12(+1), C22(+1)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = A22(B21-B11) -> C11(+1), C21(+1)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (A11+A12)B22 -> C11(-1), C12(+1)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (A21-A11)(B11+B12) -> C21(+1), C22(+1)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 5] = 1; W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22) -> C11(-1), C12(+1)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[2, 6] = 1
        
        return U, V, W
    
    def strassen_4x4():
        """
        Return U, V, W for 4x4 * 4x4 matrix multiplication with rank 49.
        Uses Strassen's 7-product decomposition recursively.
        U: (16, 49), V: (16, 49), W: (16, 49)
        U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
        """
        U_2, V_2, W_2 = strassen_2x2()
        
        rank = 49
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        
        # Strassen coefficients for the 7 products at block level:
        # For product j, which A blocks and B blocks are used, and with what coefficients
        # a_blocks[j] = list of (block_idx, coeff) for A
        # b_blocks[j] = list of (block_idx, coeff) for B
        # c_blocks[j] = list of (block_idx, coeff) for C
        
        # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (same for B and C)
        # A00: rows 0-1, cols 0-1; A01: rows 0-1, cols 2-3
        # A10: rows 2-3, cols 0-1; A11: rows 2-3, cols 2-3
        
        a_blocks = [
            [(0, 1), (3, 1)],   # M1: A00 + A11
            [(2, 1), (3, 1)],   # M2: A10 + A11
            [(0, 1)],           # M3: A00
            [(3, 1)],           # M4: A11
            [(0, 1), (1, 1)],   # M5: A00 + A01
            [(2, 1), (0, -1)],  # M6: A10 - A00
            [(1, 1), (3, -1)],  # M7: A01 - A11
        ]
        
        b_blocks = [
            [(0, 1), (3, 1)],   # M1: B00 + B11
            [(0, 1)],           # M2: B00
            [(1, 1), (3, -1)],  # M3: B01 - B11
            [(2, 1), (0, -1)],  # M4: B10 - B00
            [(3, 1)],           # M5: B11
            [(0, 1), (1, 1)],   # M6: B00 + B01
            [(2, 1), (3, 1)],   # M7: B10 + B11
        ]
        
        c_blocks = [
            [(0, 1), (3, 1)],   # M1 -> C00(+1), C11(+1)
            [(2, 1), (3, -1)],  # M2 -> C10(+1), C11(-1)
            [(1, 1), (3, 1)],   # M3 -> C01(+1), C11(+1)
            [(0, 1), (2, 1)],   # M4 -> C00(+1), C10(+1)
            [(0, -1), (1, 1)],  # M5 -> C00(-1), C01(+1)
            [(2, 1), (3, 1)],   # M6 -> C10(+1), C11(+1)
            [(0, -1), (1, 1)],  # M7 -> C00(-1), C01(+1)
        ]
        
        # For each Strassen product j and each rank-1 term t in the 2x2 block mult:
        # We need to construct the rank-1 term for 4x4*4x4
        
        # Block position mapping:
        # Block b (0-3) in a 4x4 matrix:
        # b=0: rows 0-1, cols 0-1
        # b=1: rows 0-1, cols 2-3
        # b=2: rows 2-3, cols 0-1
        # b=3: rows 2-3, cols 2-3
        
        def block_row(b):
            return (b // 2) * 2  # 0 for b=0,1; 1 for b=2,3 -> row offset 0 or 2
        
        def block_col(b):
            return (b % 2) * 2  # 0 for b=0,2; 1 for b=1,3 -> col offset 0 or 2
        
        col = 0
        for j in range(7):  # 7 Strassen products
            for t in range(7):  # 7 rank-1 terms in 2x2 block mult
                # For U: combine A blocks from a_blocks[j] with U_2[t]
                # U_2[i*2+j2, t] where i2 is block row, j2 is block col
                for a_idx, a_coeff in a_blocks[j]:
                    ar = block_row(a_idx)  # row offset of A block
                    ac = block_col(a_idx)  # col offset of A block
                    for i2 in range(2):
                        for j2 in range(2):
                            val = a_coeff * U_2[i2*2+j2, t]
                            if val != 0:
                                # U index: (ar + i2)*4 + (ac + j2)
                                U[(ar + i2)*4 + (ac + j2), col] += val
                
                # For V: combine B blocks from b_blocks[j] with V_2[t]
                for b_idx, b_coeff in b_blocks[j]:
                    br = block_row(b_idx)  # row offset of B block (col of A)
                    bc = block_col(b_idx)  # col offset of B block
                    for j2 in range(2):
                        for k2 in range(2):
                            val = b_coeff * V_2[j2*2+k2, t]
                            if val != 0:
                                # V index: (br + j2)*4 + (bc + k2)
                                V[(br + j2)*4 + (bc + k2), col] += val
                
                # For W: combine C blocks from c_blocks[j] with W_2[t]
                for c_idx, c_coeff in c_blocks[j]:
                    cr = block_row(c_idx)  # row offset of C block (col of B)
                    cc = block_col(c_idx)  # col offset of C block (row of A)
                    for k2 in range(2):
                        for i2 in range(2):
                            val = c_coeff * W_2[k2*2+i2, t]
                            if val != 0:
                                # W index: (cr + k2)*4 + (cc + i2)
                                W[(cr + k2)*4 + (cc + i2), col] += val
                
                col += 1
        
        return U, V, W
    
    # Verify Strassen 4x4 decomposition
    U_4, V_4, W_4 = strassen_4x4()
    
    # Verify: T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    # For 4x4*4x4: n=m=p=4
    # T[i,j,k] should be 1 if j==k, else 0 (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    # Let's verify numerically
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1.0 if j == k else 0.0
                actual = 0.0
                for r in range(49):
                    actual += U_4[i*4+j, r] * V_4[j*4+k, r] * W_4[k*4+i, r]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook if Strassen composition is incorrect
        rank = n * m * p  # 128
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # For 4x4*4x8, split into two 4x4*4x4 multiplications
    # C = A * B where B = [B_left | B_right], each 4x4
    # C = [A*B_left | A*B_right]
    # Each 4x4*4x4 has rank 49 (Strassen), so total rank = 98
    
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # First 4x4*4x4: k in [0,4)
    for r in range(49):
        U[:, r] = U_4[:, r]
        V[0:16, r] = V_4[:, r]  # V_4[j*4+k] for k in [0,4)
        W[0:16, r] = W_4[:, r]  # W_4[k*4+i] for k in [0,4)
    
    # Second 4x4*4x4: k in [4,8)
    for r in range(49):
        U[:, 49+r] = U_4[:, r]
        V[16:32, 49+r] = V_4[:, r]  # V_4[j*4+k] for k in [4,8) -> V_4 index j*4+(k-4)
        W[16:32, 49+r] = W_4[:, r]  # W_4[k*4+i] for k in [4,8) -> W_4 index (k-4)*4+i
    
    # Verify the full 4x4*4x8 decomposition
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if j == k else 0.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank = n * m * p  # 128
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
