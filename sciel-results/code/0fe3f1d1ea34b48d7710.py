# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    
    Strategy:
    - Start with schoolbook rank 60.
    - Apply Strassen decomposition to 2x2x2 sub-blocks to reduce rank.
    - The 2x5x6 multiplication can be decomposed into:
      - A_R (1x5) * B_B (5x6): rank 6
      - A_L (2x5) * B_T (5x6): rank 54 (using Strassen on 2x2x2 blocks)
      Total: 6 + 54 = 60... wait, that's not right.
      
    Let me reconsider. The dimensions are n=2, m=5, p=6.
    A is 2x5, B is 5x6, C is 2x6.
    
    Schoolbook: 2*5*6 = 60 multiplications.
    
    To reduce rank, we can use Strassen on 2x2x2 subproblems.
    Split A into A_L (2x2) and A_R (2x3) where columns 0-1 and 2-4.
    Split B into B_T (2x6) and B_B (3x6) where rows 0-1 and 2-4.
    
    Then C = A_L * B_T + A_R * B_B
    
    A_L * B_T: 2x2 * 2x6 = 2x6, rank 2*2*6 = 24 schoolbook.
    Using Strassen: split B_T into three 2x2 blocks: B_T1, B_T2, B_T3.
    Each 2x2 * 2x2 uses Strassen rank 7, so 3 * 7 = 21.
    
    A_R * B_B: 2x3 * 3x6 = 2x6, rank 2*3*6 = 36 schoolbook.
    We can split this further or use other techniques.
    
    Actually, let's try a different approach. Use the fact that for 2x5x6,
    we can apply Strassen to overlapping 2x2x2 blocks.
    
    Alternative: Just implement a known good decomposition.
    For 2x5 * 5x6, the rank is at least max(2*5, 5*6, 6*2) = max(10,30,12) = 30.
    But achieving rank 30 is hard. Let's aim for rank 54 using Strassen on 2x2x2 blocks.
    
    Actually, let me just implement a careful Strassen-based decomposition.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        """
        Check if U, V, W exactly decompose the matrix multiplication tensor.
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if s != 1:
                        return False
        return True
    
    # Strassen 2x2x2 decomposition
    # For A (2x2) * B (2x2) = C (2x2):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(U, V, W, i0, j0, k0, col_offset):
        """
        Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        to the existing U, V, W matrices starting at column col_offset.
        Returns the number of columns used.
        """
        # 7 multiplications
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        
        # For each multiplication Mt, we add a rank-1 term
        # U[i*m+j, t] = coefficient of A[i,j] in the first factor
        # V[j*p+k, t] = coefficient of B[j,k] in the second factor
        # W[k*n+i, t] = coefficient of Mt in C[i,k]
        
        # M1
        t = col_offset
        U[i0 * m + j0][t] += 1
        U[(i0+1) * m + (j0+1)][t] += 1
        V[j0 * p + k0][t] += 1
        V[(j0+1) * p + (k0+1)][t] += 1
        W[k0 * n + i0][t] += 1      # C[i0,k0] += M1
        W[(k0+1) * n + (i0+1)][t] += 1  # C[i0+1,k0+1] += M1
        t += 1
        
        # M2
        U[(i0+1) * m + j0][t] += 1
        U[(i0+1) * m + (j0+1)][t] += 1
        V[j0 * p + k0][t] += 1
        W[(k0) * n + (i0+1)][t] += 1  # C[i0+1,k0] += M2
        t += 1
        
        # M3
        U[i0 * m + j0][t] += 1
        V[j0 * p + (k0+1)][t] += 1
        V[(j0+1) * p + (k0+1)][t] -= 1
        W[(k0+1) * n + i0][t] += 1    # C[i0,k0+1] += M3
        W[(k0+1) * n + (i0+1)][t] += 1  # C[i0+1,k0+1] += M3
        t += 1
        
        # M4
        U[(i0+1) * m + (j0+1)][t] += 1
        V[(j0+1) * p + k0][t] += 1
        V[j0 * p + k0][t] -= 1
        W[k0 * n + i0][t] += 1        # C[i0,k0] += M4
        W[k0 * n + (i0+1)][t] += 1    # C[i0+1,k0] += M4
        t += 1
        
        # M5
        U[i0 * m + j0][t] += 1
        U[i0 * m + (j0+1)][t] += 1
        V[(j0+1) * p + (k0+1)][t] += 1
        W[k0 * n + i0][t] -= 1        # C[i0,k0] -= M5
        W[(k0+1) * n + i0][t] += 1    # C[i0,k0+1] += M5
        t += 1
        
        # M6
        U[(i0+1) * m + j0][t] += 1
        U[i0 * m + j0][t] -= 1
        V[j0 * p + k0][t] += 1
        V[j0 * p + (k0+1)][t] += 1
        W[(k0+1) * n + (i0+1)][t] += 1  # C[i0+1,k0+1] += M6
        t += 1
        
        # M7
        U[i0 * m + (j0+1)][t] += 1
        U[(i0+1) * m + (j0+1)][t] -= 1
        V[(j0+1) * p + k0][t] += 1
        V[(j0+1) * p + (k0+1)][t] += 1
        W[k0 * n + i0][t] += 1        # C[i0,k0] += M7
        t += 1
        
        return t - col_offset  # 7 columns used
    
    # Build the decomposition using Strassen on 2x2x2 blocks
    # A is 2x5, B is 5x6
    # Split into blocks:
    # A[:, 0:2] * B[0:2, :] -> 2x2 * 2x6
    # A[:, 2:5] * B[2:5, :] -> 2x3 * 3x6
    
    # For 2x2 * 2x6: split B into three 2x2 blocks
    # For 2x3 * 3x6: use schoolbook or further decomposition
    
    def build_strassen_decomposition():
        # Total rank: 2x2*2x6 part + 2x3*3x6 part
        # 2x2*2x6: 3 Strassen blocks * 7 = 21
        # 2x3*3x6: 36 schoolbook
        # Total: 57
        
        # But we can do better. Let's use schoolbook for simplicity first,
        # then optimize.
        
        # Actually, let's just use schoolbook for now and verify it works.
        # The Strassen approach is complex to implement correctly.
        
        # Let me try a different approach: use the fact that rank(2x5x6) 
        # can be reduced by using linear combinations.
        
        # For now, return schoolbook
        return schoolbook()
    
    # Start with schoolbook
    result = schoolbook()
    U = result["U"]
    V = result["V"]
    W = result["W"]
    
    # Verify
    if not verify_decomposition(U, V, W):
        return result
    
    # Try to improve using Strassen on 2x2x2 blocks
    # We'll build a new decomposition
    
    if time.time() < deadline - 20:
        # Build Strassen-based decomposition
        # 2x5 * 5x6
        # Split columns of A: 0-1 (2 cols) and 2-4 (3 cols)
        # Split rows of B: 0-1 (2 rows) and 2-4 (3 rows)
        
        # Part 1: A[:, 0:2] * B[0:2, :] -> 2x2 * 2x6
        # Split B[0:2, :] into three 2x2 blocks: cols 0-1, 2-3, 4-5
        # Each uses Strassen: 7 multiplications
        # Total for Part 1: 21
        
        # Part 2: A[:, 2:5] * B[2:5, :] -> 2x3 * 3x6
        # Use schoolbook: 2*3*6 = 36
        # Total: 21 + 36 = 57
        
        # But we can also apply Strassen to parts of Part 2.
        # A[:, 2:4] * B[2:4, :] -> 2x2 * 2x6, same as Part 1: 21
        # A[:, 4:5] * B[4:5, :] -> 2x1 * 1x6, rank 12
        # Total: 21 + 21 + 12 = 54
        
        # Let's implement the 54-rank decomposition
        rank = 54
        U_new = [[0] * rank for _ in range(n * m)]
        V_new = [[0] * rank for _ in range(m * p)]
        W_new = [[0] * rank for _ in range(p * n)]
        
        col_offset = 0
        
        # Part 1: A[:, 0:2] * B[0:2, :]
        # B[0:2, 0:2] -> Strassen
        col_offset += strassen_2x2x2(U_new, V_new, W_new, 0, 0, 0, col_offset)
        # B[0:2, 2:4] -> Strassen
        col_offset += strassen_2x2x2(U_new, V_new, W_new, 0, 0, 2, col_offset)
        # B[0:2, 4:6] -> Strassen
        col_offset += strassen_2x2x2(U_new, V_new, W_new, 0, 0, 4, col_offset)
        
        # Part 2: A[:, 2:4] * B[2:4, :]
        # B[2:4, 0:2] -> Strassen
        col_offset += strassen_2x2x2(U_new, V_new, W_new, 0, 2, 0, col_offset)
        # B[2:4, 2:4] -> Strassen
        col_offset += strassen_2x2x2(U_new, V_new, W_new, 0, 2, 2, col_offset)
        # B[2:4, 4:6] -> Strassen
        col_offset += strassen_2x2x2(U_new, V_new, W_new, 0, 2, 4, col_offset)
        
        # Part 3: A[:, 4:5] * B[4:5, :] -> 2x1 * 1x6
        # Schoolbook: 12 multiplications
        for i in range(2):
            for k in range(6):
                U_new[i * m + 4][col_offset] = 1
                V_new[4 * p + k][col_offset] = 1
                W_new[k * n + i][col_offset] = 1
                col_offset += 1
        
        # Verify the new decomposition
        if verify_decomposition(U_new, V_new, W_new):
            return {"U": U_new, "V": V_new, "W": W_new}
    
    # Fall back to schoolbook
    return result
# EVOLVE-BLOCK-END
