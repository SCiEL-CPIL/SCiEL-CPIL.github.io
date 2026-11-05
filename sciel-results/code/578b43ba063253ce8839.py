import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 12
    N_W = p * n  # 8
    
    # Build target tensor T of shape (N_U, N_V, N_W)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((N_U, N_V, N_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # 20s margin from 120s limit
    
    # Try to find rank-22 decomposition using structured Strassen approach
    # then fall back to schoolbook rank-24 if needed
    best_rank = 24
    best_decomp = None
    
    # Structured rank-22 construction using Strassen on 2x2 blocks
    # A (2x3) * B (3x4) = C (2x4)
    # Split: A = [A1 A2] where A1 is 2x2, A2 is 2x1
    # B = [B1 B2 B3] where B1,B2,B3 are 3x... 
    # Actually, let's use a known rank-22 scheme.
    
    # Rank-22 decomposition:
    # Use Strassen for the 2x2 core (rank 7) plus 15 additional terms
    # for the cross terms involving the 3rd column/row.
    
    # Let me construct this explicitly.
    # A[i,j] for i in {0,1}, j in {0,1,2}
    # B[j,k] for j in {0,1,2}, k in {0,1,2,3}
    # C[i,k] for i in {0,1}, k in {0,1,2,3}
    
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Strassen on the 2x2 part (j in {0,1}, k in {0,1}):
    # M1 = (A[i,0]+A[i,1])*(B[0,0]+B[1,1])  -> but this needs to work for both i
    # Standard Strassen for 2x2*2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B11)  -- wait, let me be careful
    
    # For multiplying A_core (2x2) by B_core (2x2):
    # A_core = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B_core = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # C_core = A_core * B_core (2x2)
    
    # Strassen:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0] - B[1,1])
    # M4 = A[1,1] * (B[0,0] - B[0,1])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # But our B has 4 columns, not 2. So we need to extend.
    # Let me split B into B_left (3x2) and B_right (3x2).
    # C_left = A * B_left (2x2), C_right = A * B_right (2x2)
    
    # For C_left: use Strassen on A's first 2 cols and B_left's first 2 rows... 
    # But A is 2x3, so A * B_left = A[:,0:2] * B_left[0:2,:] + A[:,2] * B_left[2,:]
    
    # Let me use a different approach: decompose each 2x3*3x2 product with rank 11,
    # but share some structure to get total rank 22.
    
    # Actually, the cleanest known rank-22 for 2x3x4:
    # Use 7 Strassen terms for the "main" 2x2*2x2*2x2 block,
    # and 15 terms for the remaining cross terms.
    
    # Let me just verify: is there a known explicit rank-22?
    # The cell finding says yes, score 1.091 = 24/22.
    
    # Let me construct it step by step.
    # I'll use the following scheme:
    # Split the 3 intermediate dimension into 2+1.
    # Split the 4 output dimension into 2+2.
    
    # Terms:
    # 1-7: Strassen for A[:,0:2] * B[0:2, 0:2] -> contributes to C[:, 0:2]
    # 8-14: Strassen for A[:,0:2] * B[0:2, 2:4] -> contributes to C[:, 2:4]
    # 15-18: A[:,2] * B[2, 0:2] -> 2x1 * 1x2 = 2x2, needs 4 terms (or fewer with structure)
    # 19-22: A[:,2] * B[2, 2:4] -> 2x1 * 1x2 = 2x2, needs 4 terms
    
    # Wait, that's 7+7+4+4 = 22! 
    
    # But A[:,2] * B[2, 0:2] is a 2x2 matrix = outer product of A[:,2] (2x1) and B[2,0:2] (1x2)
    # This is rank 1 as a bilinear map? No, it's a sum of 4 products:
    # C[i,k] += A[i,2] * B[2,k] for i in {0,1}, k in {0,1}
    # Each (i,k) pair needs its own term: A[i,2]*B[2,k]
    # So 4 terms for this block.
    
    # Similarly 4 terms for A[:,2] * B[2, 2:4].
    
    # Total: 7 + 7 + 4 + 4 = 22. 
    
    # Let me define the 22 terms.
    rank = 22
    U = [[0.0] * rank for _ in range(N_U)]
    V = [[0.0] * rank for _ in range(N_V)]
    W = [[0.0] * rank for _ in range(N_W)]
    
    # Helper: A[i,j] index in U: i*m + j = i*3 + j
    # B[j,k] index in V: j*p + k = j*4 + k
    # C[i,k] index in W: k*n + i = k*2 + i
    
    def a_idx(i, j):
        return i * m + j
    
    def b_idx(j, k):
        return j * p + k
    
    def c_idx(i, k):
        return k * n + i
    
    # Strassen for 2x2 * 2x2:
    # Input: A_block (rows 0,1; cols 0,1 of A), B_block (rows 0,1; cols k0, k0+1 of B)
    # Output: C_block (rows 0,1; cols k0, k0+1 of C)
    
    def strassen_terms(term_start, a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        """
        Add 7 Strassen terms starting at term_start.
        a_row_offset: which rows of A (0 or 1 base)
        a_col_offset: which cols of A (0 or 1 base) 
        b_row_offset: which rows of B (0 or 1 base)
        b_col_offset: which cols of B (0 or 1 base)
        Returns next term index.
        """
        # A sub-block: rows [a_row_offset, a_row_offset+1], cols [a_col_offset, a_col_offset+1]
        # B sub-block: rows [b_row_offset, b_row_offset+1], cols [b_col_offset, b_col_offset+1]
        # C sub-block: rows [a_row_offset, a_row_offset+1], cols [b_col_offset, b_col_offset+1]
        
        A00 = a_idx(a_row_offset, a_col_offset)
        A01 = a_idx(a_row_offset, a_col_offset + 1)
        A10 = a_idx(a_row_offset + 1, a_col_offset)
        A11 = a_idx(a_row_offset + 1, a_col_offset + 1)
        
        B00 = b_idx(b_row_offset, b_col_offset)
        B01 = b_idx(b_row_offset, b_col_offset + 1)
        B10 = b_idx(b_row_offset + 1, b_col_offset)
        B11 = b_idx(b_row_offset + 1, b_col_offset + 1)
        
        C00 = c_idx(a_row_offset, b_col_offset)
        C01 = c_idx(a_row_offset, b_col_offset + 1)
        C10 = c_idx(a_row_offset + 1, b_col_offset)
        C11 = c_idx(a_row_offset + 1, b_col_offset + 1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[A00][term_start] += 1
        U[A11][term_start] += 1
        V[B00][term_start] += 1
        V[B11][term_start] += 1
        W[C00][term_start] += 1
        W[C11][term_start] += 1
        term_start += 1
        
        # M2 = (A10 + A11) * B00
        U[A10][term_start] += 1
        U[A11][term_start] += 1
        V[B00][term_start] += 1
        W[C10][term_start] += 1
        term_start += 1
        
        # M3 = A00 * (B10 - B11)
        U[A00][term_start] += 1
        V[B10][term_start] += 1
        V[B11][term_start] -= 1
        W[C01][term_start] += 1
        term_start += 1
        
        # M4 = A11 * (B00 - B01)
        U[A11][term_start] += 1
        V[B00][term_start] += 1
        V[B01][term_start] -= 1
        W[C00][term_start] += 1
        W[C10][term_start] += 1
        term_start += 1
        
        # M5 = (A00 + A01) * B11
        U[A00][term_start] += 1
        U[A01][term_start] += 1
        V[B11][term_start] += 1
        W[C00][term_start] -= 1
        W[C01][term_start] += 1
        term_start += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[A10][term_start] += 1
        U[A00][term_start] -= 1
        V[B00][term_start] += 1
        V[B01][term_start] += 1
        W[C11][term_start] += 1
        term_start += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[A01][term_start] += 1
        U[A11][term_start] -= 1
        V[B10][term_start] += 1
        V[B11][term_start] += 1
        W[C00][term_start] += 1
        W[C11][term_start] -= 1
        term_start += 1
        
        return term_start
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # a_row_offset=0, a_col_offset=0, b_row_offset=0, b_col_offset=0
    term = 0
    term = strassen_terms(term, 0, 0, 0, 0)
    # term should be 7
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # a_row_offset=0, a_col_offset=0, b_row_offset=0, b_col_offset=2
    term = strassen_terms(term, 0, 0, 0, 2)
    # term should be 14
    
    # Block 3: A[:, 2] * B[2, 0:2] -> C[:, 0:2]
    # This is outer product: C[i,k] += A[i,2] * B[2,k]
    # 4 terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
    for i in range(2):
        for k in range(2):
            U[a_idx(i, 2)][term] = 1
            V[b_idx(2, k)][term] = 1
            W[c_idx(i, k)][term] = 1
            term += 1
    # term should be 18
    
    # Block 4: A[:, 2] * B[2, 2:4] -> C[:, 2:4]
    # 4 terms
    for i in range(2):
        for k in range(2, 4):
            U[a_idx(i, 2)][term] = 1
            V[b_idx(2, k)][term] = 1
            W[c_idx(i, k)][term] = 1
            term += 1
    # term should be 22
    
    assert term == 22, f"Expected 22 terms, got {term}"
    
    # Verify the decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    error = 0.0
    for a in range(N_U):
        for b in range(N_V):
            for c in range(N_W):
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                target = T[a, b, c]
                if abs(val - target) > 1e-9:
                    error += abs(val - target)
    
    if error < 1e-9:
        best_rank = 22
        best_decomp = (U_arr, V_arr, W_arr)
    else:
        # Fallback to schoolbook rank 24
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
    
    U, V, W = best_decomp
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
