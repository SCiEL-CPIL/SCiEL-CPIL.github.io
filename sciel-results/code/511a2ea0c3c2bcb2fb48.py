"""Rank-28 decomposition for 2x3 x 3x5 matrix multiplication using Strassen on 2x2 blocks."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# Strassen 2x2x2 decomposition
# For matrices A (2x2), B (2x2), C (2x2):
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A10)*B11
# M6 = (A11-A10)*(B00+B10)
# M7 = (A01-A11)*(B01+B11)
# C00 = M1+M4-M5+M7
# C01 = M3+M5
# C10 = M2+M4
# C11 = M1-M2+M3+M6

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # We'll build a rank-28 decomposition.
    # Strategy: Split the 3x5 matrix B into B1 (3x2) and B2 (3x3).
    # Split A (2x3) into A1 (2x2) and A2 (2x1).
    # 
    # C = A*B = A1*B1 + A2*B2
    # A1 is 2x2, B1 is 3x2 -> but we need to handle the middle dimension carefully.
    # 
    # Actually, let me use a more direct approach:
    # View the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Split j into j=0,1 (Strassen part) and j=2 (remainder)
    # For j=0,1: use Strassen on the 2x2x2 subproblem (A[:,0:2] @ B[0:2,:])
    # For j=2: schoolbook (A[:,2] @ B[2,:]) which is rank 2*5 = 10
    # 
    # Strassen gives 7 terms for the 2x2x2 part, but we need to map it to 2x3 x 3x5.
    # A[:,0:2] is 2x2, B[0:2,:] is 2x5.
    # The product A[:,0:2] @ B[0:2,:] is 2x5.
    # Using Strassen: we compute 7 intermediate 2x5 products? No.
    # 
    # Let me think differently. The tensor is T[i,j,k] = A[i,j]*B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,...,4}.
    # Total rank 30 schoolbook.
    # 
    # For j=0,1: 2x2x5 subproblem (A 2x2, B 2x5, C 2x5)
    # We can use Strassen-like decomposition: split B into two 2x2.5 blocks? Not clean.
    # 
    # Alternative: use the known result that 2x3x5 has rank 28.
    # Let me try a specific construction.
    # 
    # Split A into [A1 | A2] where A1 is 2x1, A2 is 2x2
    # Split B into [B1; B2] where B1 is 1x5, B2 is 2x5
    # C = A1*B1 + A2*B2
    # A1*B1: rank 1*5 = 5 (outer product of 2x1 and 1x5, but actually 2*5=10 if schoolbook)
    # Wait, A1 is 2x1, B1 is 1x5, so A1*B1 is 2x5, rank = 2*5/... no, the tensor rank for this subproblem is 2*1*5 = 10.
    # 
    # A2 is 2x2, B2 is 2x5. Product is 2x5.
    # For 2x2 x 2x5: we can split B2 into two 2x2.5 blocks? 
    # Actually, 2x2 x 2x5 can be done in rank 2*2*5 = 20 schoolbook.
    # Using Strassen on the 2x2 part: 7 multiplications of 2x2.5? No.
    # 
    # Let me try yet another approach. I'll use a computer algebra / linear algebra method.
    # 
    # The tensor T is in (R^2) * (R^3) * (R^5). We want rank-28 decomposition.
    # 
    # Let me just implement a simple rank-reduction:
    # Start with rank-30 schoolbook, then try to find 2 linear dependencies.
    # 
    # Actually, the simplest correct approach: use Strassen on a 2x2 subblock.
    # 
    # C = A @ B where A is 2x3, B is 3x5.
    # Write A = [A_left | A_right] where A_left is 2x2, A_right is 2x1.
    # Write B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 1x5.
    # C = A_left @ B_top + A_right @ B_bottom
    # 
    # Term 2: A_right @ B_bottom is 2x1 @ 1x5 = 2x5. Rank 2*1*5 = 10.
    # 
    # Term 1: A_left @ B_top is 2x2 @ 2x5.
    # Split B_top into [B_top_left | B_top_right] each 2x2.5? Not integer.
    # 
    # Let me split B_top into two 2x2 blocks plus a 2x1:
    # B_top = [B1 | B2 | B3] where B1,B2 are 2x2, B3 is 2x1.
    # A_left @ B_top = A_left @ B1 + A_left @ B2 + A_left @ B3
    # A_left @ B1: 2x2 @ 2x2, rank 7 (Strassen)
    # A_left @ B2: 2x2 @ 2x2, rank 7 (Strassen)
    # A_left @ B3: 2x2 @ 2x1, rank 4
    # Total for term 1: 7+7+4 = 18
    # Term 2: 10
    # Total: 18+10 = 28. Yes!
    
    # Let's build this.
    # A is 2x3: A[i,j] for i in 0,1; j in 0,1,2
    # B is 3x5: B[j,k] for j in 0,1,2; k in 0..4
    # 
    # A_left = A[:, 0:2] (2x2)
    # A_right = A[:, 2] (2x1)
    # B_top = B[0:2, :] (2x5) = [B1 | B2 | B3]
    #   B1 = B[0:2, 0:2] (2x2)
    #   B2 = B[0:2, 2:4] (2x2)
    #   B3 = B[0:2, 4:5] (2x1)
    # B_bottom = B[2:3, :] (1x5)
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 6 x 28
    V = [[0]*rank for _ in range(m*p)]  # 15 x 28
    W = [[0]*rank for _ in range(p*n)]  # 10 x 28
    
    col = 0
    
    # Strassen for 2x2 x 2x2: A_left @ B1
    # A_left[i,j], B1[j,k], result[i,k]
    # Strassen terms:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each Strassen term, U and V get contributions from the (A00+A11) type factors.
    # 
    # Let me define the Strassen decomposition properly.
    # For 2x2 matrices A and B, the 7 products are:
    # S1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # S2 = (A[1,0]+A[1,1]) * B[0,0]
    # S3 = A[0,0] * (B[0,1]-B[1,1])
    # S4 = A[1,1] * (B[1,0]-B[0,0])
    # S5 = (A[0,0]+A[0,1]) * B[1,1]
    # S6 = (A[1,1]-A[1,0]) * (B[0,0]+B[1,0])
    # S7 = (A[0,1]-A[1,1]) * (B[0,1]+B[1,1])
    
    # C[0,0] = S1 + S4 - S5 + S7
    # C[0,1] = S3 + S5
    # C[1,0] = S2 + S4
    # C[1,1] = S1 - S2 + S3 + S6
    
    # In our tensor, the indices are:
    # U[i*m + j, c] where i is row of A, j is column of A
    # V[j*p + k, c] where j is row of B, k is column of B
    # W[k*n + i, c] where k is column of B, i is row of C (= row of A)
    
    # For A_left @ B1:
    # A_left indices: i in {0,1}, j in {0,1}
    # B1 indices: j in {0,1}, k in {0,1}
    # Result: C[i,k] for i in {0,1}, k in {0,1}
    
    # In U: row index = i*3 + j (since m=3, but A_left only uses j=0,1)
    # In V: row index = j*5 + k (since p=5, B1 only uses k=0,1)
    # In W: row index = k*2 + i
    
    def add_strassen(A_row_offset, A_col_offset, B_row_offset, B_col_offset, col_start):
        """Add 7 Strassen terms for 2x2 @ 2x2.
        A is at A[A_row_offset:A_row_offset+2, A_col_offset:A_col_offset+2]
        B is at B[B_row_offset:B_row_offset+2, B_col_offset:B_col_offset+2]
        """
        nonlocal col
        # S1: (A00+A11)*(B00+B11)
        U[A_row_offset*3 + A_col_offset][col] += 1
        U[A_row_offset*3 + A_col_offset+1][col] += 1
        V[B_row_offset*5 + B_col_offset][col] += 1
        V[B_row_offset*5 + B_col_offset+1][col] += 1
        # W: C[0,0] += 1, C[1,1] += 1
        W[B_col_offset*2 + A_row_offset][col] += 1
        W[(B_col_offset+1)*2 + A_row_offset+1][col] += 1
        col += 1
        
        # S2: (A10+A11)*B00
        U[(A_row_offset+1)*3 + A_col_offset][col] += 1
        U[(A_row_offset+1)*3 + A_col_offset+1][col] += 1
        V[B_row_offset*5 + B_col_offset][col] += 1
        # W: C[1,0] += 1
        W[B_col_offset*2 + A_row_offset+1][col] += 1
        col += 1
        
        # S3: A00*(B01-B11)
        U[A_row_offset*3 + A_col_offset][col] += 1
        V[B_row_offset*5 + B_col_offset+1][col] += 1
        V[(B_row_offset+1)*5 + B_col_offset+1][col] -= 1
        # W: C[0,1] += 1
        W[(B_col_offset+1)*2 + A_row_offset][col] += 1
        col += 1
        
        # S4: A11*(B10-B00)
        U[(A_row_offset+1)*3 + A_col_offset+1][col] += 1
        V[(B_row_offset+1)*5 + B_col_offset][col] += 1
        V[B_row_offset*5 + B_col_offset][col] -= 1
        # W: C[0,0] += 1, C[1,0] += 1
        W[B_col_offset*2 + A_row_offset][col] += 1
        W[B_col_offset*2 + A_row_offset+1][col] += 1
        col += 1
        
        # S5: (A00+A01)*B11
        U[A_row_offset*3 + A_col_offset][col] += 1
        U[A_row_offset*3 + A_col_offset+1][col] += 1
        V[(B_row_offset+1)*5 + B_col_offset+1][col] += 1
        # W: C[0,1] += 1, C[0,0] -= 1
        W[(B_col_offset+1)*2 + A_row_offset][col] += 1
        W[B_col_offset*2 + A_row_offset][col] -= 1
        col += 1
        
        # S6: (A11-A10)*(B00+B10)
        U[(A_row_offset+1)*3 + A_col_offset+1][col] += 1
        U[(A_row_offset+1)*3 + A_col_offset][col] -= 1
        V[B_row_offset*5 + B_col_offset][col] += 1
        V[(B_row_offset+1)*5 + B_col_offset][col] += 1
        # W: C[1,1] += 1
        W[(B_col_offset+1)*2 + A_row_offset+1][col] += 1
        col += 1
        
        # S7: (A01-A11)*(B01+B11)
        U[A_row_offset*3 + A_col_offset+1][col] += 1
        U[(A_row_offset+1)*3 + A_col_offset+1][col] -= 1
        V[B_row_offset*5 + B_col_offset+1][col] += 1
        V[(B_row_offset+1)*5 + B_col_offset+1][col] += 1
        # W: C[0,0] += 1, C[0,1] += 1
        W[B_col_offset*2 + A_row_offset][col] += 1
        W[(B_col_offset+1)*2 + A_row_offset][col] += 1
        col += 1
    
    # Strassen for A_left @ B1: A at (0,0) to (1,1), B at (0,0) to (1,1)
    add_strassen(0, 0, 0, 0, col)
    
    # Strassen for A_left @ B2: A at (0,0) to (1,1), B at (0,2) to (1,3)
    add_strassen(0, 0, 0, 2, col)
    
    # Schoolbook for A_left @ B3: A at (0,0) to (1,1), B at (0,4) to (1,4)
    # This is 2x2 @ 2x1, rank 4
    for i in range(2):
        for j in range(2):
            U[i*3 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*2 + i][col] = 1
            col += 1
    
    # Schoolbook for A_right @ B_bottom: A at (0,2) to (1,2), B at (2,0) to (2,4)
    # This is 2x1 @ 1x5, rank 10
    for i in range(2):
        for k in range(5):
            U[i*3 + 2][col] = 1
            V[2*5 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    
    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

# Validation helper (not part of construct)
def _validate():
    result = construct()
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    n, m, p = 2, 3, 5
    # Reconstruct tensor
    # T[i*m+j, j*p+k, k*n+i] should equal delta
    # Actually, check: for random A, B, does A@B match?
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    C_comp = np.zeros((n, p))
    for c in range(U.shape[1]):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        # u is length n*m, v is length m*p, w is length p*n
        a_u = np.outer(np.sum(u.reshape(n, m), axis=1), np.ones(1))  # This isn't right
        # The decomposition is: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c])
        # Wait, that's not how it works. Let me re-derive.
        # The tensor is T[i,j,k] = A[i,j]*B[j,k].
        # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # But our U, V, W are matrices, not 3-tensors. The convention is:
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # And the reconstruction is: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j] * V[j*p+k,c]) * W[k*n+i,c] * B[j,k]
        # No, that's not right either.
        
        # Let me just check: the matrix multiplication tensor T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}
        # i.e., T[i,j,k] = 1 if the indices match the pattern for matrix multiplication.
        # Actually T[i,j,k] represents A[i,j]*B[j,k].
        # The rank-r decomposition means: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
    # So let's check:
    T_reconstructed = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    T_expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i, j, k] = 1.0  # A[i,j]*B[j,k] contributes to C[i,k]
    
    diff = np.max(np.abs(T_reconstructed - T_expected))
    print(f"Max diff: {diff}")
    return diff

if __name__ == "__main__":
    _validate()
