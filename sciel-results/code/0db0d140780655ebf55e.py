"""Matrix multiplication 4x4x4 with Strassen-style rank-49 decomposition."""
import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # We'll use a rank-49 construction based on Strassen's algorithm
    # applied to 2x2 blocks, where each 2x2 multiplication uses rank 7.
    # The 4x4 multiplication is decomposed into 4 blocks of 2x2 multiplications,
    # but with cross-terms handled via additional rank terms.
    
    # Actually, let me think more carefully. The standard approach:
    # C = A * B where A, B, C are 4x4.
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each Ci is a 2x2 matrix. Using Strassen for each 2x2 multiplication:
    # For M1*M2 (both 2x2):
    # M1 = [[a,b],[c,d]], M2 = [[e,f],[g,h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e), P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    # Result: [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    #
    # So each 2x2*2x2 needs 7 multiplications.
    # We have 4 blocks of 2x2 multiplications for C, but they share inputs.
    #
    # Let's count: 
    # C11 needs A11*B11 and A12*B21: 7+7 = 14 multiplications
    # C12 needs A11*B12 and A12*B22: 7+7 = 14 multiplications
    # C21 needs A21*B11 and A22*B21: 7+7 = 14 multiplications
    # C22 needs A21*B12 and A22*B22: 7+7 = 14 multiplications
    # Total: 56 multiplications, not 49.
    #
    # To get 49, we need a more clever sharing. Let me use the known result:
    # The rank of 4x4 matrix multiplication is 49 (or possibly lower, but 49 is known).
    #
    # Actually, let me just implement a known rank-49 decomposition.
    # I'll use a systematic construction.
    
    # Alternative: use the fact that we can represent this as a sum of rank-1 tensors.
    # Let me just implement the schoolbook for now as a fallback, then try to optimize.
    
    # For a proper rank-49, I'll use a precomputed construction.
    # The key insight: we can use the tensor decomposition from the literature.
    
    # Let me implement a constructive approach:
    # Use Strassen recursively with optimal sharing.
    
    # Actually, the simplest correct approach with rank < 64:
    # Let's use rank 56 (4 blocks * 14 multiplications) which is valid,
    # but we want to get to 49 if possible.
    
    # For now, let me implement rank 56 as a safe intermediate step,
    # then try to reduce to 49.
    
    # Actually, let me just do the straightforward rank-56 construction first.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 16 x 56
    V = [[0] * rank for _ in range(m * p)]  # 16 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Helper: add a Strassen 2x2*2x2 multiplication
    # A_block = rows [a0, a1] of A, cols [a2, a3] (0-indexed within the 2x2 block)
    # Actually, let me think in terms of global indices.
    
    # A is 4x4, indexed A[i,j] for i,j in 0..3
    # In the decomposition, U has rows indexed by (i,j) where i is row of A, j is col of A
    # U[i*m+j, r] = coefficient
    
    # Let me define a helper to add one 2x2 Strassen multiplication
    # where A_block has global row indices [r1, r2] and col indices [c1, c2]
    # and B_block has row indices [c1, c2] and col indices [k1, k2]
    # The result contributes to C at rows [r1, r2] and cols [k1, k2]
    
    def add_strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset):
        """
        Add a Strassen 2x2*2x2 decomposition.
        a_rows: [r1, r2] - rows of A block
        a_cols: [c1, c2] - cols of A block  
        b_rows: [c1, c2] - rows of B block (should match a_cols)
        b_cols: [k1, k2] - cols of B block
        c_rows: [r1, r2] - rows of C block
        c_cols: [k1, k2] - cols of C block
        offset: starting column index in U, V, W
        """
        r1, r2 = a_rows
        c1, c2 = a_cols
        k1, k2 = b_cols
        sr1, sr2 = c_rows
        sk1, sk2 = c_cols
        
        # Strassen variables for 2x2:
        # A = [[A00, A01], [A10, A11]] where A00=A[r1,c1], A01=A[r1,c2], A10=A[r2,c1], A11=A[r2,c2]
        # B = [[B00, B01], [B10, B11]] where B00=B[c1,k1], B01=B[c1,k2], B10=B[c2,k1], B11=B[c2,k2]
        
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # Result:
        # C00 = P1+P4-P5+P7
        # C01 = P3+P5
        # C10 = P2+P4
        # C11 = P1-P2+P3+P6
        
        # Each Pi corresponds to one rank-1 term (column in U,V,W)
        # For Pi = linear_comb(A) * linear_comb(B), we have:
        # U[row_a, col] = coeff of A[row_a] in linear_comb(A)
        # V[row_b, col] = coeff of B[row_b] in linear_comb(B)  
        # W[row_c, col] = coeff for C[row_c] from this Pi
        
        # P1: (A[r1,c1]+A[r2,c2]) * (B[c1,k1]+B[c2,k2])
        #   U[r1,c1][off] = 1, U[r2,c2][off] = 1
        #   V[c1,k1][off] = 1, V[c2,k2][off] = 1
        #   W contributes to C00 with +1, C11 with +1
        #   W[sr1,sk1][off] += 1, W[sr2,sk2][off] += 1
        
        col = offset
        # P1
        U[r1*m+c1][col] = 1
        U[r2*m+c2][col] = 1
        V[c1*p+k1][col] = 1
        V[c2*p+k2][col] = 1
        W[sk1*n+sr1][col] = 1   # C00
        W[sk2*n+sr2][col] = 1   # C11
        col += 1
        
        # P2: (A[r2,c1]+A[r2,c2]) * B[c1,k1]
        U[r2*m+c1][col] = 1
        U[r2*m+c2][col] = 1
        V[c1*p+k1][col] = 1
        W[sk1*n+sr2][col] = 1   # C10
        col += 1
        
        # P3: A[r1,c1] * (B[c1,k2]-B[c2,k2])
        U[r1*m+c1][col] = 1
        V[c1*p+k2][col] = 1
        V[c2*p+k2][col] = -1
        W[sk2*n+sr1][col] = 1   # C01
        col += 1
        
        # P4: A[r2,c2] * (B[c2,k1]-B[c1,k1])
        U[r2*m+c2][col] = 1
        V[c2*p+k1][col] = 1
        V[c1*p+k1][col] = -1
        W[sk1*n+sr1][col] = 1   # C00
        W[sk1*n+sr2][col] = 1   # C10
        col += 1
        
        # P5: (A[r1,c1]+A[r1,c2]) * B[c2,k2]
        U[r1*m+c1][col] = 1
        U[r1*m+c2][col] = 1
        V[c2*p+k2][col] = 1
        W[sk2*n+sr1][col] = 1   # C01
        W[sk1*n+sr1][col] = -1  # C00 (subtract P5)
        col += 1
        
        # P6: (A[r2,c1]-A[r1,c1]) * (B[c1,k1]+B[c1,k2])
        U[r2*m+c1][col] = 1
        U[r1*m+c1][col] = -1
        V[c1*p+k1][col] = 1
        V[c1*p+k2][col] = 1
        W[sk2*n+sr2][col] = 1   # C11
        col += 1
        
        # P7: (A[r1,c2]-A[r2,c2]) * (B[c2,k1]+B[c2,k2])
        U[r1*m+c2][col] = 1
        U[r2*m+c2][col] = -1
        V[c2*p+k1][col] = 1
        V[c2*p+k2][col] = 1
        W[sk1*n+sr1][col] = 1   # C00
        col += 1
        
        return col
    
    # Now build the 4x4 decomposition using 4 Strassen 2x2 multiplications
    # Block 1: C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # We need two Strassen multiplications for this block.
    
    # For C[0:2, 0:2]:
    #   Part 1: A[0:2,0:2] * B[0:2,0:2]
    add_strassen_2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    if time.time() > deadline - 20:
        # Return partial result
        pass
    
    #   Part 2: A[0:2,2:4] * B[2:4,0:2]
    add_strassen_2x2([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 7)
    
    # For C[0:2, 2:4]:
    #   Part 1: A[0:2,0:2] * B[0:2,2:4]
    add_strassen_2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 14)
    
    #   Part 2: A[0:2,2:4] * B[2:4,2:4]
    add_strassen_2x2([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], 21)
    
    # For C[2:4, 0:2]:
    #   Part 1: A[2:4,0:2] * B[0:2,0:2]
    add_strassen_2x2([2,3], [0,1], [0,1], [0,1], [2,3], [0,1], 28)
    
    #   Part 2: A[2:4,2:4] * B[2:4,0:2]
    add_strassen_2x2([2,3], [2,3], [2,3], [0,1], [2,3], [0,1], 35)
    
    # For C[2:4, 2:4]:
    #   Part 1: A[2:4,0:2] * B[0:2,2:4]
    add_strassen_2x2([2,3], [0,1], [0,1], [2,3], [2,3], [2,3], 42)
    
    #   Part 2: A[2:4,2:4] * B[2:4,2:4]
    add_strassen_2x2([2,3], [2,3], [2,3], [2,3], [2,3], [2,3], 49)
    
    # This gives rank 56. Let's return this.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
