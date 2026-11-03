"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 32
    
    # The tensor T has dimensions (n*m) x (m*p) x (p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We want to find rank r such that T = sum_{c=1}^r U[:,c] * V[:,c] * W[:,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # Known lower bounds for 2x4x4 matrix multiplication:
    # The Strassen-like decomposition might give rank < 32
    # Let's try to find a decomposition with rank < 32
    
    # Approach: Use the fact that for small matrices, we can try various
    # algebraic decompositions. One known approach is to use block Strassen
    # or similar divide-and-conquer methods.
    
    # For 2x4 * 4x4, we can think of this as:
    # A (2x4) * B (4x4) = C (2x4)
    # Split B into two 4x2 blocks: B = [B1 | B2]
    # Then C = [A*B1 | A*B2]
    # Each A*B1 and A*B2 is a 2x4 * 4x2 multiplication
    
    # For 2x4 * 4x2, the schoolbook rank is 2*4*2 = 16
    # But we can use Strassen-like methods to reduce this.
    
    # Actually, let's try a direct approach: enumerate possible rank-1
    # decompositions and use optimization to find a sparse solution.
    
    # Alternative: Use known results. The minimal rank for multiplying
    # a 2x4 matrix by a 4x4 matrix is known to be at most 24 (using
    # Strassen's method adapted for rectangular matrices).
    
    # Let me try to construct a decomposition with rank 24.
    
    # Strassen's method for 2x2x2 uses 7 multiplications instead of 8.
    # For 2x4 * 4x4, we can partition as follows:
    # A (2x4) = [A11 A12] where A11, A12 are 2x2
    # B (4x4) = [[B11, B12], [B21, B22]] where each is 2x2
    # C (2x4) = [C11 C12] where C11, C12 are 2x2
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    
    # Using Strassen for each 2x2 multiplication:
    # For C11 = A11*B11 + A12*B21:
    # This is a sum of two 2x2 products. We can compute each with 7 mults,
    # total 14 mults.
    # Similarly for C12: 14 mults.
    # Total: 28 mults. Still better than 32.
    
    # Can we do better? Let's think about sharing computations.
    
    # Actually, let's try a different approach. Use the fact that we can
    # use bilinear algorithms. Let me try to find a decomposition
    # computationally.
    
    # For now, let's return the schoolbook decomposition if we can't find
    # a better one quickly.
    
    # Try to find a rank-28 decomposition using the block Strassen approach:
    
    def strassen_2x2(A, B):
        """Compute 2x2 * 2x2 using 7 multiplications."""
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
        e, f, g, h = B[0][0], B[0][1], B[1][0], B[1][1]
        
        m1 = (a + d) * (e + h)
        m2 = (c + d) * e
        m3 = a * (f - h)
        m4 = d * (g - e)
        m5 = (a + b) * h
        m6 = (c - a) * (e + f)
        m7 = (b - d) * (g + h)
        
        C = [[m1 + m4 - m5 + m7, m3 + m5],
             [m2 + m4, m1 - m2 + m3 + m6]]
        return C
    
    # Build the rank-28 decomposition using block Strassen
    # A (2x4) split into A11, A12 (each 2x2)
    # B (4x4) split into B11, B12, B21, B22 (each 2x2)
    # C11 = A11*B11 + A12*B21 (use Strassen for each, 7+7=14 mults)
    # C12 = A11*B12 + A12*B22 (use Strassen for each, 7+7=14 mults)
    # Total: 28 mults
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Indexing:
    # A[i,j] for i in 0..1, j in 0..3
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # B[i,j] for i in 0..3, j in 0..3
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C[i,j] for i in 0..1, j in 0..3
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4]
    
    # Strassen mults for C11 = A11*B11 + A12*B21:
    # Let's denote the 7 Strassen mults for X*Y as S1..S7(X,Y)
    # For A11*B11:
    #   S1 = (A11_00 + A11_11) * (B11_00 + B11_11)
    #   S2 = (A11_10 + A11_11) * B11_00
    #   S3 = A11_00 * (B11_01 - B11_11)
    #   S4 = A11_11 * (B11_10 - B11_00)
    #   S5 = (A11_00 + A11_01) * B11_11
    #   S6 = (A11_10 - A11_00) * (B11_00 + B11_01)
    #   S7 = (A11_01 - A11_11) * (B11_10 + B11_11)
    # C11_00 = S1 + S4 - S5 + S7
    # C11_01 = S3 + S5
    # C11_10 = S2 + S4
    # C11_11 = S1 - S2 + S3 + S6
    
    # Similarly for A12*B21.
    
    # This is getting complex. Let me just build it programmatically.
    
    def add_strassen_product(U, V, W, rank_offset, A_block, B_block, C_block):
        """Add Strassen decomposition for a 2x2 * 2x2 product.
        A_block: list of (i, j) indices in A for the 2x2 block
        B_block: list of (i, j) indices in B for the 2x2 block  
        C_block: list of (i, j) indices in C for the 2x2 block
        Returns new rank_offset."""
        # A_block[i][j] gives the (row, col) in A for the (i,j) element of the 2x2 block
        # Similarly for B_block and C_block
        
        # Strassen mults
        # S1 = (a00 + a11) * (b00 + b11)
        # S2 = (a10 + a11) * b00
        # S3 = a00 * (b01 - b11)
        # S4 = a11 * (b10 - b00)
        # S5 = (a00 + a01) * b11
        # S6 = (a10 - a00) * (b00 + b01)
        # S7 = (a01 - a11) * (b10 + b11)
        
        # C00 = S1 + S4 - S5 + S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S2 + S3 + S6
        
        a00 = A_block[0][0]
        a01 = A_block[0][1]
        a10 = A_block[1][0]
        a11 = A_block[1][1]
        
        b00 = B_block[0][0]
        b01 = B_block[0][1]
        b10 = B_block[1][0]
        b11 = B_block[1][1]
        
        c00 = C_block[0][0]
        c01 = C_block[0][1]
        c10 = C_block[1][0]
        c11 = C_block[1][1]
        
        # S1
        col = rank_offset
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        rank_offset += 1
        
        # S2
        col = rank_offset
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        rank_offset += 1
        
        # S3
        col = rank_offset
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        rank_offset += 1
        
        # S4
        col = rank_offset
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        rank_offset += 1
        
        # S5
        col = rank_offset
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        rank_offset += 1
        
        # S6
        col = rank_offset
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        rank_offset += 1
        
        # S7
        col = rank_offset
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        rank_offset += 1
        
        return rank_offset
    
    # Define blocks
    # A is 2x4: A[i,j] with i in 0..1, j in 0..3
    # A11 = rows 0-1, cols 0-1
    # A12 = rows 0-1, cols 2-3
    A11 = [[(0, 0), (0, 1)],
           [(1, 0), (1, 1)]]
    A12 = [[(0, 2), (0, 3)],
           [(1, 2), (1, 3)]]
    
    # B is 4x4: B[i,j] with i in 0..3, j in 0..3
    # B11 = rows 0-1, cols 0-1
    # B12 = rows 0-1, cols 2-3
    # B21 = rows 2-3, cols 0-1
    # B22 = rows 2-3, cols 2-3
    B11 = [[(0, 0), (0, 1)],
           [(1, 0), (1, 1)]]
    B12 = [[(0, 2), (0, 3)],
           [(1, 2), (1, 3)]]
    B21 = [[(2, 0), (2, 1)],
           [(3, 0), (3, 1)]]
    B22 = [[(2, 2), (2, 3)],
           [(3, 2), (3, 3)]]
    
    # C is 2x4: C[i,j] with i in 0..1, j in 0..3
    # C11 = rows 0-1, cols 0-1
    # C12 = rows 0-1, cols 2-3
    C11 = [[(0, 0), (0, 1)],
           [(1, 0), (1, 1)]]
    C12 = [[(0, 2), (0, 3)],
           [(1, 2), (1, 3)]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    
    # But we need to add the contributions from both products to the same C block.
    # So we need to add Strassen for A11*B11 with C_block = C11,
    # and Strassen for A12*B21 with C_block = C11,
    # etc.
    
    rank_offset = 0
    rank_offset = add_strassen_product(U, V, W, rank_offset, A11, B11, C11)
    rank_offset = add_strassen_product(U, V, W, rank_offset, A12, B21, C11)
    rank_offset = add_strassen_product(U, V, W, rank_offset, A11, B12, C12)
    rank_offset = add_strassen_product(U, V, W, rank_offset, A12, B22, C12)
    
    # rank_offset should be 28
    assert rank_offset == 28, f"Expected rank 28, got {rank_offset}"
    
    # Verify the decomposition
    # Check that for all i, j, k: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Check decomposition
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank_offset):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                err = abs(val - T[i*m+j, j*p+k, k*n+i])
                max_err = max(max_err, err)
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank = schoolbook_rank
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
