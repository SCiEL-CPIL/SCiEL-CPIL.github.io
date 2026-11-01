import time
import numpy as np

def construct():
    """
    Rank-minimal tensor decomposition for multiplying a 2x2 matrix by a 2x8 matrix.
    
    C = AB where A is 2x2, B is 2x8, C is 2x8.
    Convention: C_idx = k*n + i, A_idx = i*m + j, B_idx = j*p + k.
    
    Strategy: Use Strassen-style decomposition.
    A = 2x2, B = 2x8 (8 columns).
    
    We can group the 8 columns of B into 4 pairs. For each pair of columns,
    we perform a 2x2 by 2x2 multiplication using Strassen (rank 7 instead of 8).
    Total rank = 4 * 7 = 28.
    
    Alternatively, we can look for a better global decomposition.
    
    Let's think more carefully. The tensor is:
    T(i, j, k) where i is row of A (0,1), j is col of A / row of B (0,1), k is col of B (0..7)
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    This is equivalent to 8 independent matrix-vector products: for each k, C[:, k] = A * B[:, k].
    The rank of a single 2x2 matrix by 2x1 vector multiplication is 4 (schoolbook).
    But can we do better?
    
    Actually, for 2x2 matrix times 2x1 vector, the tensor has entries T(i,j,k) = delta_{j,k} 
    where i is output row, j is input to A column, k is which vector component.
    Wait, let me re-index.
    
    C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    
    The tensor T is in the space of (output i) x (A input j) x (B input k).
    T(i,j,k) = 1 if this term appears in the sum, else 0.
    Specifically, C[i] = sum_j A[i,j] * (sum_k B[j,k])... no wait.
    
    For fixed k (column of B), C[:, k] = A * B[:, k].
    C[i, k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has indices (i, j, k) where i in {0,1}, j in {0,1}, k in {0,1,2,3,4,5,6,7}.
    T(i,j,k) = 1 (this term appears).
    
    Actually, the matrix multiplication tensor for A (m x n) * B (n x p) = C (m x p) has
    T(a, b, c) where a indexes A, b indexes B, c indexes C.
    T(a, b, c) = 1 if the product of A_element and B_element contributes to C_element.
    
    Here A is 2x2, B is 2x8.
    A has 4 entries: A[i,j] for i,j in {0,1}
    B has 16 entries: B[j,k] for j in {0,1}, k in {0..7}
    C has 16 entries: C[i,k] for i in {0,1}, k in {0..7}
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T is in R^{4} x R^{16} x R^{16}.
    T(A_idx, B_idx, C_idx) = 1 if A[idx_A] * B[idx_B] contributes to C[idx_C].
    
    A_idx = i*2 + j
    B_idx = j*8 + k
    C_idx = k*2 + i
    
    T(i*2+j, j*8+k, k*2+i) = 1
    
    This is a 4 x 16 x 16 tensor.
    
    The schoolbook rank is the number of non-zero entries = 8 (for each i, j, k combination where j matches).
    Wait, for each (i, j, k) triple, T is 1. There are 2*2*8 = 32 such triples.
    So schoolbook rank = 32.
    
    Can we do better? 
    
    Key insight: For each fixed k, we're computing C[:,k] = A * B[:,k].
    This is a 2x2 matrix times a 2x1 vector.
    The rank of the 2x2x2 tensor (for matrix-vector) is actually 4, not reducible further
    for exact computation over reals in general position... 
    
    Wait, but we have 8 such independent problems. The total rank is at least the sum of
    ranks if they're independent. But they share the same A matrix!
    
    C[:, k] for k=0..7 all use the same A.
    So we can share the linear combinations of A across all 8 columns.
    
    This is like computing A * B where B is 2x8.
    If we precompute some linear combinations of rows of A, we can reuse them.
    
    Strassen-style: For 2x2 * 2x2, we use 7 multiplications instead of 8.
    For 2x2 * 2x8, we can split B into 4 blocks of 2x2 and use Strassen on each.
    But the 7 intermediate products depend on A and the corresponding B block.
    
    Actually, the 7 intermediate values are:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11) B00
    M3 = A00 (B10 - B11)
    M4 = A11 (B01 - B00)
    M5 = (A00 + A01) B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    Each of these is a product of a linear combo of A entries and a linear combo of B entries.
    For B being 2x8, we have 4 such 2x2 blocks.
    
    For each block, we need 7 products. But the A-side linear combinations can be shared!
    The A-side combinations are:
    A00, A11, A10+A11, A00+A11, A00+A01, A10-A00, A01-A11, A11, A00, A10, A11, A01, A11
    Unique A combinations: A00, A11, A10+A11, A00+A11, A00+A01, A10-A00, A01-A11
    
    That's 7 unique A-side linear combinations.
    
    For each of the 4 B-blocks, we need 7 products.
    Total products = 4 * 7 = 28.
    
    So rank = 28.
    
    Let me construct this explicitly.
    
    B is 2x8. Split into blocks: B_block_s = B[:, 2s:2s+2] for s = 0,1,2,3.
    
    For each block s, the 7 products are:
    P1_s = (A00 + A11) * (B[0,2s] + B[1,2s+1])
    P2_s = (A10 + A11) * B[0,2s]
    P3_s = A00 * (B[1,2s] - B[1,2s+1])
    P4_s = A11 * (B[0,2s+1] - B[0,2s])
    P5_s = (A00 + A01) * B[1,2s+1]
    P6_s = (A10 - A00) * (B[0,2s] + B[0,2s+1])
    P7_s = (A01 - A11) * (B[1,2s] + B[1,2s+1])
    
    Then the output for block s:
    C[0,2s] = P1_s - P4_s - P5_s + P7_s
    C[1,2s] = P3_s + P5_s
    C[0,2s+1] = P2_s + P4_s
    C[1,2s+1] = P1_s - P2_s - P3_s + P6_s
    
    Let me verify this. Standard Strassen:
    C00 = M1 - M4 - M5 + M7
    C01 = M2 + M4
    C10 = M3 + M5
    C11 = M1 - M2 - M3 + M6
    
    Where:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    Yes, that's correct.
    
    So for our problem, for each block s (columns 2s and 2s+1 of B):
    B_block_s = [[B[0,2s], B[0,2s+1]], [B[1,2s], B[1,2s+1]]]
    
    The 7 products for block s:
    M1_s = (A00+A11) * (B[0,2s] + B[1,2s+1])
    M2_s = (A10+A11) * B[0,2s]
    M3_s = A00 * (B[1,2s] - B[1,2s+1])
    M4_s = A11 * (B[0,2s+1] - B[0,2s])
    M5_s = (A00+A01) * B[1,2s+1]
    M6_s = (A10-A00) * (B[0,2s] + B[0,2s+1])
    M7_s = (A01-A11) * (B[1,2s] + B[1,2s+1])
    
    Output:
    C[0,2s] = M1_s - M4_s - M5_s + M7_s
    C[0,2s+1] = M2_s + M4_s
    C[1,2s] = M3_s + M5_s
    C[1,2s+1] = M1_s - M2_s - M3_s + M6_s
    
    Now, in the tensor decomposition, each product M_r_s corresponds to one rank-1 term.
    The U matrix gives the coefficients for A, V for B, W for C.
    
    A indices: A[i,j] -> i*2+j
    A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    
    B indices: B[j,k] -> j*8+k
    B[0,2s] -> 0*8+2s = 2s
    B[0,2s+1] -> 0*8+2s+1 = 2s+1
    B[1,2s] -> 1*8+2s = 8+2s
    B[1,2s+1] -> 1*8+2s+1 = 9+2s
    
    C indices: C[i,k] -> k*2+i
    C[0,2s] -> 2s*2+0 = 4s
    C[0,2s+1] -> (2s+1)*2+0 = 4s+2
    C[1,2s] -> 2s*2+1 = 4s+1
    C[1,2s+1] -> (2s+1)*2+1 = 4s+3
    
    Total rank = 4 blocks * 7 products = 28.
    
    Let me build the U, V, W matrices.
    """
    n, m, p = 2, 2, 8
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (16, 28)
    
    col = 0
    for s in range(4):
        # Block s: columns 2s, 2s+1 of B
        B0_2s = 0 * 8 + 2 * s        # B[0, 2s]
        B0_2s1 = 0 * 8 + 2 * s + 1   # B[0, 2s+1]
        B1_2s = 1 * 8 + 2 * s        # B[1, 2s]
        B1_2s1 = 1 * 8 + 2 * s + 1   # B[1, 2s+1]
        
        C0_2s = 2 * s * 2 + 0        # C[0, 2s]
        C0_2s1 = (2 * s + 1) * 2 + 0 # C[0, 2s+1]
        C1_2s = 2 * s * 2 + 1        # C[1, 2s]
        C1_2s1 = (2 * s + 1) * 2 + 1 # C[1, 2s+1]
        
        # M1 = (A00+A11)(B0_2s + B1_2s1)
        U[0][col] += 1  # A00
        U[3][col] += 1  # A11
        V[B0_2s][col] += 1
        V[B1_2s1][col] += 1
        W[C0_2s][col] += 1
        W[C1_2s1][col] += 1
        col += 1
        
        # M2 = (A10+A11) * B0_2s
        U[2][col] += 1  # A10
        U[3][col] += 1  # A11
        V[B0_2s][col] += 1
        W[C0_2s1][col] += 1
        W[C1_2s1][col] += -1
        col += 1
        
        # M3 = A00 * (B1_2s - B1_2s1)
        U[0][col] += 1  # A00
        V[B1_2s][col] += 1
        V[B1_2s1][col] += -1
        W[C1_2s][col] += 1
        W[C1_2s1][col] += -1
        col += 1
        
        # M4 = A11 * (B0_2s1 - B0_2s)
        U[3][col] += 1  # A11
        V[B0_2s1][col] += 1
        V[B0_2s][col] += -1
        W[C0_2s][col] += -1
        W[C0_2s1][col] += 1
        col += 1
        
        # M5 = (A00+A01) * B1_2s1
        U[0][col] += 1  # A00
        U[1][col] += 1  # A01
        V[B1_2s1][col] += 1
        W[C0_2s][col] += -1
        W[C1_2s][col] += 1
        col += 1
        
        # M6 = (A10-A00) * (B0_2s + B0_2s1)
        U[2][col] += 1  # A10
        U[0][col] += -1 # A00
        V[B0_2s][col] += 1
        V[B0_2s1][col] += 1
        W[C1_2s1][col] += 1
        col += 1
        
        # M7 = (A01-A11) * (B1_2s + B1_2s1)
        U[1][col] += 1  # A01
        U[3][col] += -1 # A11
        V[B1_2s][col] += 1
        V[B1_2s1][col] += 1
        W[C0_2s][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

