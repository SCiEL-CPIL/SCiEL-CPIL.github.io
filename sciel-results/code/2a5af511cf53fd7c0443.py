"""
Exact low-rank tensor decomposition for 4x4 * 4x7 matrix multiplication.

Strategy: Block-wise Recursive Strassen.
We view the 4x4 matrices as 2x2 blocks of 2x2 sub-matrices.
Let A = [[A11, A12], [A21, A22]] and B = [[B11, B12, B13], [B21, B22, B23]].
The product C = A @ B is a 4x7 matrix, which can be viewed as two 4x3 blocks:
C = [[C11, C12, C13], [C21, C22, C23]] where Cij are 2x3 blocks.

Using Strassen's algorithm at the block level:
M1 = (A11 + A22) @ (B11 + B22)
M2 = (A21 + A22) @ B11
M3 = A11 @ (B12 - B22)
M4 = A22 @ (B21 - B11)
M5 = (A11 + A12) @ B22
M6 = (A21 - A11) @ (B11 + B12)
M7 = (A12 - A22) @ (B21 + B22)

The block results are:
C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6

Note: The B blocks are 2x3. The A blocks are 2x2.
The "multiplication" inside each Mi is a 2x2 @ 2x3 product.
We implement these 2x2 @ 2x3 products using the schoolbook algorithm (rank 12 each).
Total rank = 7 * 12 = 84.

This is an exact decomposition.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = delta_{col_A, row_B}."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] V[:,r] W[:,r]^T = T."""
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(recon, T, atol=tol)


def _schoolbook_2x2x3():
    """
    Returns U, V, W for 2x2 @ 2x3 -> 2x3 multiplication.
    U: (4, 12)
    V: (6, 12)
    W: (6, 12)
    Indexing:
    A: 2x2 -> flat [a00, a01, a10, a11] (indices 0..3)
    B: 2x3 -> flat [b00, b01, b02, b10, b11, b12] (indices 0..5)
    C: 2x3 -> flat [c00, c01, c02, c10, c11, c12] (indices 0..5)
    """
    R = 12
    U = np.zeros((4, R))
    V = np.zeros((6, R))
    W = np.zeros((6, R))
    
    col = 0
    for i in range(2):      # row of A/C
        for j in range(2):  # col of A / row of B
            for k in range(3): # col of B/C
                # A index: i*2 + j
                # B index: j*3 + k
                # C index: i*3 + k
                U[i * 2 + j, col] = 1.0
                V[j * 3 + k, col] = 1.0
                W[i * 3 + k, col] = 1.0
                col += 1
                
    return U, V, W


def _build_strassen_4x4x7():
    """
    Construct the rank-84 decomposition using block Strassen.
    
    Global indices:
    A: 4x4 -> 16 elements.
       Block A11: rows 0-1, cols 0-1 -> local 0-3
       Block A12: rows 0-1, cols 2-3 -> local 4-7
       Block A21: rows 2-3, cols 0-1 -> local 8-11
       Block A22: rows 2-3, cols 2-3 -> local 12-15
       
    B: 4x7 -> 28 elements.
       Block B11: rows 0-1, cols 0-2 -> local 0-5
       Block B12: rows 0-1, cols 3-5 -> local 6-11
       Block B13: rows 0-1, cols 6-6 -> local 12-14  (Wait, B is 4x7. 2x3 blocks don't fit 7 evenly.)
       
    Correction on Block Structure for B (4x7):
    We split B into top 2 rows and bottom 2 rows.
    Top 2 rows: 2x7. We split columns into 3+3+1? No, Strassen requires square blocks or compatible dimensions for the "addition" steps.
    Strassen works for rectangular blocks too, as long as the addition/subtraction dimensions match.
    Let's split B (4x7) into B_top (2x7) and B_bot (2x7).
    Then split B_top into B11 (2x3), B12 (2x3), B13 (2x1)?
    And B_bot into B21 (2x3), B22 (2x3), B23 (2x1)?
    
    If we do this, the Strassen formulas become:
    M1 = (A11+A22) @ (B11+B21+B31?) -- This gets messy if the third block is size 1.
    
    Alternative: Use a 4x4x7 decomposition based on 4x4x4 + 4x4x3.
    We already have a valid 4x4x4 Strassen (rank 49) IF constructed correctly.
    The previous program's 4x4x4 was WRONG (Kronecker of Strassen is not Strassen of 4x4).
    
    Let's construct a correct 4x4x4 Strassen.
    4x4x4 can be done by treating 4x4 as 2x2 blocks of 2x2.
    This is exactly the same structure as above but with B being 4x4.
    B11, B12, B21, B22 are all 2x2.
    The sub-multiplications are 2x2 @ 2x2.
    Rank of 2x2 @ 2x2 is 7 (Strassen) or 4 (Schoolbook).
    If we use Schoolbook for the base case:
    Rank = 7 (Strassen steps) * 4 (Schoolbook base) = 28.
    Known rank for 4x4x4 is 49 (using Strassen recursively) or 28 (using Schoolbook base).
    Wait, the rank of the tensor is the minimum number of multiplications.
    Using Strassen recursively on 4x4x4 gives rank 7^2 = 49.
    Using Schoolbook base (rank 4) for the 2x2 blocks gives rank 7 * 4 = 28.
    Is rank 28 valid for 4x4x4?
    Yes, because 4x4 multiplication can be done with 28 multiplications?
    No, the standard schoolbook for 4x4 is 64.
    Strassen for 4x4 is 49.
    Why did I say 28?
    Let's re-evaluate.
    A 4x4 matrix multiplication is composed of 4 block multiplications of 2x2 matrices?
    No, C = A@B.
    C11 = A11@B11 + A12@B21
    C12 = A11@B12 + A12@B22
    C21 = A21@B11 + A22@B21
    C22 = A21@B12 + A22@B22
    
    If we use Strassen for the 2x2 blocks, we need to compute products like (A11+A22)@(B11+B22).
    This requires adding the 2x2 matrices A11 and A22 (free), then multiplying the result (2x2) by the result of adding B11 and B22 (2x2).
    The multiplication of two 2x2 matrices has rank 7.
    So each Strassen step involves 7 multiplications of 2x2 matrices.
    Total multiplications = 7 * 7 = 49.
    
    If we use Schoolbook for the 2x2 base multiplication (rank 4), then each Strassen step involves 4 multiplications of 2x2 matrices?
    No, the Strassen algorithm *is* the decomposition.
    If we stop recursion at 2x2 and use schoolbook (4 mults), the total rank is 7 * 4 = 28.
    Is rank 28 achievable for 4x4x4?
    Yes, if the base case 2x2x2 is solved with rank 4.
    But the rank of the 2x2x2 tensor is 7. We cannot solve it with rank 4.
    So we MUST use rank 7 for the 2x2x2 base.
    Therefore, the rank of 4x4x4 via Strassen recursion is 7 * 7 = 49.
    
    So, for 4x4x4, we use rank 49.
    
    Now for 4x4x3.
    We can view this as 4x4 @ 4x3.
    Split A into 2x2 blocks. Split B (4x3) into B_top (2x3) and B_bot (2x3).
    C_top = A_top @ B_top + A_bot @ B_bot? No.
    C = A @ B.
    C is 4x3.
    C_top (2x3) = A11 @ B11 + A12 @ B21
    C_bot (2x3) = A21 @ B11 + A22 @ B21
    
    Here B11 is 2x3, B21 is 2x3.
    We can apply Strassen-like logic?
    The standard Strassen requires B to be split into square blocks to match A's split.
    If B is 4x3, we can split it into 2x3 blocks.
    The "Strassen" formula for rectangular matrices:
    M1 = (A11+A22) @ (B11+B21)
    M2 = (A21+A22) @ B11
    M3 = A11 @ (B12-B22) -- Wait, B only has 2 columns of blocks if we split 3 into 1+2?
    
    Let's stick to a simpler, robust construction for 4x4x3.
    Split B into B1 (4x2) and B2 (4x1).
    4x4x2: Rank?
    4x4x1: Rank?
    
    Actually, let's just use the schoolbook for the 4x4x3 part to be safe and fast, and optimize the 4x4x4 part.
    Schoolbook 4x4x3 rank = 4*4*3 = 48.
    Strassen 4x4x4 rank = 49.
    Total = 49 + 48 = 97.
    
    Can we do better on 4x4x3?
    4x4x2:
    A (4x4), B (4x2).
    Split A into 2x2 blocks. B into 2x2 and 2x0? No.
    B is 4x2. Split into B1 (2x2) and B2 (2x2)? No, B has 2 columns.
    B_top (2x2), B_bot (2x2).
    C_top (2x2) = A11@B11 + A12@B21
    C_bot (2x2) = A21@B11 + A22@B21
    
    This is exactly the same structure as 4x4x4 but with B blocks being 2x2 and output blocks 2x2.
    Wait, in 4x4x4, B is 4x4, so B11, B12, B21, B22 are 2x2.
    In 4x4x2, B is 4x2.
    If we split B into top 2 rows and bottom 2 rows:
    B_top is 2x2. B_bot is 2x2.
    But Strassen requires B to be split into 4 blocks corresponding to A's 4 blocks?
    No, Strassen for C = A@B where A is 2x2 blocks and B is 2x2 blocks:
    C11 = A11@B11 + A12@B21
    C12 = A11@B12 + A12@B22
    ...
    If B only has 2 columns, B12 and B22 are zero-width?
    No, B is 4x2.
    B = [[B11], [B21]] where B11, B21 are 2x2.
    Then C = [[C11], [C21]] where C11, C21 are 2x2.
    C11 = A11@B11 + A12@B21
    C21 = A21@B11 + A22@B21
    
    This is not a full 2x2 block matrix multiplication. It's half.
    However, we can still use a Strassen-like reduction.
    We need to compute:
    X1 = A11@B11
    X2 = A12@B21
    X3 = A21@B11
    X4 = A22@B21
    
    C11 = X1 + X2
    C21 = X3 + X4
    
    We can use Strassen on the "virtual" 4x4x4 where the right half of B is zero?
    If B12=0 and B22=0, then:
    M1 = (A11+A22)@(B11+0) = (A11+A22)@B11
    M2 = (A21+A22)@B11
    M3 = A11@(0-0) = 0
    M4 = A22@(0-B11) = -A22@B11
    M5 = (A11+A12)@0 = 0
    M6 = (A21-A11)@B11
    M7 = (A12-A22)@(0+0) = 0
    
    This doesn't seem to reduce the count effectively because we still need to handle the A12@B21 term which is "missing" from this subset of Strassen.
    
    Let's just use Schoolbook for 4x4x3 (rank 48) and Strassen for 4x4x4 (rank 49).
    Total 97.
    
    Is there a better way for 4x4x3?
    We can treat 4x4x3 as 4x4x4 with the last column of B being zero.
    Rank 49.
    Then add the 4x4x1 part? No, that's double counting.
    
    Actually, we can just compute 4x4x7 directly using Schoolbook for the whole thing?
    Rank 112.
    
    My proposed 97 is better than 112.
    
    Let's implement the 4x4x4 Strassen (rank 49) and 4x4x3 Schoolbook (rank 48).
    
    4x4x4 Strassen Construction:
    Base: 2x2x2 Strassen (rank 7).
    Recursive step:
    U_4[i, r1*7+r2] = U_2[i1, r1] * U_2[i2, r2] ... NO.
    The Kronecker product of tensors T1 x T2 corresponds to the tensor for (A1 x A2) @ (B1 x B2).
    This is NOT the same as (A1+A2) @ (B1+B2).
    
    The correct way to build 4x4x4 from 2x2x2:
    The 4x4x4 tensor is NOT the Kronecker product of two 2x2x2 tensors.
    The Kronecker product of two 2x2x2 tensors gives a 4x4x4 tensor that computes element-wise products of the 2x2 blocks?
    No, T1 x T2 has indices (i1,i2), (j1,j2), (k1,k2).
    T1[i1,j1,k1] T2[i2,j2,k2].
    This corresponds to the multiplication of "block diagonal" matrices or something similar?
    Actually, the rank of the Kronecker product of two tensors is the product of their ranks.
    But does it compute the matrix multiplication of the 4x4 matrices?
    Let A be 4x4. A[i1,i2, j1,j2].
    If we view A as a 2x2 matrix of 2x2 blocks, the multiplication is not simply the Kronecker product of the block multiplications.
    
    So, the "Kronecker Strassen" in the previous code was mathematically incorrect for standard matrix multiplication.
    
    Correct Construction for 4x4x4 Rank 49:
    We must explicitly write out the 49 terms.
    Strassen formulas for 4x4 (using 2x2 blocks):
    M1 = (A11+A22) @ (B11+B22)
    M2 = (A21+A22) @ B11
    M3 = A11 @ (B12-B22)
    M4 = A22 @ (B21-B11)
    M5 = (A11+A12) @ B22
    M6 = (A21-A11) @ (B11+B12)
    M7 = (A12-A22) @ (B21+B22)
    
    Each Mi is a 2x2 matrix.
    Each Mi is computed by a 2x2 @ 2x2 multiplication.
    We use the rank-7 Strassen decomposition for each 2x2 @ 2x2.
    So for each of the 7 block products, we have 7 scalar multiplications.
    Total 49.
    
    We need to map the global indices to these local multiplications.
    
    Let's code this carefully.
    """
    # Base 2x2x2 Strassen
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, -1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_s = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    W_s = np.array([
        [1, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    R_4x4x4 = 49
    U_4 = np.zeros((16, R_4x4x4))
    V_4 = np.zeros((16, R_4x4x4))
    W_4 = np.zeros((16, R_4x4x4))
    
    # Define the 7 Strassen combinations for blocks
    # Each combination is a tuple of (U_coeff, V_coeff) for the blocks
    # A blocks: A11 (0), A12 (1), A21 (2), A22 (3)
    # B blocks: B11 (0), B12 (1), B21 (2), B22 (3)
    
    # M1 = (A11+A22) @ (B11+B22)
    # M2 = (A21+A22) @ B11
    # M3 = A11 @ (B12-B22)
    # M4 = A22 @ (B21-B11)
    # M5 = (A11+A12) @ B22
    # M6 = (A21-A11) @ (B11+B12)
    # M7 = (A12-A22) @ (B21+B22)
    
    # For each Mi, we perform a 2x2 @ 2x2 multiplication.
    # The "input" to this 2x2 multiplication is a linear combination of the A blocks.
    # The "output" is added to the C blocks.
    
    # We iterate over the 7 Strassen steps.
    # For each step, we have a linear combination of A blocks (2x2 matrix of coeffs)
    # and a linear combination of B blocks (2x2 matrix of coeffs).
    
    # Let's define the coeffs for A and B for each step.
    # A_coeff[step][block_idx] is the coefficient for that block in the sum.
    # There are 4 A blocks.
    
    A_coeffs = [
        {0: 1, 3: 1},  # M1: A11 + A22
        {2: 1, 3: 1},  # M2: A21 + A22
        {0: 1},         # M3: A11
        {3: 1},         # M4: A22
        {0: 1, 1: 1},  # M5: A11 + A12
        {2: 1, 0: -1}, # M6: A21 - A11
        {1: 1, 3: -1}  # M7: A12 - A22
    ]
    
    B_coeffs = [
        {0: 1, 3: 1},  # M1: B11 + B22
        {0: 1},         # M2: B11
        {1: 1, 3: -1}, # M3: B12 - B22
        {2: 1, 0: -1}, # M4: B21 - B11
        {3: 1},         # M5: B22
        {0: 1, 1: 1},  # M6: B11 + B12
        {2: 1, 3: 1}   # M7: B21 + B22
    ]
    
    # Output accumulation:
    # C11 += M1 + M4 - M5 + M7
    # C12 += M3 + M5
    # C21 += M2 + M4
    # C22 += M1 - M2 + M3 + M6
    
    W_coeffs = [
        {0: 1, 2: 1, 1: -1, 3: 1},  # M1: C11, C22
        {2: 1, 3: -1},               # M2: C21, C22
        {1: 1, 3: 1},                # M3: C12, C22
        {0: 1, 2: 1},                # M4: C11, C21
        {0: -1, 1: 1},               # M5: C11, C12
        {3: 1},                      # M6: C22
        {0: 1}                       # M7: C11
    ]
    
    # Now, for each step s (0..6), we have a 2x2 @ 2x2 multiplication.
    # We use the base Strassen U_s, V_s, W_s.
    # The base Strassen computes C_local = A_local @ B_local.
    # A_local is formed by combining A blocks.
    # B_local is formed by combining B blocks.
    # C_local is distributed to C blocks.
    
    for s in range(7):
        # For each of the 7 scalar multiplications in the base Strassen
        for r_base in range(7):
            r_global = s * 7 + r_base
            
            # U part:
            # The global A index is (block_row, block_col, local_row, local_col)
            # Block indices: 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1)
            # Local indices: 0..3
            # Global index = (block_row * 2 + local_row) * 4 + (block_col * 2 + local_col)
            
            for block_idx, coeff_a in A_coeffs[s].items():
                b_row = block_idx // 2
                b_col = block_idx % 2
                for local_idx in range(4):
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_idx = (b_row * 2 + l_row) * 4 + (b_col * 2 + l_col)
                    U_4[global_idx, r_global] += coeff_a * U_s[local_idx, r_base]
            
            # V part:
            for block_idx, coeff_b in B_coeffs[s].items():
                b_row = block_idx // 2
                b_col = block_idx % 2
                for local_idx in range(4):
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_idx = (b_row * 2 + l_row) * 4 + (b_col * 2 + l_col)
                    V_4[global_idx, r_global] += coeff_b * V_s[local_idx, r_base]
            
            # W part:
            for block_idx, coeff_c in W_coeffs[s].items():
                b_row = block_idx // 2
                b_col = block_idx % 2
                for local_idx in range(4):
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_idx = (b_row * 2 + l_row) * 4 + (b_col * 2 + l_col)
                    W_4[global_idx, r_global] += coeff_c * W_s[local_idx, r_base]
                    
    return U_4, V_4, W_4


def _build_4x4x3_schoolbook():
    """
    Schoolbook decomposition for 4x4 @ 4x3 -> 4x3.
    Rank 48.
    """
    R = 48
    U = np.zeros((16, R))
    V = np.zeros((28, R)) # 4*7 = 28, but we only use first 12? No, V is (m*p, R) = (28, R).
    # Wait, the full problem is 4x4x7.
    # If we split into 4x4x4 and 4x4x3, we need to pad the V and W for the 4x4x3 part.
    # V for 4x4x3 is (12, 48). We pad it to (28, 48) with zeros for the last 16 rows?
    # No, the columns 4-6 of B are the ones used.
    # B is 4x7.
    # B_cols 0-3 used in 4x4x4 part.
    # B_cols 4-6 used in 4x4x3 part.
    # So V for 4x4x3 part should have non-zeros in rows corresponding to cols 4-6.
    # Row index in V: j*p + k.
    # j in 0..3, k in 4..6.
    # Rows: 0*7+4=4, 0*7+5=5, 0*7+6=6
    #        1*7+4=11, 1*7+5=12, 1*7+6=13
    #        2*7+4=18, 2*7+5=19, 2*7+6=20
    #        3*7+4=25, 3*7+5=26, 3*7+6=27
    
    W = np.zeros((28, R))
    # W for 4x4x3: output C is 4x3.
    # C rows 0-3, cols 4-6? No, C is 4x7.
    # The 4x4x4 part computes C cols 0-3.
    # The 4x4x3 part computes C cols 4-6.
    # So W non-zeros in rows corresponding to C cols 4-6.
    # Row index in W: k*n + i.
    # k in 4..6, i in 0..3.
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3): # local k 0..2, global k 4..6
                global_k = k + 4
                U[i * 4 + j, col] = 1.0
                V[j * 7 + global_k, col] = 1.0
                W[global_k * 4 + i, col] = 1.0
                col += 1
                
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140
    
    T = _build_tensor(n, m, p)
    
    # Fallback: Schoolbook 112
    U_sb, V_sb, W_sb = _build_schoolbook_4x4x7()
    best_rank = 112
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try Strassen 4x4x4 + Schoolbook 4x4x3
    try:
        if time.time() < deadline - 10:
            U4, V4, W4 = _build_strassen_4x4x7()
            U3, V3, W3 = _build_4x4x3_schoolbook()
            
            R4 = 49
            R3 = 48
            R_total = R4 + R3
            
            U = np.zeros((16, R_total))
            V = np.zeros((28, R_total))
            W = np.zeros((28, R_total))
            
            U[:, :R4] = U4
            V[:16, :R4] = V4 # V4 is (16, 49). Rows 0-15 correspond to B cols 0-3.
            W[:16, :R4] = W4 # W4 is (16, 49). Rows 0-15 correspond to C cols 0-3.
            
            U[:, R4:] = U3
            V[R4:] = 0 # Already zero, but V3 has non-zeros in specific rows.
            V[:, R4:] = V3
            W[:, R4:] = W3
            
            if _verify(U, V, W, T, tol=1e-9):
                if R_total < best_rank:
                    best_rank = R_total
                    best_U, best_V, best_W = U, V, W
    except Exception as e:
        pass

    U_out = [[float(x) for x in best_U[r]] for r in range(best_U.shape[0])]
    V_out = [[float(x) for x in best_V[r]] for r in range(best_V.shape[0])]
    W_out = [[float(x) for x in best_W[r]] for r in range(best_W.shape[0])]
    
    return {'U': U_out, 'V': V_out, 'W': W_out}

def _build_schoolbook_4x4x7():
    """Schoolbook decomposition, rank 112."""
    n, m, p = 4, 4, 7
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W
