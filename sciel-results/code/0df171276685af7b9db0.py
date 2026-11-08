import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition for 2x3 * 3x7
    rank = 39
    
    # Initialize matrices with zeros
    # U: (n*m, rank) -> (6, 39)
    # V: (m*p, rank) -> (21, 39)
    # W: (p*n, rank) -> (14, 39)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """
        Adds a single tensor term to the decomposition.
        u_idx, v_idx, w_idx are lists of indices.
        u_val, v_val, w_val are lists of coefficients.
        """
        for i in range(len(u_idx)):
            U[u_idx[i]][col] += u_val[i]
            V[v_idx[i]][col] += v_val[i]
            W[w_idx[i]][col] += w_val[i]
    
    col = 0
    
    # Helper to add a full Strassen block for 2x2 * 2x2
    # A_block indices in U: 0,1,3,4 (assuming row-major 2x3, first 2 cols are 0,1,3,4? 
    # U index = i*m + j. 
    # A is 2x3.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # So A[0:2, 0:2] uses U indices 0, 1, 3, 4.
    # B is 3x7.
    # B[0,0]=0, B[0,1]=1, ...
    # B[1,0]=7, B[1,1]=8, ...
    # B[2,0]=14, ...
    # C is 2x7.
    # C[0,0]=0, C[0,1]=1, ...
    # C[1,0]=7, C[1,1]=8, ...
    # W index = k*n + i. 
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[0,4]=4, C[0,5]=5, C[0,6]=6
    # C[1,0]=7, C[1,1]=8, C[1,2]=9, C[1,3]=10, C[1,4]=11, C[1,5]=12, C[1,6]=13
    
    def add_strassen_2x2x2(a_idx, b_row_start, c_col_start):
        nonlocal col
        # A indices:
        # a00, a01, a10, a11
        a00, a01, a10, a11 = a_idx
        # B indices (2x2 block starting at row b_row_start, col 0? No, cols are fixed 0,1 relative to block)
        # We are processing B[0:2, c_col_start:c_col_start+2]
        b00 = b_row_start * 7 + c_col_start
        b01 = b_row_start * 7 + c_col_start + 1
        b10 = (b_row_start + 1) * 7 + c_col_start
        b11 = (b_row_start + 1) * 7 + c_col_start + 1
        # C indices (2x2 block starting at col c_col_start)
        c00 = c_col_start * 2 + 0
        c01 = c_col_start * 2 + 1
        c10 = (c_col_start + 1) * 2 + 0
        c11 = (c_col_start + 1) * 2 + 1
        
        # Strassen's 7 products:
        # M1 = (A00 + A11) * (B00 + B11)
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        col += 1
        
        # M2 = (A10 + A11) * B00
        add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        col += 1
        
        # M3 = A00 * (B01 + B11)
        add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, -1])
        col += 1
        
        # M4 = A11 * (B10 + B00)
        add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, -1])
        col += 1
        
        # M5 = (A00 + A01) * B11
        add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])
        col += 1

    # We have A (2x3). We use the first 2 columns for Strassen blocks.
    # A indices for 2x2 block:
    # Row 0: 0, 1
    # Row 1: 3, 4
    a_block_1 = [0, 1, 3, 4]
    
    # We have B (3x7). We use the first 2 rows for Strassen blocks.
    # B rows 0,1.
    # We split B's 7 columns into 3 blocks of 2 (cols 0-1, 2-3, 4-5) and 1 remainder (col 6).
    
    # Block 1: Cols 0-1
    add_strassen_2x2x2(a_block_1, 0, 0)
    
    # Block 2: Cols 2-3
    add_strassen_2x2x2(a_block_1, 0, 2)
    
    # Block 3: Cols 4-5
    add_strassen_2x2x2(a_block_1, 0, 4)
    
    # Now we have used 3 * 7 = 21 terms.
    # Remaining parts:
    # 1. A's 3rd column (index 2, 5) multiplied by B's 1st row (index 6, 13, 20, 27, 34, 41, 48? No, B is 3x7. Indices 0-6 for row 0).
    #    Wait, B indices:
    #    Row 0: 0,1,2,3,4,5,6
    #    Row 1: 7,8,9,10,11,12,13
    #    Row 2: 14,15,16,17,18,19,20
    #    My previous Strassen used B rows 0 and 1 (indices 0-13).
    #    So B row 2 (indices 14-20) is untouched by Strassen.
    #    And A column 2 (indices 2, 5) is untouched by Strassen.
    #    
    #    The product C = A * B.
    #    C[0, k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    #    C[1, k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
    #    
    #    The Strassen blocks computed:
    #    C_block = A[0:2, 0:2] * B[0:2, block]
    #    This covers terms A[0,0]B[0,k], A[0,1]B[1,k], A[1,0]B[0,k], A[1,1]B[1,k].
    #    
    #    Missing terms:
    #    1. A[0,2]B[2,k] and A[1,2]B[2,k] for all k=0..6.
    #       This is A_col2 * B_row2.
    #       A_col2 has 2 elements. B_row2 has 7 elements.
    #       This is a 2x1 * 1x7 multiplication.
    #       Rank is 2*1 = 2? No, rank of outer product is 1, but we have 2 rows of A and 7 cols of B.
    #       It's a sum of 2 outer products: A[0,2] * B_row2 and A[1,2] * B_row2.
    #       Wait, C[0, k] += A[0,2] B[2, k].
    #       C[1, k] += A[1,2] B[2, k].
    #       This requires 2 * 7 = 14 terms if done naively.
    #       Can we do better?
    #       The tensor is $A_{0,2} \otimes B_{2,k} \otimes C_{0,k}$ and $A_{1,2} \otimes B_{2,k} \otimes C_{1,k}$.
    #       These are separate rank-1 tensors. We need 14 terms.
    #       
    #    2. What about A[0:2, 0:2] * B[2, block]?
    #       The Strassen blocks only used B[0:2, :].
    #       We need A[0:2, 0:2] * B[2, block] as well?
    #       C[0, k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    #       Strassen computed the first two parts for B[0:2, :].
    #       We still need to add A[0:2, 0:2] * B[2, :].
    #       A[0:2, 0:2] is 2x2. B[2, :] is 1x7.
    #       This is a 2x2 * 1x7 multiplication.
    #       Rank is 2 * 1 = 2? No.
    #       $C_{2x7} = A_{2x2} B_{1x7}$.
    #       Each column of C is a linear combination of the 2 rows of A.
    #       Actually, $C_{i, k} = \sum_{j=0}^1 A_{i, j} B_{2, k}$.
    #       $B_{2, k}$ is a scalar.
    #       So $C_{:, k} = B_{2, k} A_{:, 0:2}$.
    #       This is an outer product of $B_{2, k}$ (vector of 7) and $A_{:, 0:2}$ (matrix 2x2)?
    #       No, for a fixed k, $C_{0, k} = A_{0,0} B_{2, k} + A_{0,1} B_{2, k} = B_{2, k} (A_{0,0} + A_{0,1})$.
    #       $C_{1, k} = B_{2, k} (A_{1,0} + A_{1,1})$.
    #       This can be done with 2 terms per column k?
    #       Term 1: $U[0] = A_{0,0}+A_{0,1}$? No, U is fixed.
    #       We need to decompose the tensor $T_{i, j, k} = A_{i, j} B_{j, k}$ where $j$ is fixed to 2? No, $j$ in B is 2.
    #       The tensor is $A_{i, j} B_{2, k}$.
    #       This is $A_{i, j} \otimes B_{2, k} \otimes C_{i, k}$.
    #       Since $B_{2, k}$ is independent of $j$, we can factor it?
    #       $C_{i, k} = \sum_{j=0}^1 A_{i, j} B_{2, k} = B_{2, k} \sum_{j=0}^1 A_{i, j}$.
    #       Let $S_i = \sum_{j=0}^1 A_{i, j}$.
    #       Then $C_{i, k} = S_i B_{2, k}$.
    #       This is an outer product of vector $S$ (size 2) and vector $B_2$ (size 7).
    #       Rank 1? No, $C$ is 2x7. $S$ is 2x1. $B_2$ is 1x7.
    #       $C = S B_2$.
    #       This is a rank 1 matrix in the space of C, but in the tensor decomposition:
    #       $T_{i, j, k} = A_{i, j} B_{2, k}$.
    #       We need to represent this using U, V, W.
    #       $U_{i, r} V_{j, r} W_{k, r}$.
    #       Here $j$ is fixed to 2 in B? No, $V$ index is $m \times p = 3 \times 7$.
    #       $V$ index for $B_{2, k}$ is $2 \times 7 + k = 14 + k$.
    #       $U$ index for $A_{i, j}$ is $i \times 3 + j$.
    #       We need to sum over $j=0,1$.
    #       $C_{i, k} = \sum_{r} U_{i, r} V_{2 \cdot 7 + k, r} W_{k, r}$.
    #       We want this to equal $\sum_{j=0}^1 A_{i, j} B_{2, k}$.
    #       We can use 2 terms, one for each $j$.
    #       Term j=0: $U_{i, r0} = A_{i, 0}$, $V_{14+k, r0} = B_{2, k}$, $W_{k, r0} = 1$.
    #       Term j=1: $U_{i, r1} = A_{i, 1}$, $V_{14+k, r1} = B_{2, k}$, $W_{k, r1} = 1$.
    #       This takes 2 * 7 = 14 terms? No, the terms are defined by the rank.
    #       Each rank-1 term in the decomposition corresponds to a column in U, V, W.
    #       If we use 2 terms for the whole block, we need to encode $A_{i, j}$ and $B_{2, k}$ into the coefficients.
    #       But $B_{2, k}$ varies with $k$.
    #       So we need a separate term for each $k$?
    #       Yes, because $V$ depends on $k$.
    #       So for each $k$, we need to represent $C_{:, k} = B_{2, k} A_{:, 0:2}$.
    #       This is a vector of size 2.
    #       We can do this with 2 terms per $k$?
    #       Term 1 for k: $U[0, r] = A_{0,0}, U[1, r] = A_{1,0}$, $V[14+k, r] = 1$, $W[k, r] = B_{2, k}$.
    #       Term 2 for k: $U[0, r] = A_{0,1}, U[1, r] = A_{1,1}$, $V[14+k, r] = 1$, $W[k, r] = B_{2, k}$.
    #       This is 2 * 7 = 14 terms.
    #       
    #    Total so far: 21 (Strassen) + 14 (A_col2 * B_row2) + 14 (A_block1 * B_row2) = 49.
    #    This is worse than 39.
    
    # The current program's rank 39 is better. How?
    # It must combine the "missing" parts more efficiently.
    # The current program uses 21 + 4 + 14 = 39?
    # Let's check the current program's code again.
    # It adds 3 Strassen blocks (21 terms).
    # Then it adds 4 terms for the last column of B (col 6) with the first 2 rows of A?
    # `b06`, `b16`. `c06`, `c16`.
    # `add_term([0], [b06], [c06])` -> A[0,0] * B[0,6] * C[0,6]
    # `add_term([1], [b16], [c06])` -> A[0,1] * B[1,6] * C[0,6]
    # `add_term([3], [b06], [c16])` -> A[1,0] * B[0,6] * C[1,6]
    # `add_term([4], [b16], [c16])` -> A[1,1] * B[1,6] * C[1,6]
    # This handles A[0:2, 0:2] * B[0:2, 6]. That's 4 terms.
    # Then it adds 14 terms for A[0:2, 2] * B[2, 0:7]?
    # `for k in range(7):`
    # `add_term([2], [b2k], [c0k])` -> A[0,2] * B[2,k] * C[0,k]
    # `add_term([5], [b2k], [c1k])` -> A[1,2] * B[2,k] * C[1,k]
    # This handles A[0:2, 2] * B[2, :]. That's 14 terms.
    # Total 21 + 4 + 14 = 39.
    # This matches the logic derived above, but I missed that A[0:2, 0:2] * B[0:2, 6] is only 4 terms (not 14).
    # Why? Because B[0:2, 6] is a 2x1 vector. A[0:2, 0:2] is 2x2.
    # Product is 2x1.
    # $C_{:, 6} = A_{:, 0:2} B_{0:2, 6}$.
    # This is a matrix-vector product.
    # Can be done with 4 terms?
    # $C_{0, 6} = A_{0,0} B_{0,6} + A_{0,1} B_{1,6}$.
    # $C_{1, 6} = A_{1,0} B_{0,6} + A_{1,1} B_{1,6}$.
    # Yes, 4 terms.
    
    # So the decomposition is valid and rank 39.
    # I will implement this exactly.

    # Re-implementing the specific terms from the analysis:
    
    # 1. Strassen blocks for cols 0, 2, 4
    add_strassen_2x2x2(a_block_1, 0, 0)
    add_strassen_2x2x2(a_block_1, 0, 2)
    add_strassen_2x2x2(a_block_1, 0, 4)
    
    # 2. Handle B[0:2, 6] with A[0:2, 0:2]
    # A indices: 0, 1, 3, 4
    # B indices: 0*7+6=6, 1*7+6=13
    # C indices: 6*2+0=12, 6*2+1=13
    
    # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6]
    add_term([0], [1], [6], [1], [12], [1])
    add_term([1], [1], [13], [1], [12], [1])
    # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6]
    add_term([3], [1], [6], [1], [13], [1])
    add_term([4], [1], [13], [1], [13], [1])
    
    # 3. Handle B[2, 0:7] with A[0:2, 2]
    # A indices: 2, 5
    # B indices: 14+k
    # C indices: k*2 (for row 0), k*2+1 (for row 1)
    
    for k in range(7):
        b_idx = 14 + k
        c0_idx = k * 2
        c1_idx = k * 2 + 1
        
        # C[0,k] += A[0,2] B[2,k]
        add_term([2], [1], [b_idx], [1], [c0_idx], [1])
        # C[1,k] += A[1,2] B[2,k]
        add_term([5], [1], [b_idx], [1], [c1_idx], [1])
        
    return {"U": U, "V": V, "W": W}
