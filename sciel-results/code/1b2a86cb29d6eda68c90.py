import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 54
    # 3 blocks of Strassen (21) + 1 block Strassen (21) + 1 block Schoolbook (12)
    # Wait, let's re-verify the blocks.
    # Block 1: A(0:2, 0:2) * B(0:2, :) -> C(0:2, :)
    #   B(0:2, :) is 2x6. Split into 3 2x2 blocks.
    #   Each 2x2x2 Strassen is rank 7. Total 21.
    # Block 2: A(0:2, 2:4) * B(2:4, :) -> C(0:2, :)
    #   A(0:2, 2:4) is 2x2. B(2:4, :) is 2x6.
    #   Same as Block 1. Total 21.
    # Block 3: A(0:2, 4:5) * B(4:5, :) -> C(0:2, :)
    #   A(0:2, 4:5) is 2x1. B(4:5, :) is 1x6.
    #   Schoolbook: 2*1*6 = 12.
    # Block 4: A(2:3, :) * B(:, :)?
    #   A(2:3, 0:2) * B(0:2, :) -> C(2:3, :)
    #   A(2:3, 2:4) * B(2:4, :) -> C(2:3, :)
    #   A(2:3, 4:5) * B(4:5, :) -> C(2:3, :)
    
    # I missed the contributions from Row 2 of A!
    # My previous calculation only covered Rows 0-1 of A.
    # A has 3 rows. Rows 0,1,2.
    # B has 5 rows. Rows 0,1,2,3,4.
    
    # Let's partition A rows: 2, 1.
    # Let's partition B rows: 2, 2, 1.
    # Let's partition A cols: 2, 2, 1.
    # Let's partition B cols: 2, 2, 2.
    
    # We need to compute C(3x6) = A(3x5) * B(5x6).
    # C(i, k) = sum_j A(i, j) B(j, k).
    
    # We can sum the contributions from each block of A and B.
    # Let A be partitioned into A_1 (2x5) and A_2 (1x5).
    # Let B be partitioned into B_1 (2x6), B_2 (2x6), B_3 (1x6).
    
    # C = A_1 * (B_1 + B_2 + B_3) + A_2 * (B_1 + B_2 + B_3)
    # C = A_1 B_1 + A_1 B_2 + A_1 B_3 + A_2 B_1 + A_2 B_2 + A_2 B_3
    
    # Term 1: A_1 (2x5) * B_1 (2x6).
    #   Split A_1 cols: 2, 2, 1.
    #   B_1 rows: 2, 2, 1? No, B_1 is 2x6. Rows 0-1 of B.
    #   Wait, B_1 corresponds to B rows 0-1.
    #   A_1 corresponds to A rows 0-1.
    #   A_1 * B_1: (2x5) * (2x6).
    #   Inner dim 5. B_1 has 2 rows? No, B is 5x6.
    #   B_1 is rows 0-1 of B. So B_1 is 2x6.
    #   A_1 is rows 0-1 of A. So A_1 is 2x5.
    #   A_1 * B_1 is undefined (5 != 2).
    #   
    #   My block partitioning of B must match the columns of A.
    #   A cols: 0-4.
    #   B rows: 0-4.
    #   We must split A cols and B rows into matching sizes.
    #   A cols: 2, 2, 1.
    #   B rows: 2, 2, 1.
    #   
    #   So:
    #   Term 1: A_1 (2x2) * B_1 (2x6). (A rows 0-1, Cols 0-1; B rows 0-1, Cols 0-5)
    #   Term 2: A_1 (2x2) * B_2 (2x6). (A rows 0-1, Cols 2-3; B rows 2-3, Cols 0-5)
    #   Term 3: A_1 (2x1) * B_3 (1x6). (A rows 0-1, Cols 4; B rows 4, Cols 0-5)
    #   Term 4: A_2 (1x2) * B_1 (2x6). (A row 2, Cols 0-1; B rows 0-1, Cols 0-5)
    #   Term 5: A_2 (1x2) * B_2 (2x6). (A row 2, Cols 2-3; B rows 2-3, Cols 0-5)
    #   Term 6: A_2 (1x1) * B_3 (1x6). (A row 2, Col 4; B row 4, Cols 0-5)
    
    # Ranks:
    # Term 1: A(2x2) * B(2x6). B splits into 3 2x2 blocks. 3 * 7 = 21.
    # Term 2: A(2x2) * B(2x6). Same. 21.
    # Term 3: A(2x1) * B(1x6). Schoolbook 2*1*6 = 12.
    # Term 4: A(1x2) * B(2x6). B splits into 3 2x2 blocks.
    #   A is 1x2. B is 2x2.
    #   Can we use Strassen? Strassen is for 2x2 * 2x2.
    #   Here A is 1x2. It's a vector-matrix product?
    #   No, A_2 is 1x5. A_2(1x2) is a row vector.
    #   (1x2) * (2x6) = 1x6.
    #   This is just a dot product for each column of B.
    #   Rank: 1 * 2 * 6 = 12?
    #   Or can we optimize?
    #   $c_k = \sum_{j=0}^1 a_j b_{j,k}$.
    #   This is essentially 2 dot products of length 2?
    #   No, $a_0 b_{0,k} + a_1 b_{1,k}$.
    #   For each $k$, this is a bilinear form.
    #   Total rank for one $k$ is 2 (schoolbook).
    #   For 6 $k$'s, rank 12.
    #   Is there a better way?
    #   $a_0 (b_{0,0}..b_{0,5}) + a_1 (b_{1,0}..b_{1,5})$.
    #   This is $u \otimes v_1 + w \otimes v_2$.
    #   Rank 2 per output element?
    #   Actually, the whole term is rank 2?
    #   $C = A_{1 \times 2} B_{2 \times 6}$.
    #   $C_k = A_0 B_{0k} + A_1 B_{1k}$.
    #   $C = A_0 \cdot B_{0, \cdot} + A_1 \cdot B_{1, \cdot}$.
    #   This is a sum of 2 rank-1 tensors?
    #   $T = A_0 \otimes B_{0, \cdot} \otimes I + A_1 \otimes B_{1, \cdot} \otimes I$.
    #   Yes, rank 2 for the whole block?
    #   Wait, the output is a vector of length 6.
    #   The tensor is $T_{a, b, c}$.
    #   $a \in \{0,1\}$ (A index), $b \in \{0..5\}$ (B col), $c \in \{0..5\}$ (C col).
    #   $T_{a, b, c} = \delta_{a, \text{row of B?}}$.
    #   $C_c = \sum_{j=0}^1 A_j B_{j,c}$.
    #   Term $j=0$: $A_0 B_{0,c}$.
    #   Term $j=1$: $A_1 B_{1,c}$.
    #   Each term is rank 1 in the tensor sense?
    #   $u_0 = [1, 0]$ (A), $v_0 = [1, 0, 0, 0, 0, 0]$ (B row 0), $w_0 = I_6$?
    #   No, $w$ must pick the $c$ index.
    #   $C_c$ gets contribution from $B_{0,c}$.
    #   So $w_0$ should be 1 at index $c$?
    #   But $w$ is shared for all $c$ in the rank-1 term?
    #   A rank-1 tensor $u \otimes v \otimes w$ produces $C = (u \cdot A) (v \cdot B) w$.
    #   Here $v \cdot B$ is a scalar? No, $v$ is a vector, $B$ is a matrix.
    #   In the decomposition, $V$ is a vector. $v^T B$ is a row vector.
    #   So $C = (u^T A) (v^T B) w^T$?
    #   $u^T A$ is a scalar (if A is row vector) or vector?
    #   In our case, A is $1 \times 2$. $u^T A$ is scalar.
    #   $v^T B$ is $1 \times 6$ vector.
    #   $w$ is $1 \times 6$ vector?
    #   Product is outer product? No, element-wise?
    #   The tensor decomposition is $C_{ik} = \sum_r U_{ir} V_{kr} W_{?r}$.
    #   Wait, $C$ is $1 \times 6$. Index $c$.
    #   $C_c = \sum_r U_r (scalar) V_{r, c} W_{r, ?}$.
    #   If $W$ is a scalar 1, then $C_c = \sum_r U_r V_{r, c}$.
    #   This matches.
    #   So for Term 4:
    #   $r_1$: $U=1, V=B_{0, \cdot}, W=1$.
    #   $r_2$: $U=1, V=B_{1, \cdot}, W=1$.
    #   Rank 2.
    #   
    #   So Term 4 Rank: 2.
    #   Term 5 Rank: 2.
    #   Term 6: A(1x1) * B(1x6).
    #   $C_c = A_0 B_{0,c}$.
    #   Rank 1.
    #
    # Total Rank:
    # T1: 21
    # T2: 21
    # T3: 12
    # T4: 2
    # T5: 2
    # T6: 1
    # Sum: 59.
    
    # This is 59, which is better than 90.
    
    n, m, p = 3, 5, 6
    rank = 59
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(A_r1, A_r2, A_c1, A_c2, B_r1, B_r2, B_c1, B_c2, C_r1, C_r2, C_c1, C_c2):
        nonlocal col
        # A is 2x2, B is 2x2, C is 2x2
        # Strassen terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices in A: (r1, c1)=11, (r1, c2)=12, (r2, c1)=21, (r2, c2)=22
        # Indices in B: (r1, c1)=11, (r1, c2)=12, (r2, c1)=21, (r2, c2)=22
        # Indices in C: (r1, c1)=11, (r1, c2)=12, (r2, c1)=21, (r2, c2)=22
        
        # Define linear forms
        # u: index in A, v: index in B, w: index in C, coeff
        # For each M, we add 1 column to U, V, W.
        
        terms = []
        
        # M1
        terms.append( { 'A': [(A_r1, A_c1, 1), (A_r2, A_c2, 1)], 'B': [(B_r1, B_c1, 1), (B_r2, B_c2, 1)], 'C': [(C_r1, C_c1, 1), (C_r2, C_c2, -1)] } )
        # Wait, C11 = M1 + M4 - M5 + M7
        # So M1 contributes +1 to C11, 0 to others?
        # No, M1 is a scalar.
        # C11 gets M1. C12 gets 0. C21 gets 0. C22 gets 0?
        # No, C22 gets M1.
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        # Let's map M_k to C coefficients.
        # M1: C11=1, C22=1
        # M2: C21=1, C22=-1
        # M3: C12=1, C22=1
        # M4: C11=1, C21=1
        # M5: C11=-1, C12=1
        # M6: C22=1
        # M7: C11=1
        
        # Correct Mappings:
        # M1: C11=1, C22=1
        # M2: C21=1, C22=-1
        # M3: C12=1, C22=1
        # M4: C11=1, C21=1
        # M5: C11=-1, C12=1
        # M6: C22=1
        # M7: C11=1
        
        strassen_defs = [
            # (A_coeffs, B_coeffs, C_coeffs)
            # A_coeffs: list of (row, col, val)
            # C_coeffs: list of (row, col, val)
            
            # M1
            ( [(A_r1, A_c1, 1), (A_r2, A_c2, 1)],
              [(B_r1, B_c1, 1), (B_r2, B_c2, 1)],
              [(C_r1, C_c1, 1), (C_r2, C_c2, 1)] ),
            # M2
            ( [(A_r2, A_c1, 1), (A_r2, A_c2, 1)],
              [(B_r1, B_c1, 1)],
              [(C_r2, C_c1, 1), (C_r2, C_c2, -1)] ),
            # M3
            ( [(A_r1, A_c1, 1)],
              [(B_r1, B_c2, 1), (B_r2, B_c2, -1)],
              [(C_r1, C_c2, 1), (C_r2, C_c2, 1)] ),
            # M4
            ( [(A_r2, A_c2, 1)],
              [(B_r2, B_c1, 1), (B_r1, B_c1, -1)],
              [(C_r1, C_c1, 1), (C_r2, C_c1, 1)] ),
            # M5
            ( [(A_r1, A_c1, 1), (A_r1, A_c2, 1)],
              [(B_r2, B_c2, 1)],
              [(C_r1, C_c1, -1), (C_r1, C_c2, 1)] ),
            # M6
            ( [(A_r2, A_c1, 1), (A_r1, A_c1, -1)],
              [(B_r1, B_c1, 1), (B_r1, B_c2, 1)],
              [(C_r2, C_c2, 1)] ),
            # M7
            ( [(A_r1, A_c2, 1), (A_r2, A_c2, -1)],
              [(B_r2, B_c1, 1), (B_r2, B_c2, 1)],
              [(C_r1, C_c1, 1)] ),
        ]
        
        for A_def, B_def, C_def in strassen_defs:
            for r, c_idx, val in A_def:
                U[r * m + c_idx][col] += val
            for r, c_idx, val in B_def:
                V[r * p + c_idx][col] += val
            for r, c_idx, val in C_def:
                # W index: k*n + i. Here c_idx is k (col), r is i (row).
                W[c_idx * n + r][col] += val
            col += 1

    # Helper to add schoolbook term
    def add_schoolbook(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        # A_rows, A_cols, etc are lists or ranges
        # We iterate over all i in A_rows, j in A_cols, k in B_cols?
        # No, schoolbook for block (h x k) * (k x l) -> (h x l)
        # Rank h*k*l.
        # Each term is A[i, j] * B[j, k] -> C[i, k].
        
        # A_rows: list of row indices in A
        # A_cols: list of col indices in A
        # B_rows: list of row indices in B (must match A_cols count?)
        # B_cols: list of col indices in B
        
        # We assume A_cols and B_rows are the same set of "inner" dimensions.
        # Actually, A[i, j] and B[j, k]. j is the inner index.
        # So A_cols corresponds to B_rows.
        
        for j_idx, j in enumerate(A_cols):
            b_row = B_rows[j_idx]
            for i in A_rows:
                for k in B_cols:
                    U[i * m + j][col] = 1
                    V[b_row * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Block 1: A(0:2, 0:2) * B(0:2, 0:6)
    # A rows 0,1. Cols 0,1.
    # B rows 0,1. Cols 0,1, 2,3, 4,5.
    # B splits into 3 2x2 blocks.
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    add_strassen(0, 1, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5)
    
    # Block 2: A(0:2, 2:4) * B(2:4, 0:6)
    # A rows 0,1. Cols 2,3.
    # B rows 2,3. Cols 0-5.
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)
    add_strassen(0, 1, 2, 3, 2, 3, 4, 5, 0, 1, 4, 5)
    
    # Block 3: A(0:2, 4:5) * B(4:5, 0:6)
    # A rows 0,1. Col 4.
    # B row 4. Cols 0-5.
    add_schoolbook([0, 1], [4], [4], [0, 1, 2, 3, 4, 5], [0, 1], [0, 1, 2, 3, 4, 5])
    
    # Block 4: A(2:3, 0:2) * B(0:2, 0:6)
    # A row 2. Cols 0,1.
    # B rows 0,1. Cols 0-5.
    # Rank 2.
    # Term 1: A[2,0] * B[0, :]
    U[2 * m + 0][col] = 1
    for k in range(6):
        V[0 * p + k][col] = 1
    for k in range(6):
        W[k * n + 2][col] = 1
    col += 1
    # Term 2: A[2,1] * B[1, :]
    U[2 * m + 1][col] = 1
    for k in range(6):
        V[1 * p + k][col] = 1
    for k in range(6):
        W[k * n + 2][col] = 1
    col += 1

    # Block 5: A(2:3, 2:4) * B(2:4, 0:6)
    # A row 2. Cols 2,3.
    # B rows 2,3. Cols 0-5.
    # Rank 2.
    # Term 1: A[2,2] * B[2, :]
    U[2 * m + 2][col] = 1
    for k in range(6):
        V[2 * p + k][col] = 1
    for k in range(6):
        W[k * n + 2][col] = 1
    col += 1
    # Term 2: A[2,3] * B[3, :]
    U[2 * m + 3][col] = 1
    for k in range(6):
        V[3 * p + k][col] = 1
    for k in range(6):
        W[k * n + 2][col] = 1
    col += 1

    # Block 6: A(2:3, 4:5) * B(4:5, 0:6)
    # A row 2. Col 4.
    # B row 4. Cols 0-5.
    # Rank 1.
    U[2 * m + 4][col] = 1
    for k in range(6):
        V[4 * p + k][col] = 1
    for k in range(6):
        W[k * n + 2][col] = 1
    col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
