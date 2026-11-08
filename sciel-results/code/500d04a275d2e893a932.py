import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculated as 68
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen decomposition for 2x2x2
    # Local indices: i,j in {0,1} for A, j,k in {0,1} for B, k,i in {0,1} for C
    # A is 2x2, B is 2x2, C is 2x2
    def add_strassen_block(base_i, base_j, base_k):
        # base_i: starting row in A
        # base_j: starting col in A / row in B
        # base_k: starting col in B
        # A indices: (base_i+0, base_j+0), (base_i+0, base_j+1), (base_i+1, base_j+0), (base_i+1, base_j+1)
        # B indices: (base_j+0, base_k+0), (base_j+0, base_k+1), (base_j+1, base_k+0), (base_j+1, base_k+1)
        # C indices: (base_k+0, base_i+0), (base_k+0, base_i+1), (base_k+1, base_i+0), (base_k+1, base_i+1)
        
        a00 = (base_i, base_j)
        a01 = (base_i, base_j+1)
        a10 = (base_i+1, base_j)
        a11 = (base_i+1, base_j+1)
        
        b00 = (base_j, base_k)
        b01 = (base_j, base_k+1)
        b10 = (base_j+1, base_k)
        b11 = (base_j+1, base_k+1)
        
        c00 = (base_k, base_i)
        c01 = (base_k, base_i+1)
        c10 = (base_k+1, base_i)
        c11 = (base_k+1, base_i+1)
        
        # M1 = (a00+a11)(b00+b11)
        add_term(*a00, *b00, *c00, 1, 1, 1)
        add_term(*a11, *b00, *c00, 1, 1, 1)
        add_term(*a00, *b11, *c00, 1, 1, 1)
        add_term(*a11, *b11, *c00, 1, 1, 1)
        
        # M2 = (a10+a11)b00
        add_term(*a10, *b00, *c10, 1, 1, 1)
        add_term(*a11, *b00, *c10, 1, 1, 1)
        add_term(*a10, *b00, *c11, 1, 1, 1)
        add_term(*a11, *b00, *c11, 1, 1, 1)
        
        # M3 = a00(b10-b11)
        add_term(*a00, *b10, *c01, 1, 1, 1)
        add_term(*a00, *b11, *c01, 1, -1, 1)
        add_term(*a00, *b10, *c11, 1, 1, 1)
        add_term(*a00, *b11, *c11, 1, -1, 1)
        
        # M4 = a11(b01-b00)
        add_term(*a11, *b01, *c00, 1, 1, 1)
        add_term(*a11, *b00, *c00, 1, -1, 1)
        add_term(*a11, *b01, *c10, 1, 1, 1)
        add_term(*a11, *b00, *c10, 1, -1, 1)
        
        # M5 = (a00+a01)b11
        add_term(*a00, *b11, *c01, 1, 1, 1)
        add_term(*a01, *b11, *c01, 1, 1, 1)
        add_term(*a00, *b11, *c11, 1, 1, 1)
        add_term(*a01, *b11, *c11, 1, 1, 1)
        
        # M6 = (a10-a00)(b00+b10)
        add_term(*a10, *b00, *c11, 1, 1, 1)
        add_term(*a10, *b10, *c11, 1, 1, 1)
        add_term(*a00, *b00, *c11, -1, 1, 1)
        add_term(*a00, *b10, *c11, -1, 1, 1)
        
        # M7 = (a01-a11)(b01+b11)
        add_term(*a01, *b01, *c00, 1, 1, 1)
        add_term(*a01, *b11, *c00, 1, 1, 1)
        add_term(*a11, *b01, *c00, -1, 1, 1)
        add_term(*a11, *b11, *c00, -1, 1, 1)
        add_term(*a01, *b01, *c10, 1, 1, 1)
        add_term(*a01, *b11, *c10, 1, 1, 1)
        add_term(*a11, *b01, *c10, -1, 1, 1)
        add_term(*a11, *b11, *c10, -1, 1, 1)

    # Wait, the above add_term calls are incorrect for Strassen.
    # Strassen combines terms. I need to add the combined terms as single rank-1 tensors?
    # No, the decomposition is a sum of rank-1 tensors.
    # Each M_i is a product of a linear combination of A entries and a linear combination of B entries.
    # This product is then distributed to C entries.
    # So for M1, we have 4 terms in the sum for C.
    # But M1 itself is one "rank 1" in the intermediate sense?
    # No, the rank is the number of terms in the sum $\sum U_r V_r W_r$.
    # In Strassen, we have 7 "products" $M_i$.
    # Each $M_i$ contributes to multiple $C_{ij}$.
    # So for each $M_i$, we add 4 terms to the global sum?
    # No, $M_i$ is a scalar. $C_{ij} = \sum c_{ij}^i M_i$.
    # So $C = \sum_i M_i \otimes C_i$ where $C_i$ is the matrix of coefficients.
    # This means for each $M_i$, we add 4 rank-1 tensors (one for each non-zero coeff in $C_i$).
    # So 7 M's * 4 C's = 28 terms?
    # Yes, that matches the rank 28 for 4 blocks of 2x2x2.
    
    # My add_term function adds ONE rank-1 tensor.
    # So I should call it 4 times for each M_i?
    # Yes.
    
    # Let's re-verify the structure.
    # The rank of 2x2x2 is 7.
    # This means there are 7 terms in the decomposition.
    # Term r: $U_r \otimes V_r \otimes W_r$.
    # $U_r$ is a 2x2 matrix (sparse).
    # $V_r$ is a 2x2 matrix (sparse).
    # $W_r$ is a 2x2 matrix (sparse).
    # In my add_term, I am adding a term where U, V, W have a single non-zero entry.
    # This is not how Strassen works directly.
    # Strassen terms have multiple non-zero entries in U, V, or W.
    # Example M1: $U = \text{diag}(1,1)$? No.
    # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
    # This corresponds to $U_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $V_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $W_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$?
    # No, $W_1$ would be the matrix of coefficients for M1 in C.
    # $C_{00}$ gets +1, $C_{11}$ gets +1. So $W_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$.
    # So one rank-1 tensor can have multiple non-zero entries in U, V, W.
    # My current `U`, `V`, `W` are 2D arrays.
    # I can store multiple non-zero entries in a single column (rank index).
    # So I should NOT use `add_term` for each element.
    # I should use a function that adds to a specific column index.
    
    col = 0
    def add_strassen_block_v2(base_i, base_j, base_k):
        nonlocal col
        # Define the 7 terms
        # Term 1: M1
        U[base_i*3+base_j][col] = 1
        U[(base_i+1)*3+(base_j+1)][col] = 1
        V[base_j*8+base_k][col] = 1
        V[(base_j+1)*8+(base_k+1)][col] = 1
        W[base_k*3+base_i][col] = 1
        W[(base_k+1)*3+(base_i+1)][col] = 1
        col += 1
        
        # Term 2: M2
        U[(base_i+1)*3+base_j][col] = 1
        U[(base_i+1)*3+(base_j+1)][col] = 1
        V[base_j*8+base_k][col] = 1
        W[(base_k+1)*3+base_i][col] = 1
        W[(base_k+1)*3+(base_i+1)][col] = 1
        col += 1
        
        # Term 3: M3
        U[base_i*3+base_j][col] = 1
        V[(base_j+1)*8+base_k][col] = 1
        V[(base_j+1)*8+(base_k+1)][col] = -1
        W[base_k*3+(base_i+1)][col] = 1
        W[(base_k+1)*3+(base_i+1)][col] = 1
        col += 1
        
        # Term 4: M4
        U[(base_i+1)*3+(base_j+1)][col] = 1
        V[base_j*8+(base_k+1)][col] = 1
        V[base_j*8+base_k][col] = -1
        W[base_k*3+base_i][col] = 1
        W[(base_k+1)*3+base_i][col] = 1
        col += 1
        
        # Term 5: M5
        U[base_i*3+base_j][col] = 1
        U[base_i*3+(base_j+1)][col] = 1
        V[(base_j+1)*8+(base_k+1)][col] = 1
        W[base_k*3+(base_i+1)][col] = 1
        W[(base_k+1)*3+(base_i+1)][col] = 1
        col += 1
        
        # Term 6: M6
        U[(base_i+1)*3+base_j][col] = 1
        U[base_i*3+base_j][col] = -1
        V[base_j*8+base_k][col] = 1
        V[(base_j+1)*8+base_k][col] = 1
        W[(base_k+1)*3+(base_i+1)][col] = 1
        col += 1
        
        # Term 7: M7
        U[base_i*3+(base_j+1)][col] = 1
        U[(base_i+1)*3+(base_j+1)][col] = -1
        V[base_j*8+(base_k+1)][col] = 1
        V[(base_j+1)*8+(base_k+1)][col] = 1
        W[base_k*3+base_i][col] = 1
        W[(base_k+1)*3+base_i][col] = 1
        col += 1

    # Now add the blocks
    # 4 blocks of 2x2x2
    add_strassen_block_v2(0, 0, 0)
    add_strassen_block_v2(0, 0, 2)
    add_strassen_block_v2(0, 0, 4)
    add_strassen_block_v2(0, 0, 6)
    
    # 4 blocks of 1x2x2 (A is 1x2, B is 2x2)
    # A row: 2 (base_i=2). B rows: 0,1. B cols: 0..7.
    # Wait, A is 3x3. Row 2 is the last row.
    # B is 3x8. Rows 0,1 are the top 2 rows.
    # So A_{2, 0..1} (1x2) * B_{0..1, k..k+1} (2x2).
    # A indices: (2, 0), (2, 1).
    # B indices: (0, k), (0, k+1), (1, k), (1, k+1).
    # C indices: (k, 2), (k+1, 2).
    
    def add_1x2x2_block(base_i, base_j, base_k):
        nonlocal col
        # A: (base_i, base_j), (base_i, base_j+1)
        # B: (base_j, base_k), (base_j, base_k+1), (base_j+1, base_k), (base_j+1, base_k+1)
        # C: (base_k, base_i), (base_k+1, base_i)
        
        # Term 1: A00 * B00 -> C0
        U[base_i*3+base_j][col] = 1
        V[base_j*8+base_k][col] = 1
        W[base_k*3+base_i][col] = 1
        col += 1
        
        # Term 2: A00 * B01 -> C1
        U[base_i*3+base_j][col] = 1
        V[base_j*8+(base_k+1)][col] = 1
        W[(base_k+1)*3+base_i][col] = 1
        col += 1
        
        # Term 3: A01 * B10 -> C0
        U[base_i*3+(base_j+1)][col] = 1
        V[(base_j+1)*8+base_k][col] = 1
        W[base_k*3+base_i][col] = 1
        col += 1
        
        # Term 4: A01 * B11 -> C1
        U[base_i*3+(base_j+1)][col] = 1
        V[(base_j+1)*8+(base_k+1)][col] = 1
        W[(base_k+1)*3+base_i][col] = 1
        col += 1

    add_1x2x2_block(2, 0, 0)
    add_1x2x2_block(2, 0, 2)
    add_1x2x2_block(2, 0, 4)
    add_1x2x2_block(2, 0, 6)
    
    # 4 blocks of 2x1x2 (A is 2x1, B is 1x2)
    # A rows: 0,1. Col: 2.
    # B row: 2. Cols: 0..7.
    # A indices: (0, 2), (1, 2).
    # B indices: (2, k), (2, k+1).
    # C indices: (k, 0), (k, 1), (k+1, 0), (k+1, 1).
    
    def add_2x1x2_block(base_i, base_j, base_k):
        nonlocal col
        # A: (base_i, base_j), (base_i+1, base_j)
        # B: (base_j, base_k), (base_j, base_k+1)
        # C: (base_k, base_i), (base_k, base_i+1), (base_k+1, base_i), (base_k+1, base_i+1)
        
        # Term 1: A00 * B00 -> C00
        U[base_i*3+base_j][col] = 1
        V[base_j*8+base_k][col] = 1
        W[base_k*3+base_i][col] = 1
        col += 1
        
        # Term 2: A00 * B01 -> C01
        U[base_i*3+base_j][col] = 1
        V[base_j*8+(base_k+1)][col] = 1
        W[(base_k+1)*3+base_i][col] = 1
        col += 1
        
        # Term 3: A10 * B00 -> C10
        U[(base_i+1)*3+base_j][col] = 1
        V[base_j*8+base_k][col] = 1
        W[base_k*3+(base_i+1)][col] = 1
        col += 1
        
        # Term 4: A10 * B01 -> C11
        U[(base_i+1)*3+base_j][col] = 1
        V[base_j*8+(base_k+1)][col] = 1
        W[(base_k+1)*3+(base_i+1)][col] = 1
        col += 1

    add_2x1x2_block(0, 2, 0)
    add_2x1x2_block(0, 2, 2)
    add_2x1x2_block(0, 2, 4)
    add_2x1x2_block(0, 2, 6)
    
    # 4 blocks of 1x1x2 (A is 1x1, B is 1x2)
    # A: (2, 2). B: (2, k), (2, k+1). C: (k, 2), (k+1, 2).
    
    def add_1x1x2_block(base_i, base_j, base_k):
        nonlocal col
        # A: (base_i, base_j)
        # B: (base_j, base_k), (base_j, base_k+1)
        # C: (base_k, base_i), (base_k+1, base_i)
        
        # Term 1: A00 * B00 -> C0
        U[base_i*3+base_j][col] = 1
        V[base_j*8+base_k][col] = 1
        W[base_k*3+base_i][col] = 1
        col += 1
        
        # Term 2: A00 * B01 -> C1
        U[base_i*3+base_j][col] = 1
        V[base_j*8+(base_k+1)][col] = 1
        W[(base_k+1)*3+base_i][col] = 1
        col += 1

    add_1x1x2_block(2, 2, 0)
    add_1x1x2_block(2, 2, 2)
    add_1x1x2_block(2, 2, 4)
    add_1x1x2_block(2, 2, 6)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
