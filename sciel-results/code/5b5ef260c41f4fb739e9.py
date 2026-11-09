import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen helper for 2x2 blocks
    # A indices: (i, j) with i in [0,2), j in [j_off, j_off+2)
    # B indices: (j, k) with j in [j_off, j_off+2), k in [k_off, k_off+2)
    # C indices: (i, k) with i in [0,2), k in [k_off, k_off+2)
    def add_strassen(i_off, j_off, k_off):
        # A is 2x2 at (i_off, j_off)
        # B is 2x2 at (j_off, k_off)
        # C is 2x2 at (i_off, k_off)
        
        # T1: (A00+A11)(B00+B11)(C00+C11)
        add_term(i_off*4 + j_off, j_off*4 + k_off, k_off*3 + i_off, 1, 1, 1)
        add_term(i_off*4 + j_off, j_off*4 + k_off, (k_off+1)*3 + i_off, 1, 1, 1)
        add_term(i_off*4 + j_off, (j_off+1)*4 + k_off, k_off*3 + i_off, 1, 1, 1)
        add_term(i_off*4 + j_off, (j_off+1)*4 + k_off, (k_off+1)*3 + i_off, 1, 1, 1)
        add_term((i_off+1)*4 + j_off, j_off*4 + k_off, k_off*3 + i_off, 1, 1, 1)
        add_term((i_off+1)*4 + j_off, j_off*4 + k_off, (k_off+1)*3 + i_off, 1, 1, 1)
        add_term((i_off+1)*4 + j_off, (j_off+1)*4 + k_off, k_off*3 + i_off, 1, 1, 1)
        add_term((i_off+1)*4 + j_off, (j_off+1)*4 + k_off, (k_off+1)*3 + i_off, 1, 1, 1)
        # Wait, this adds 8 terms for T1? No, T1 is one rank-1 term in the tensor sense?
        # No, the term is (A00+A11)(B00+B11)(C00+C11).
        # This expands to 8 products of single entries.
        # But in the tensor decomposition, we want to minimize the number of rank-1 tensors.
        # Strassen's algorithm says the rank is 7.
        # The 7 terms are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # And C is a linear combination of M's.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So the rank-1 terms are:
        # 1. (A00+A11)(B00+B11)C00  -> No, this is not a rank-1 tensor in the original variables.
        # The rank-1 tensors are of the form u v w where u, v, w are linear forms.
        # Term 1: u1 = A00+A11, v1 = B00+B11, w1 = C00+C11
        # Term 2: u2 = A10+A11, v2 = B00, w2 = C10-C11
        # Term 3: u3 = A00, v3 = B01-B11, w3 = C01+C11
        # Term 4: u4 = A11, v4 = B10-B00, w4 = C00+C10
        # Term 5: u5 = A00+A01, v5 = B11, w5 = -C00+C01
        # Term 6: u6 = A10-A00, v6 = B00+B01, w6 = C11
        # Term 7: u7 = A01-A11, v7 = B10+B11, w7 = C00
        
        # I need to add these 7 terms.
        
        # Term 1
        add_term(i_off*4 + j_off, j_off*4 + k_off, k_off*3 + i_off, 1, 1, 1) # A00 B00 C00
        add_term(i_off*4 + j_off, j_off*4 + k_off, (k_off+1)*3 + i_off, 1, 1, 1) # A00 B00 C11
        add_term(i_off*4 + j_off, (j_off+1)*4 + k_off, k_off*3 + i_off, 1, 1, 1) # A00 B11 C00
        add_term(i_off*4 + j_off, (j_off+1)*4 + k_off, (k_off+1)*3 + i_off, 1, 1, 1) # A00 B11 C11
        add_term((i_off+1)*4 + j_off, j_off*4 + k_off, k_off*3 + i_off, 1, 1, 1) # A10 B00 C00
        add_term((i_off+1)*4 + j_off, j_off*4 + k_off, (k_off+1)*3 + i_off, 1, 1, 1) # A10 B00 C11
        add_term((i_off+1)*4 + j_off, (j_off+1)*4 + k_off, k_off*3 + i_off, 1, 1, 1) # A10 B11 C00
        add_term((i_off+1)*4 + j_off, (j_off+1)*4 + k_off, (k_off+1)*3 + i_off, 1, 1, 1) # A10 B11 C11
        # This is adding 8 terms for the first linear form product.
        # But the rank-1 tensor is defined by the linear forms.
        # The evaluator expects U, V, W matrices.
        # U has entries for A indices.
        # If I add 1 to U[A00] and U[A10], that's a linear form.
        # So for Term 1:
        # U[A00] += 1, U[A10] += 1
        # V[B00] += 1, V[B11] += 1
        # W[C00] += 1, W[C11] += 1
        # This is ONE rank-1 term in the decomposition?
        # No, the decomposition is $\sum_{r} U_{i,r} V_{j,r} W_{k,r}$.
        # If I set $U_{A00, r}=1, U_{A10, r}=1$, then the term is $1 \cdot B \cdot C$.
        # This corresponds to the linear form $A00+A10$.
        # So yes, adding to the same column index `col` for multiple A indices represents a linear form.
        # So I should NOT increment `col` for each entry in the linear form.
        # I should increment `col` once per rank-1 tensor.
        
        # Let's restart the add_term logic.
        
    # Corrected add_term
    def add_rank1_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        nonlocal col
        for u_idx, u_val in zip(u_indices, u_vals):
            U[u_idx][col] += u_val
        for v_idx, v_val in zip(v_indices, v_vals):
            V[v_idx][col] += v_val
        for w_idx, w_val in zip(w_indices, w_vals):
            W[w_idx][col] += w_val
        col += 1

    # Strassen 2x2
    def add_strassen(i_off, j_off, k_off):
        # A: (i_off, j_off) to (i_off+1, j_off+1)
        # B: (j_off, k_off) to (j_off+1, k_off+1)
        # C: (i_off, k_off) to (i_off+1, k_off+1)
        
        # Term 1: (A00+A10)(B00+B11)(C00+C11)
        add_rank1_term(
            [i_off*4+j_off, (i_off+1)*4+j_off], [1, 1],
            [j_off*4+k_off, (j_off+1)*4+k_off], [1, 1],
            [k_off*3+i_off, (k_off+1)*3+i_off], [1, 1]
        )
        # Term 2: (A10+A11)B00(C10-C11)
        add_rank1_term(
            [(i_off+1)*4+j_off, (i_off+1)*4+j_off+1], [1, 1],
            [j_off*4+k_off], [1],
            [(k_off)*3+(i_off+1), (k_off+1)*3+(i_off+1)], [1, -1]
        )
        # Term 3: A00(B01-B11)(C01+C11)
        add_rank1_term(
            [i_off*4+j_off], [1],
            [j_off*4+k_off+1, (j_off+1)*4+k_off+1], [1, -1],
            [(k_off+1)*3+i_off, (k_off+1)*3+(i_off+1)], [1, 1]
        )
        # Term 4: A11(B10-B00)(C00+C10)
        add_rank1_term(
            [(i_off+1)*4+j_off+1], [1],
            [(j_off+1)*4+k_off, j_off*4+k_off], [1, -1],
            [k_off*3+i_off, k_off*3+(i_off+1)], [1, 1]
        )
        # Term 5: (A00+A01)B11(-C00+C01)
        add_rank1_term(
            [i_off*4+j_off, i_off*4+j_off+1], [1, 1],
            [(j_off+1)*4+k_off+1], [1],
            [k_off*3+i_off, (k_off+1)*3+i_off], [-1, 1]
        )
        # Term 6: (A10-A00)(B00+B01)C11
        add_rank1_term(
            [(i_off+1)*4+j_off, i_off*4+j_off], [1, -1],
            [j_off*4+k_off, j_off*4+k_off+1], [1, 1],
            [(k_off+1)*3+(i_off+1)], [1]
        )
        # Term 7: (A01-A11)(B10+B11)C00
        add_rank1_term(
            [i_off*4+j_off+1, (i_off+1)*4+j_off+1], [1, -1],
            [(j_off+1)*4+k_off, (j_off+1)*4+k_off+1], [1, 1],
            [k_off*3+i_off], [1]
        )

    # Schoolbook 1x2x2
    def add_schoolbook_1x2x2(i_idx, j_off, k_off):
        # A: (i_idx, j_off) to (i_idx, j_off+1)
        # B: (j_off, k_off) to (j_off+1, k_off+1)
        # C: (i_idx, k_off) to (i_idx, k_off+1)
        
        # T1: A0 B0 C0
        add_rank1_term(
            [i_idx*4+j_off], [1],
            [j_off*4+k_off], [1],
            [k_off*3+i_idx], [1]
        )
        # T2: A1 B1 C0
        add_rank1_term(
            [i_idx*4+j_off+1], [1],
            [(j_off+1)*4+k_off], [1],
            [k_off*3+i_idx], [1]
        )
        # T3: A0 B2 C1
        add_rank1_term(
            [i_idx*4+j_off], [1],
            [j_off*4+k_off+1], [1],
            [(k_off+1)*3+i_idx], [1]
        )
        # T4: A1 B3 C1
        add_rank1_term(
            [i_idx*4+j_off+1], [1],
            [(j_off+1)*4+k_off+1], [1],
            [(k_off+1)*3+i_idx], [1]
        )

    # Block 1: A_top (2x4) * B_left (4x2)
    # A_top: i in [0,2), j in [0,4)
    # B_left: j in [0,4), k in [0,2)
    # C: i in [0,2), k in [0,2)
    # Split into A11 (2x2) B11 (2x2) and A12 (2x2) B21 (2x2)
    add_strassen(0, 0, 0) # A11 B11
    add_strassen(0, 2, 0) # A12 B21
    
    # Block 2: A_top (2x4) * B_right (4x2)
    # B_right: j in [0,4), k in [2,4)
    # C: i in [0,2), k in [2,4)
    add_strassen(0, 0, 2) # A11 B12
    add_strassen(0, 2, 2) # A12 B22
    
    # Block 3: A_bot (1x4) * B_left (4x2)
    # A_bot: i=2, j in [0,4)
    # B_left: j in [0,4), k in [0,2)
    # C: i=2, k in [0,2)
    # Split into A21 (1x2) B11 (2x2) and A22 (1x2) B21 (2x2)
    add_schoolbook_1x2x2(2, 0, 0) # A21 B11
    add_schoolbook_1x2x2(2, 2, 0) # A22 B21
    
    # Block 4: A_bot (1x4) * B_right (4x2)
    # B_right: j in [0,4), k in [2,4)
    # C: i=2, k in [2,4)
    add_schoolbook_1x2x2(2, 0, 2) # A21 B12
    add_schoolbook_1x2x2(2, 2, 2) # A22 B22
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
