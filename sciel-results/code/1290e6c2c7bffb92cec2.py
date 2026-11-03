import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n x m x p using a decomposed approach (Strassen + Schoolbook).
    
    Dimensions: n=2, m=5, p=6.
    Rank: 54 (improved from 60).
    
    Strategy:
    Split A (2x5) into A_L (2x2) and A_R (2x3).
    Split B (5x6) into B_L (2x6) and B_R (3x6).
    C = A_L B_L + A_R B_R.
    
    A_L B_L: 2x2 * 2x6. Split B_L into 3 blocks of 2x2.
    Each 2x2*2x2 uses Strassen (rank 7). Total 3 * 7 = 21.
    
    A_R B_R: 2x3 * 3x6. Split B_R into 3 blocks of 3x2.
    Each 2x3*3x2 is split into (2x2*2x2) + (2x1*1x2).
    2x2*2x2 uses Strassen (rank 7).
    2x1*1x2 uses Schoolbook (rank 4).
    Total per block: 11. Total 3 * 11 = 33.
    
    Total Rank: 21 + 33 = 54.
    """
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to the decomposition
    # u_idx, v_idx, w_idx, u_val, v_val, w_val
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen decomposition for 2x2 * 2x2
    # Inputs: A (2x2), B (2x2)
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 in local 2x2
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 in local 2x2
    # Output C (2x2)
    def add_strassen(a_base, b_base, c_base):
        # a_base, b_base, c_base are offsets in the global matrices
        # A local (i,j) -> global U index: (a_base_i + i)*m + (a_base_j + j)
        # B local (i,j) -> global V index: (b_base_i + i)*p + (b_base_j + j)
        # C local (i,j) -> global W index: (c_base_j + j)*n + (c_base_i + i)
        
        # We need to map local 2x2 indices to global indices
        # A is 2x2. Local (0,0) is global (a_base_i, a_base_j)
        # Local (0,1) is global (a_base_i, a_base_j+1)
        # Local (1,0) is global (a_base_i+1, a_base_j)
        # Local (1,1) is global (a_base_i+1, a_base_j+1)
        
        # B is 2x2. Local (0,0) is global (b_base_i, b_base_j)
        # Local (0,1) is global (b_base_i, b_base_j+1)
        # Local (1,0) is global (b_base_i+1, b_base_j)
        # Local (1,1) is global (b_base_i+1, b_base_j+1)
        
        # C is 2x2. Local (0,0) is global (c_base_i, c_base_j)
        # Local (0,1) is global (c_base_i, c_base_j+1)
        # Local (1,0) is global (c_base_i+1, c_base_j)
        # Local (1,1) is global (c_base_i+1, c_base_j+1)
        
        # Strassen terms:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We decompose this into 7 rank-1 terms.
        # Each term is a product of a linear combo of A, linear combo of B, and a coefficient for C.
        
        # Term 1: M1 contributes to C00 (+1) and C11 (+1)
        # This is NOT a single rank 1 term in the tensor sense if it contributes to multiple C entries with different coefficients?
        # Wait. The tensor decomposition is sum_c U_c V_c W_c.
        # If a term contributes to multiple C entries, it means W_c has multiple non-zero entries.
        # Yes, W_c can have multiple non-zeros.
        
        # Let's define the 7 terms explicitly.
        
        # Term 1: (A00 + A11) * (B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        add_term(
            a_base[0]*m + a_base[1], 1, 
            b_base[0]*p + b_base[1], 1, 
            c_base[0]*n + c_base[1], 1
        )
        # Wait, add_term only sets ONE entry in U, ONE in V, ONE in W.
        # If U has two non-zeros, I need to call add_term twice? No, that would increase rank.
        # A rank 1 term is u_i v_j w_k.
        # If u has two non-zeros, it's not a basis vector.
        # The decomposition is sum_c (sum_i u_ic) (sum_j v_jc) (sum_k w_kc) ?
        # No, the tensor T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}.
        # So U_{ic} can be any value.
        # My add_term function sets a single entry.
        # I need to accumulate values for the current column.
        
        # Let's rewrite add_term to allow multiple entries per column.
        # Actually, I should build the column vectors first.
        
        pass # Placeholder, will implement correctly below

    # Reset col and use a buffer
    col = 0
    U_buf = [[0.0] * rank for _ in range(n * m)]
    V_buf = [[0.0] * rank for _ in range(m * p)]
    W_buf = [[0.0] * rank for _ in range(p * n)]
    
    def set_val(mat, idx, val):
        mat[idx][col] += val

    def add_strassen(a_base, b_base, c_base):
        # a_base: (i, j) of A's top-left
        # b_base: (i, j) of B's top-left
        # c_base: (i, j) of C's top-left
        
        ai, aj = a_base
        bi, bj = b_base
        ci, cj = c_base
        
        # Indices in global matrices
        # A: (ai, aj), (ai, aj+1), (ai+1, aj), (ai+1, aj+1)
        # B: (bi, bj), (bi, bj+1), (bi+1, bj), (bi+1, bj+1)
        # C: (ci, cj), (ci, cj+1), (ci+1, cj), (ci+1, cj+1)
        
        # Term 1: (A00 + A11) * (B00 + B11) -> C00 += 1, C11 += 1
        set_val(U_buf, ai*m + aj, 1)
        set_val(U_buf, (ai+1)*m + (aj+1), 1)
        set_val(V_buf, bi*p + bj, 1)
        set_val(V_buf, (bi+1)*p + (bj+1), 1)
        set_val(W_buf, cj*n + ci, 1) # C00
        set_val(W_buf, (cj+1)*n + (ci+1), 1) # C11
        col += 1

        # Term 2: (A10 + A11) * B00 -> C10 += 1
        set_val(U_buf, (ai+1)*m + aj, 1)
        set_val(U_buf, (ai+1)*m + (aj+1), 1)
        set_val(V_buf, bi*p + bj, 1)
        set_val(W_buf, cj*n + (ci+1), 1) # C10
        col += 1

        # Term 3: A00 * (B10 - B11) -> C01 += 1
        set_val(U_buf, ai*m + aj, 1)
        set_val(V_buf, (bi+1)*p + bj, 1)
        set_val(V_buf, (bi+1)*p + (bj+1), -1)
        set_val(W_buf, (cj+1)*n + ci, 1) # C01
        col += 1

        # Term 4: A11 * (B01 - B00) -> C00 -= 1, C10 += 1
        set_val(U_buf, (ai+1)*m + (aj+1), 1)
        set_val(V_buf, bi*p + (bj+1), 1)
        set_val(V_buf, bi*p + bj, -1)
        set_val(W_buf, cj*n + ci, -1) # C00
        set_val(W_buf, cj*n + (ci+1), 1) # C10
        col += 1

        # Term 5: (A00 + A01) * B11 -> C00 += 1
        set_val(U_buf, ai*m + aj, 1)
        set_val(U_buf, ai*m + (aj+1), 1)
        set_val(V_buf, (bi+1)*p + (bj+1), 1)
        set_val(W_buf, cj*n + ci, 1) # C00
        col += 1

        # Term 6: (A10 - A00) * (B00 + B01) -> C11 += 1
        set_val(U_buf, (ai+1)*m + aj, 1)
        set_val(U_buf, ai*m + aj, -1)
        set_val(V_buf, bi*p + bj, 1)
        set_val(V_buf, bi*p + (bj+1), 1)
        set_val(W_buf, (cj+1)*n + (ci+1), 1) # C11
        col += 1

        # Term 7: (A01 - A11) * (B10 + B11) -> C00 -= 1, C01 += 1
        set_val(U_buf, ai*m + (aj+1), 1)
        set_val(U_buf, (ai+1)*m + (aj+1), -1)
        set_val(V_buf, (bi+1)*p + bj, 1)
        set_val(V_buf, (bi+1)*p + (bj+1), 1)
        set_val(W_buf, cj*n + ci, -1) # C00
        set_val(W_buf, (cj+1)*n + ci, 1) # C01
        col += 1

    def add_schoolbook_2x1x2(a_base, b_base, c_base):
        # A is 2x1 (col aj), B is 1x2 (row bi), C is 2x2
        ai, aj = a_base
        bi, bj = b_base
        ci, cj = c_base
        
        # C00 = A00 * B00
        set_val(U_buf, ai*m + aj, 1)
        set_val(V_buf, bi*p + bj, 1)
        set_val(W_buf, cj*n + ci, 1)
        col += 1
        
        # C01 = A00 * B01
        set_val(U_buf, ai*m + aj, 1)
        set_val(V_buf, bi*p + (bj+1), 1)
        set_val(W_buf, (cj+1)*n + ci, 1)
        col += 1
        
        # C10 = A10 * B00
        set_val(U_buf, (ai+1)*m + aj, 1)
        set_val(V_buf, bi*p + bj, 1)
        set_val(W_buf, cj*n + (ci+1), 1)
        col += 1
        
        # C11 = A10 * B01
        set_val(U_buf, (ai+1)*m + aj, 1)
        set_val(V_buf, bi*p + (bj+1), 1)
        set_val(W_buf, (cj+1)*n + (ci+1), 1)
        col += 1

    # Process C_L: A (2x2) * B (2x6)
    # A base: (0,0)
    # B base: (0, 0), (0, 2), (0, 4)
    # C base: (0, 0), (0, 2), (0, 4)
    for c_col in [0, 2, 4]:
        add_strassen((0, 0), (0, c_col), (0, c_col))

    # Process C_R: A (2x3) * B (3x6)
    # A base: (0, 2) -> 2x3 block
    # B base: (2, 0), (2, 2), (2, 4) -> 3x2 blocks
    # C base: (0, 0), (0, 2), (0, 4)
    for c_col in [0, 2, 4]:
        # Split 2x3 * 3x2 into (2x2 * 2x2) + (2x1 * 1x2)
        # Part 1: A (2x2) at (0,2), B (2x2) at (2, c_col)
        add_strassen((0, 2), (2, c_col), (0, c_col))
        
        # Part 2: A (2x1) at (0,4), B (1x2) at (4, c_col)
        add_schoolbook_2x1x2((0, 4), (4, c_col), (0, c_col))

    if col != rank:
        print(f"Warning: Used {col} columns, expected {rank}")
        
    # Convert to lists of integers if possible, or keep floats
    # The problem asks for integer/half-integer entries.
    # Our entries are -1, 0, 1. These are integers.
    
    return {"U": U_buf, "V": V_buf, "W": W_buf}

# EVOLVE-BLOCK-END
