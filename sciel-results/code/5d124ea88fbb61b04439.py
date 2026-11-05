"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We use Strassen for 2x2x2 blocks where possible, and schoolbook/outer-product for the rest.
    # n=4, m=5, p=6
    # Blocks for Strassen: i in {0,1}, j in {0,1} (2 blocks of size 2), k in {0,1,2} (3 blocks of size 2)
    # This covers i:0-3, j:0-3, k:0-5
    # Remainder: j=4, all i, all k. This is an outer product, rank 1.
    
    rank = 0
    # Pre-allocate lists or use numpy? The interface expects lists of lists.
    # Using lists of lists is safe for the evaluator.
    
    # We will build the columns and then transpose to the required shapes.
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Let's accumulate U_cols, V_cols, W_cols as lists of (index, value) tuples or just build the full matrices if rank is known.
    # Rank is fixed at 85.
    
    # 1. Strassen Blocks
    # 2. Remainder Block
    
    # Helper to add a rank-1 term
    # U_idx: list of (row, val)
    # V_idx: list of (row, val)
    # W_idx: list of (row, val)
    
    # We will build the final arrays.
    # n*m = 20
    # m*p = 30
    # p*n = 24
    
    U_data = []
    V_data = []
    W_data = []
    
    def add_term(u_terms, v_terms, w_terms):
        """
        u_terms: list of (u_row, coeff)
        v_terms: list of (v_row, coeff)
        w_terms: list of (w_row, coeff)
        """
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        
        for r, c in u_terms:
            U_col[r] = c
        for r, c in v_terms:
            V_col[r] = c
        for r, c in w_terms:
            W_col[r] = c
            
        U_data.append(U_col)
        V_data.append(V_col)
        W_data.append(W_col)

    # Strassen 2x2x2 Decomposition
    # P1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # P2 = (A10+A11)B00 -> C10 += 1
    # P3 = A00(B01-B11) -> C01 += 1, C11 -= 1 (Wait, check Strassen formulas)
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to add these as separate rank-1 terms in the U, V, W matrices.
    # Each M corresponds to one multiplication, so one column in U, V, W.
    
    def add_strassen_block(i_base, j_base, k_base):
        # Local indices 0,1 for i, j, k
        # Global indices:
        # I0 = i_base + 0, I1 = i_base + 1
        # J0 = j_base + 0, J1 = j_base + 1
        # K0 = k_base + 0, K1 = k_base + 1
        
        # M1: (A_I0_J0 + A_I1_J1) * (B_J0_K0 + B_J1_K1)
        # U: A_I0_J0 (1), A_I1_J1 (1)
        # V: B_J0_K0 (1), B_J1_K1 (1)
        # W: C_I0_K0 (1), C_I1_K1 (1)
        # Note: W index is k*n + i.
        # C_I0_K0 -> W[K0*n + I0]
        # C_I1_K1 -> W[K1*n + I1]
        
        add_term(
            [(I0*m + J0, 1), (I1*m + J1, 1)],
            [(J0*p + K0, 1), (J1*p + K1, 1)],
            [(K0*n + I0, 1), (K1*n + I1, 1)]
        )
        
        # M2: (A_I1_J0 + A_I1_J1) * B_J0_K0
        # U: A_I1_J0 (1), A_I1_J1 (1)
        # V: B_J0_K0 (1)
        # W: C_I1_K0 (1)
        add_term(
            [(I1*m + J0, 1), (I1*m + J1, 1)],
            [(J0*p + K0, 1)],
            [(K0*n + I1, 1)]
        )
        
        # M3: A_I0_J0 * (B_J0_K1 - B_J1_K1)
        # U: A_I0_J0 (1)
        # V: B_J0_K1 (1), B_J1_K1 (-1)
        # W: C_I0_K1 (1)
        add_term(
            [(I0*m + J0, 1)],
            [(J0*p + K1, 1), (J1*p + K1, -1)],
            [(K1*n + I0, 1)]
        )
        
        # M4: A_I1_J1 * (B_J1_K0 - B_J0_K0)
        # U: A_I1_J1 (1)
        # V: B_J1_K0 (1), B_J0_K0 (-1)
        # W: C_I1_K0 (1)
        add_term(
            [(I1*m + J1, 1)],
            [(J1*p + K0, 1), (J0*p + K0, -1)],
            [(K0*n + I1, 1)]
        )
        
        # M5: (A_I0_J0 + A_I0_J1) * B_J1_K1
        # U: A_I0_J0 (1), A_I0_J1 (1)
        # V: B_J1_K1 (1)
        # W: C_I0_K1 (-1)  (From C00 = ... - M5 ... and C01 = ... + M5 ...)
        # Wait, C01 = M3 + M5. So C_I0_K1 gets +1 from M5.
        # C00 gets -1 from M5.
        # So W: C_I0_K1 (1), C_I0_K0 (-1)
        add_term(
            [(I0*m + J0, 1), (I0*m + J1, 1)],
            [(J1*p + K1, 1)],
            [(K1*n + I0, 1), (K0*n + I0, -1)]
        )
        
        # M6: (A_I1_J0 - A_I0_J0) * (B_J0_K0 + B_J0_K1)
        # U: A_I1_J0 (1), A_I0_J0 (-1)
        # V: B_J0_K0 (1), B_J0_K1 (1)
        # W: C_I1_K1 (1)
        add_term(
            [(I1*m + J0, 1), (I0*m + J0, -1)],
            [(J0*p + K0, 1), (J0*p + K1, 1)],
            [(K1*n + I1, 1)]
        )
        
        # M7: (A_I0_J1 - A_I1_J1) * (B_J1_K0 + B_J1_K1)
        # U: A_I0_J1 (1), A_I1_J1 (-1)
        # V: B_J1_K0 (1), B_J1_K1 (1)
        # W: C_I0_K0 (1)
        add_term(
            [(I0*m + J1, 1), (I1*m + J1, -1)],
            [(J1*p + K0, 1), (J1*p + K1, 1)],
            [(K0*n + I0, 1)]
        )

    # Loop over blocks
    # i_base in [0, 2]
    # j_base in [0, 2]  (covers j=0,1 and j=2,3)
    # k_base in [0, 2, 4] (covers k=0,1; 2,3; 4,5)
    
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2, 4]:
                I0 = i_base
                I1 = i_base + 1
                J0 = j_base
                J1 = j_base + 1
                K0 = k_base
                K1 = k_base + 1
                add_strassen_block(i_base, j_base, k_base)

    # Remainder: j=4
    # A_i4 * B_4k for all i in 0..3, k in 0..5
    # This is an outer product.
    # U: A_i4 = 1 for all i
    # V: B_4k = 1 for all k
    # W: C_ik = 1 for all i, k
    
    u_rem = []
    for i in range(n):
        u_rem.append((i*m + 4, 1))
        
    v_rem = []
    for k in range(p):
        v_rem.append((4*p + k, 1))
        
    w_rem = []
    for k in range(p):
        for i in range(n):
            w_rem.append((k*n + i, 1))
            
    add_term(u_rem, v_rem, w_rem)

    # Convert lists of columns to lists of rows (transposed)
    # Current: U_data is list of columns, each is list of rows.
    # Target: U is list of rows, each is list of columns.
    
    U = [[0] * len(U_data) for _ in range(n * m)]
    V = [[0] * len(V_data) for _ in range(m * p)]
    W = [[0] * len(W_data) for _ in range(p * n)]
    
    for c in range(len(U_data)):
        for r in range(n * m):
            U[r][c] = U_data[c][r]
        for r in range(m * p):
            V[r][c] = V_data[c][r]
        for r in range(p * n):
            W[r][c] = W_data[c][r]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
