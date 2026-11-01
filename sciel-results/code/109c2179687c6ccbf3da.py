"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    # We use Strassen for pairs of columns and trivial for the last column.
    # 15 = 7*2 + 1
    # Rank = 7*7 + 4 = 53
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For a pair of columns (k1, k2):
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices for this pair: (0,k1), (1,k1), (0,k2), (1,k2)
    #   Let's map: B00->(0,k1), B01->(0,k2), B10->(1,k1), B11->(1,k2)
    # C indices for this pair: (0,k1), (1,k1), (0,k2), (1,k2)
    #   C00->(0,k1), C01->(0,k2), C10->(1,k1), C11->(1,k2)
    
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict from A_index (0-3) to coeff
        v_coeffs: dict from B_local_index (0-3) to coeff, where
                  0->(0,k1), 1->(0,k2), 2->(1,k1), 3->(1,k2)
        w_coeffs: dict from C_local_index (0-3) to coeff, where
                  0->(0,k1), 1->(0,k2), 2->(1,k1), 3->(1,k2)
        """
        U_rows.append(u_coeffs)
        V_rows.append(v_coeffs)
        W_rows.append(w_coeffs)
    
    # The 7 Strassen terms (u, v, w) where indices are local:
    # A: 00=0, 01=1, 10=2, 11=3
    # B: 00=0, 01=1, 10=2, 11=3
    # C: 00=0, 01=1, 10=2, 11=3
    
    strassen_terms = [
        # M1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:-1, 1:1}),  # C00 += M1, C11 -= M1 ... wait let me re-derive
    ]
    
    # Let me be more careful. The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each product Mi, we have a U vector (coeffs of A), V vector (coeffs of B),
    # and for each C entry, a coefficient of Mi.
    
    # Let's define the 7 terms as (u_dict, v_dict) and then the C combination matrix.
    terms = [
        # (u_dict, v_dict)
        ({0:1, 3:1}, {0:1, 3:1}),  # M1
        ({2:1, 3:1}, {0:1}),       # M2
        ({0:1}, {2:1, 3:-1}),      # M3
        ({3:1}, {1:1, 0:-1}),      # M4
        ({0:1, 1:1}, {3:1}),       # M5
        ({2:1, 0:-1}, {0:1, 1:1}), # M6
        ({1:1, 3:-1}, {2:1, 3:1}), # M7
    ]
    
    # C combination: C[c] = sum_i coeff[i][c] * Mi
    # c: 0=C00, 1=C01, 2=C10, 3=C11
    c_comb = [
        {0:1, 3:1, 4:-1, 6:1},   # C00 = M1 + M4 - M5 + M7
        {2:1, 4:1},               # C01 = M3 + M5
        {1:1, 3:1},               # C10 = M2 + M4
        {0:1, 1:-1, 2:1, 5:1},   # C11 = M1 - M2 + M3 + M6
    ]
    
    # For a pair (k1, k2), local B indices:
    # 0 -> (0, k1), 1 -> (0, k2), 2 -> (1, k1), 3 -> (1, k2)
    # Local C indices:
    # 0 -> (0, k1), 1 -> (0, k2), 2 -> (1, k1), 3 -> (1, k2)
    
    for group in range(7):
        k1 = 2 * group
        k2 = 2 * group + 1
        for t_idx in range(7):
            u_dict, v_dict = terms[t_idx]
            # Build U vector for A (4 entries)
            u_vec = [0.0] * 4
            for a_idx, coeff in u_dict.items():
                u_vec[a_idx] = coeff
            # Build V vector for B (4 entries, local to this pair)
            v_vec = [0.0] * 4
            for b_idx, coeff in v_dict.items():
                v_vec[b_idx] = coeff
            # Build W vector for C (4 entries, local to this pair)
            w_vec = [0.0] * 4
            for c_idx, c_dict in enumerate(c_comb):
                if t_idx in c_dict:
                    w_vec[c_idx] = c_dict[t_idx]
            
            U_rows.append(u_vec)
            V_rows.append(v_vec)
            W_rows.append(w_vec)
            # Store which global indices these local indices map to
            # We'll handle this by expanding later
    
    # For the single column k=14:
    # Trivial: 4 terms
    # C[i,k] = sum_j A[i,j] B[j,k]
    for i in range(2):
        for j in range(2):
            u_vec = [0.0] * 4
            u_vec[i * 2 + j] = 1.0
            v_vec = [0.0] * 15
            v_vec[j * 15 + 14] = 1.0
            w_vec = [0.0] * 30
            w_vec[14 * 2 + i] = 1.0
            U_rows.append(u_vec)
            V_rows.append(v_vec)
            W_rows.append(w_vec)
    
    # Now we need to expand the Strassen terms to full dimensionality
    # Let's redo this more carefully by building full vectors directly
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    for group in range(7):
        k1 = 2 * group
        k2 = 2 * group + 1
        for t_idx in range(7):
            u_dict, v_dict = terms[t_idx]
            # Full U vector (4 entries)
            u_vec = [0.0] * 4
            for a_idx, coeff in u_dict.items():
                u_vec[a_idx] = coeff
            
            # Full V vector (30 entries)
            v_vec = [0.0] * 30
            # Local B index mapping: 0->(0,k1), 1->(0,k2), 2->(1,k1), 3->(1,k2)
            local_to_global = {
                0: 0 * 15 + k1,   # (0, k1)
                1: 0 * 15 + k2,   # (0, k2)
                2: 1 * 15 + k1,   # (1, k1)
                3: 1 * 15 + k2,   # (1, k2)
            }
            for b_idx, coeff in v_dict.items():
                v_vec[local_to_global[b_idx]] = coeff
            
            # Full W vector (30 entries)
            w_vec = [0.0] * 30
            # Local C index mapping: 0->(0,k1), 1->(0,k2), 2->(1,k1), 3->(1,k2)
            local_to_global_c = {
                0: k1 * 2 + 0,    # (0, k1) -> k*n+i = k1*2+0
                1: k2 * 2 + 0,    # (0, k2) -> k2*2+0
                2: k1 * 2 + 1,    # (1, k1) -> k1*2+1
                3: k2 * 2 + 1,    # (1, k2) -> k2*2+1
            }
            for c_idx, c_dict in enumerate(c_comb):
                if t_idx in c_dict:
                    w_vec[local_to_global_c[c_idx]] = c_dict[t_idx]
            
            U_rows.append(u_vec)
            V_rows.append(v_vec)
            W_rows.append(w_vec)
    
    # Single column k=14: trivial 4 terms
    for i in range(2):
        for j in range(2):
            u_vec = [0.0] * 4
            u_vec[i * 2 + j] = 1.0
            v_vec = [0.0] * 30
            v_vec[j * 15 + 14] = 1.0
            w_vec = [0.0] * 30
            w_vec[14 * 2 + i] = 1.0
            U_rows.append(u_vec)
            V_rows.append(v_vec)
            W_rows.append(w_vec)
    
    rank = len(U_rows)
    return {"U": U_rows, "V": V_rows, "W": W_rows}
# EVOLVE-BLOCK-END
