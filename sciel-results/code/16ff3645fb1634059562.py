"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    terms = []  # Each term: (U_row, V_row, W_row, coeff)
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append((i * m + j, j * p + k, k * n + i, 1))
    
    def add_strassen(i0, j0, k0):
        """Add 7 Strassen rank-1 terms for 2x2x2 block at (i0,j0,k0)."""
        # A indices: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
        # B indices: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
        # C indices: (i0,k0), (i0,k0+1), (i0+1,k0), (i0+1,k0+1)
        
        # u indexed as (A00,A01,A10,A11), v as (B00,B01,B10,B11), w as (C00,C01,C10,C11)
        # Strassen's 7 terms:
        strassen_terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            ((1,0,0,1), (1,0,0,1), (1,0,0,1)),  # M1
            ((0,0,1,1), (1,0,0,0), (0,0,1,0)),  # M2
            ((1,0,0,0), (0,1,0,-1), (0,0,0,1)), # M3
            ((0,0,0,1), (0,1,-1,0), (0,1,0,0)), # M4
            ((1,1,0,0), (0,0,0,1), (0,0,1,0)),  # M5
            ((0,0,1,-1), (1,0,1,0), (0,0,0,1)), # M6
            ((1,0,-1,0), (0,1,0,1), (0,1,0,0)), # M7
        ]
        
        # A index mapping: (0,0)->(i0,j0), (0,1)->(i0,j0+1), (1,0)->(i0+1,j0), (1,1)->(i0+1,j0+1)
        a_map = [(i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)]
        # B index mapping: (0,0)->(j0,k0), (0,1)->(j0,k0+1), (1,0)->(j0+1,k0), (1,1)->(j0+1,k0+1)
        b_map = [(j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)]
        # C index mapping: (0,0)->(i0,k0), (0,1)->(i0,k0+1), (1,0)->(i0+1,k0), (1,1)->(i0+1,k0+1)
        c_map = [(i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Expand this rank-1 term into individual contributions
            # For each combination of (a_idx, b_idx, c_idx) where u[a_idx]*v[b_idx]*w[c_idx] != 0
            for a_i in range(4):
                if u_coeffs[a_i] == 0:
                    continue
                for b_i in range(4):
                    if v_coeffs[b_i] == 0:
                        continue
                    for c_i in range(4):
                        if w_coeffs[c_i] == 0:
                            continue
                        i_a, j_a = a_map[a_i]
                        j_b, k_b = b_map[b_i]
                        i_c, k_c = c_map[c_i]
                        # The rank-1 term contributes to U[i_a*m+j_a], V[j_b*p+k_b], W[k_c*n+i_c]
                        # But wait - this is a single rank-1 term in the tensor, so it's ONE term
                        # with U, V, W having multiple non-zero entries.
                        # Actually no - in the tensor decomposition, each rank-1 term has
                        # u as a vector over all A entries, v over all B entries, w over all C entries.
                        # So this is ONE term with multiple non-zero components.
                        pass
        
        # Actually, each Strassen term is a single rank-1 term where U, V, W have multiple non-zero entries.
        # Let me restructure: for each Strassen term, I add ONE term to the list, but with
        # the understanding that U, V, W will have multiple non-zero entries in that column.
        
        # Let me redo: collect all (U_row, coeff), (V_row, coeff), (W_row, coeff) for each Strassen term
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for a_i in range(4):
                if u_coeffs[a_i] != 0:
                    i_a, j_a = a_map[a_i]
                    u_entries.append((i_a * m + j_a, u_coeffs[a_i]))
            for b_i in range(4):
                if v_coeffs[b_i] != 0:
                    j_b, k_b = b_map[b_i]
                    v_entries.append((j_b * p + k_b, v_coeffs[b_i]))
            for c_i in range(4):
                if w_coeffs[c_i] != 0:
                    i_c, k_c = c_map[c_i]
                    w_entries.append((k_c * n + i_c, w_coeffs[c_i]))
            terms.append(('strassen', u_entries, v_entries, w_entries))
    
    # Add blocks:
    add_strassen(0, 0, 0)   # i{0,1}, j{0,1}, k{0,1}: 7 terms
    add_strassen(0, 0, 2)   # i{0,1}, j{0,1}, k{2,3}: 7 terms
    add_schoolbook(0, 2, 0, 2, 4, 5)  # i{0,1}, j{0,1}, k{4}: 4 terms
    add_strassen(0, 2, 0)   # i{0,1}, j{2,3}, k{0,1}: 7 terms
    add_strassen(0, 2, 2)   # i{0,1}, j{2,3}, k{2,3}: 7 terms
    add_schoolbook(0, 2, 2, 4, 4, 5)  # i{0,1}, j{2,3}, k{4}: 4 terms
    add_schoolbook(2, 3, 0, 2, 0, 2)  # i{2}, j{0,1}, k{0,1}: 4 terms
    add_schoolbook(2, 3, 0, 2, 2, 4)  # i{2}, j{0,1}, k{2,3}: 4 terms
    add_schoolbook(2, 3, 0, 2, 4, 5)  # i{2}, j{0,1}, k{4}: 2 terms
    add_schoolbook(2, 3, 2, 4, 0, 2)  # i{2}, j{2,3}, k{0,1}: 4 terms
    add_schoolbook(2, 3, 2, 4, 2, 4)  # i{2}, j{2,3}, k{2,3}: 4 terms
    add_schoolbook(2, 3, 2, 4, 4, 5)  # i{2}, j{2,3}, k{4}: 2 terms
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        if term[0] == 'strassen':
            _, u_entries, v_entries, w_entries = term
            for row, coeff in u_entries:
                U[row][r] = coeff
            for row, coeff in v_entries:
                V[row][r] = coeff
            for row, coeff in w_entries:
                W[row][r] = coeff
        else:
            u_idx, v_idx, w_idx, coeff = term
            U[u_idx][r] = coeff
            V[v_idx][r] = coeff
            W[w_idx][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
