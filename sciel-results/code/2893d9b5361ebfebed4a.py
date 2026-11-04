"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8

    # Rank-1 terms: each is (U_idx_list, V_idx_list, W_idx_list)
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k
    # W indices: (k, i) -> k*n + i
    # Coefficients: u_c, v_c, w_c for each index in the list

    # We build the terms as (coeffs, indices) for each factor.
    # A rank-1 term contributes prod_u * prod_v * prod_w to C[i,k].
    # Actually, the decomposition is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For a block algorithm, we can think of it as:
    # C_block = sum_t u_t * v_t * w_t * (sum of A terms) * (sum of B terms) -> no, that's not right.
    # The tensor T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # A decomposition is T = sum_r u_r \otimes v_r \otimes w_r
    # where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    # The entries are u_r[i*m+j], v_r[j*p+k], w_r[k*n+i].
    # The product for a given (i,j,k) is sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # This must equal 1 if (i,j,k) is a valid triple, and 0 otherwise?
    # No, the matrix multiplication tensor is T_{i,j,k} = 1 for all i,j,k.
    # Wait, the tensor for matrix multiplication M_{n,m,p} is defined such that
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is T = sum_{i,j,k} e_{i} \otimes e_{j} \otimes e_{k} where the indices map to
    # A: (i,j), B: (j,k), C: (k,i).
    # So T_{(i,j),(j,k),(k,i)} = 1.
    # Any other index combination is 0.
    # So we need sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 for all valid (i,j,k).

    # Let's use a hybrid approach.
    # The dimensions are 3, 4, 8.
    # We can partition the indices into blocks.
    # Since 4 is divisible by 2, and 8 is divisible by 2, we can use 2x2 blocks for the j and k dimensions.
    # But n=3 is not divisible by 2.
    # Let's partition j into 2 blocks of size 2.
    # Let's partition k into 4 blocks of size 2.
    # Let's partition i into... 3 is 1+2. So one block of size 1, one of size 2.
    
    # Block sizes:
    # i: [1, 2]
    # j: [2, 2]
    # k: [2, 2, 2, 2]
    
    # This gives 2 * 2 * 4 = 16 blocks.
    # Block (a, b, c) has dimensions n_a x m_b x p_c.
    # For block (0, b, c): n_a=1. Size 1 x 2 x 2. Rank for 1x2x2 is 2 (schoolbook).
    # For block (1, b, c): n_a=2. Size 2 x 2 x 2. Rank for 2x2x2 is 7 (Strassen).
    
    # Total rank = 8 * (rank for 1x2x2) + 8 * (rank for 2x2x2)
    # Rank for 1x2x2: C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]. 2 terms.
    # Rank for 2x2x2: 7 terms.
    # Total rank = 8 * 2 + 8 * 7 = 16 + 56 = 72.
    # This is much better than 96.

    # Let's implement this.
    
    terms = [] # List of (U_dict, V_dict, W_dict) where dict maps index to coeff.
    
    # Define block starts
    i_starts = [0, 1]
    m_starts = [0, 2]
    p_starts = [0, 2, 4, 6]
    i_sizes = [1, 2]
    m_sizes = [2, 2]
    p_sizes = [2, 2, 2, 2]
    
    # Precompute Strassen coefficients for 2x2x2
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)(B00)
    # P3 = (A00)(B10+B11)
    # P4 = (A01)(B01)
    # P5 = (A00-A11)(B11)
    # P6 = (A10-A00)(B01)
    # P7 = (A01-A10)(B10)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    strassen_terms = []
    # Each term is (u_coeffs, v_coeffs, w_coeffs) for local 2x2x2
    # Local indices: (li, lj) for A, (lj, lk) for B, (lk, li) for C
    # u: {0:1, 2:1}, v: {0:1, 2:1}, w: {0:1, 2:1} -> P1
    # Wait, w is for C. C has indices (li, lk).
    # In the global tensor, W index is (k, i).
    # Local k is lk, local i is li.
    # So W local index is (lk, li).
    
    # P1
    strassen_terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # P2
    strassen_terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}))
    # P3
    strassen_terms.append(({(0,0):1}, {(1,0):1, (1,1):1}, {(0,1):1}))
    # P4
    strassen_terms.append(({(0,1):1}, {(0,1):1}, {(0,0):1, (1,0):1}))
    # P5
    strassen_terms.append(({(0,0):1, (1,1):-1}, {(1,1):1}, {(0,1):1, (1,0):-1}))
    # P6
    strassen_terms.append(({(1,0):1, (0,0):-1}, {(0,1):1}, {(1,1):1}))
    # P7
    strassen_terms.append(({(0,1):1, (1,0):-1}, {(1,0):1}, {(0,0):1, (1,1):-1}))
    
    # Wait, I need to be careful with the Strassen coefficients.
    # Let's re-verify P1..P7 and C reconstruction.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's check P1 contribution to C.
    # P1 = (A00+A11)(B00+B11).
    # u = e00 + e11. v = e00 + e11.
    # This term is added to C00 and C11.
    # So w = e00 + e11.
    # In my list above, w for P1 is {(0,0):1, (1,1):1}. Correct.
    
    # P2 = (A10+A11)(B00).
    # u = e10 + e11. v = e00.
    # Contributes to C10.
    # w = e10.
    # In my list: w={(1,0):1}. Correct.
    
    # P3 = A00(B10+B11).
    # u = e00. v = e10 + e11.
    # Contributes to C01.
    # w = e01.
    # In my list: w={(0,1):1}. Correct.
    
    # P4 = A01 B01.
    # u = e01. v = e01.
    # Contributes to C00 and C10.
    # w = e00 + e10.
    # In my list: w={(0,0):1, (1,0):1}. Correct.
    
    # P5 = (A00-A11) B11.
    # u = e00 - e11. v = e11.
    # Contributes to C01 (+1) and C10 (-1).
    # w = e01 - e10.
    # In my list: w={(0,1):1, (1,0):-1}. Correct.
    
    # P6 = (A10-A00) B01.
    # u = e10 - e00. v = e01.
    # Contributes to C11.
    # w = e11.
    # In my list: w={(1,1):1}. Correct.
    
    # P7 = (A01-A10) B10.
    # u = e01 - e10. v = e10.
    # Contributes to C00 (+1) and C11 (-1).
    # w = e00 - e11.
    # In my list: w={(0,0):1, (1,1):-1}. Correct.
    
    # Now iterate over blocks
    for a in range(2): # i blocks
        for b in range(2): # j blocks
            for c in range(4): # k blocks
                i0 = i_starts[a]
                j0 = m_starts[b]
                k0 = p_starts[c]
                ni = i_sizes[a]
                mj = m_sizes[b]
                pk = p_sizes[c]
                
                if ni == 1:
                    # 1x2x2 block
                    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
                    # Term 1: A[0,0]B[0,0] -> C[0,0]
                    # u: {(0,0):1}, v: {(0,0):1}, w: {(0,0):1}
                    terms.append((
                        {(0,0):1},
                        {(0,0):1},
                        {(0,0):1}
                    ))
                    # Term 2: A[0,1]B[1,0] -> C[0,0]
                    # u: {(0,1):1}, v: {(1,0):1}, w: {(0,0):1}
                    terms.append((
                        {(0,1):1},
                        {(1,0):1},
                        {(0,0):1}
                    ))
                else:
                    # 2x2x2 block (Strassen)
                    for u_loc, v_loc, w_loc in strassen_terms:
                        u_g = {}
                        for (li, lj), coeff in u_loc.items():
                            gi = i0 + li
                            gj = j0 + lj
                            u_g[(gi, gj)] = coeff
                        v_g = {}
                        for (lj, lk), coeff in v_loc.items():
                            gj = j0 + lj
                            gk = k0 + lk
                            v_g[(gj, gk)] = coeff
                        w_g = {}
                        for (lk, li), coeff in w_loc.items():
                            gk = k0 + lk
                            gi = i0 + li
                            w_g[(gk, gi)] = coeff
                        terms.append((u_g, v_g, w_g))
    
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    for r, (u_g, v_g, w_g) in enumerate(terms):
        for (gi, gj), coeff in u_g.items():
            U[gi*m + gj][r] = coeff
        for (gj, gk), coeff in v_g.items():
            V[gj*p + gk][r] = coeff
        for (gk, gi), coeff in w_g.items():
            W[gk*n + gi][r] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
