"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Block-structured rank reduction for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Partition the m=4 dimension into 2 blocks of size 2, and the p=8 
    dimension into 4 blocks of size 2. For each (j_block, k_block) pair, we have
    a 2x2 * 2x2 matrix multiplication. Apply Strassen's rank-7 decomposition to
    each 2x2*2x2 sub-problem, yielding total rank = 2 * 4 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Partition m into num_j_blocks blocks of size block_m
    # Partition p into num_k_blocks blocks of size block_p
    num_j_blocks = 2  # m=4 -> 2 blocks of size 2
    num_k_blocks = 4  # p=8 -> 4 blocks of size 2
    block_m = m // num_j_blocks  # 2
    block_p = p // num_k_blocks  # 2
    
    # Strassen's rank-7 decomposition for 2x2 * 2x2 multiplication
    # For matrices A (2x2), B (2x2), C = A*B:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b22)
    # m4 = a22(b12-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b21)
    # m7 = (a12-a22)(b22+b12)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # In tensor decomposition terms for a 2x2 * 2x2 problem:
    # A has entries indexed by (i_local, j_local) where i_local in {0,1}, j_local in {0,1}
    # B has entries indexed by (j_local, k_local)
    # C has entries indexed by (k_local, i_local) per trace convention W[k*n+i]
    
    # For each Strassen multiplier m_l, we define:
    # U factor: linear combination of A entries
    # V factor: linear combination of B entries  
    # W factor: linear combination of C entries
    
    # A entries: (0,0), (0,1), (1,0), (1,1)
    # B entries: (0,0), (0,1), (1,0), (1,1)
    # C entries: (0,0), (0,1), (1,0), (1,1) in (k,i) local indices
    
    # Strassen multipliers as linear forms on A and B:
    strassen_factors = [
        # (U_coeffs for A, V_coeffs for B, W_coeffs for C)
        # U: coefficients for A[0,0], A[0,1], A[1,0], A[1,1]
        # V: coefficients for B[0,0], B[0,1], B[1,0], B[1,1]
        # W: coefficients for C[0,0], C[0,1], C[1,0], C[1,1] where C[k,i]
        #
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B10-B11)
        # m4 = A11(B01-B00)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B10)
        # m7 = (A01-A11)(B11+B01)
        
        # m1
        ([(1, 0, 0, 1),  # A00+A11
          (1, 0, 0, 1)], # B00+B11 -- wait, this is wrong structure
        ),
    ]
    
    # Let me be more careful. The tensor T[i,j,k] = 1 for the bilinear form A[i,j]*B[j,k] = C[i,k]
    # But we're using trace convention W[k*n+i], so C[k,i] = sum_j A[i,j]*B[j,k]
    
    # For 2x2*2x2 with local indices:
    # A: 4 entries: A[0,0], A[0,1], A[1,0], A[1,1] (indexed as (i_local, j_local))
    # B: 4 entries: B[0,0], B[0,1], B[1,0], B[1,1] (indexed as (j_local, k_local))
    # C: 4 entries: C[0,0], C[0,1], C[1,0], C[1,1] (indexed as (k_local, i_local))
    
    # The multiplication tensor for 2x2*2x2 has rank 7 (Strassen).
    # Let me write out the 7 terms explicitly.
    
    # Term l: U_l is a vector in R^4 (coefficients on A entries), 
    #           V_l is a vector in R^4 (coefficients on B entries),
    #           W_l is a vector in R^4 (coefficients on C entries)
    # such that sum_l U_l[a] * V_l[b] * W_l[c] = T[a,b,c]
    
    # A entries ordered: [A00, A01, A10, A11]
    # B entries ordered: [B00, B01, B10, B11]
    # C entries ordered: [C00, C01, C10, C11] where C[k,i]
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen's 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B11+B01)
    
    # Reconstruct:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Now express each as U_l (on A), V_l (on B), W_l (on C):
    
    strassen_A = [
        # P1: A00+A11
        [1, 0, 0, 1],
        # P2: A10+A11
        [0, 0, 1, 1],
        # P3: A00
        [1, 0, 0, 0],
        # P4: A11
        [0, 0, 0, 1],
        # P5: A00+A01
        [1, 1, 0, 0],
        # P6: A10-A00
        [-1, 0, 1, 0],
        # P7: A01-A11
        [0, 1, 0, -1],
    ]
    
    strassen_B = [
        # P1: B00+B11
        [1, 0, 0, 1],
        # P2: B00
        [1, 0, 0, 0],
        # P3: B10-B11
        [0, 0, 1, -1],
        # P4: B01-B00
        [-1, 1, 0, 0],
        # P5: B11
        [0, 0, 0, 1],
        # P6: B00+B10
        [1, 0, 1, 0],
        # P7: B11+B01
        [0, 1, 0, 1],
    ]
    
    strassen_C = [
        # P1: appears in C00 and C11
        [1, 0, 0, 1],
        # P2: appears in C10 and C11 (negative)
        [0, 0, 1, -1],
        # P3: appears in C01 and C11
        [0, 1, 0, 1],
        # P4: appears in C00 and C10
        [1, 0, 1, 0],
        # P5: appears in C00 (negative) and C01
        [-1, 1, 0, 0],
        # P6: appears in C11
        [0, 0, 0, 1],
        # P7: appears in C00
        [1, 0, 0, 0],
    ]
    
    # Total rank
    total_rank = num_j_blocks * num_k_blocks * 7
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for jb in range(num_j_blocks):
        for kb in range(num_k_blocks):
            for l in range(7):
                # U: for each local i in {0,1}, local j in {0,1}
                for li in range(2):
                    for lj in range(2):
                        global_i = li
                        global_j = jb * block_m + lj
                        u_row = global_i * m + global_j
                        coeff = strassen_A[l][lj * 2 + li]  # A[li, lj]
                        if coeff != 0:
                            U[u_row][col] += coeff
                
                # V: for each local j in {0,1}, local k in {0,1}
                for lj in range(2):
                    for lk in range(2):
                        global_j = jb * block_m + lj
                        global_k = kb * block_p + lk
                        v_row = global_j * p + global_k
                        coeff = strassen_B[l][lj * 2 + lk]  # B[lj, lk]
                        if coeff != 0:
                            V[v_row][col] += coeff
                
                # W: for each local k in {0,1}, local i in {0,1}
                for lk in range(2):
                    for li in range(2):
                        global_k = kb * block_p + lk
                        global_i = li
                        w_row = global_k * n + global_i
                        coeff = strassen_C[l][lk * 2 + li]  # C[lk, li]
                        if coeff != 0:
                            W[w_row][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
