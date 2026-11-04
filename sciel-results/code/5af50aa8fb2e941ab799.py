"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 56 using recursive Strassen decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A11-A01)(B10+B11)
    
    # A indices: (i, j) -> 0:00, 1:01, 2:10, 3:11
    # B indices: (j, k) -> 0:00, 1:01, 2:10, 3:11
    # C indices: (k, i) -> 0:00, 1:01, 2:10, 3:11
    
    # Define the 7 terms as (coeffs_A, coeffs_B, coeffs_C)
    # coeffs are dicts mapping (local_idx) to value
    # Local A idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local C idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 0, 2: 0, 3: 0}), # C00 += M1
        # M2
        ({2: 1, 3: 1}, {0: 1}, {0: 0, 1: 0, 2: 1, 3: 0}), # C10 += M2, C11 -= M2
        # M3
        ({0: 1}, {1: 1, 3: -1}, {0: 0, 1: 1, 2: 0, 3: 1}), # C01 += M3, C11 += M3
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 0, 2: 1, 3: 0}), # C00 += M4, C10 += M4
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1, 2: 0, 3: 0}), # C00 -= M5, C01 += M5
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {0: 0, 1: 0, 2: 0, 3: 1}), # C11 += M6
        # M7
        ({3: 1, 1: -1}, {2: 1, 3: 1}, {0: 1, 1: 0, 2: 0, 3: 0}), # C00 += M7
    ]
    
    # Wait, the C coefficients in the tuple above are for the specific term's contribution.
    # Let's re-verify the C accumulation.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My tuples above:
    # Term 0 (M1): C00 += 1. Correct.
    # Term 1 (M2): C10 += 1, C11 -= 1. Correct.
    # Term 2 (M3): C01 += 1, C11 += 1. Correct.
    # Term 3 (M4): C00 += 1, C10 += 1. Correct.
    # Term 4 (M5): C00 -= 1, C01 += 1. Correct.
    # Term 5 (M6): C11 += 1. Correct.
    # Term 6 (M7): C00 += 1. Correct.
    
    col = 0
    # Iterate over the 8 blocks
    # b = k1 * 4 + k2p * 2 + j1
    for b in range(8):
        k1 = b // 4
        k2p = (b % 4) // 2
        j1 = b % 2
        
        for term in strassen_terms:
            coeffs_A, coeffs_B, coeffs_C = term
            
            # Process A
            for local_idx, coeff in coeffs_A.items():
                if coeff == 0: continue
                # local_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                i = local_idx // 2
                j_loc = local_idx % 2
                j_glob = 2 * j1 + j_loc
                I_A = i * m + j_glob
                U[I_A][col] += coeff
                
            # Process B
            for local_idx, coeff in coeffs_B.items():
                if coeff == 0: continue
                j_loc = local_idx // 2
                k_loc = local_idx % 2
                j_glob = 2 * j1 + j_loc
                k_glob = 4 * k1 + 2 * k2p + k_loc
                I_B = j_glob * p + k_glob
                V[I_B][col] += coeff
                
            # Process C
            for local_idx, coeff in coeffs_C.items():
                if coeff == 0: continue
                k_loc = local_idx // 2
                i = local_idx % 2
                k_glob = 4 * k1 + 2 * k2p + k_loc
                I_C = k_glob * n + i
                W[I_C][col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
