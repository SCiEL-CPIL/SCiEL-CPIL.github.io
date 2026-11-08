"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 140
    
    n, m, p = DIMENSIONS
    
    # Use 2x2 blocks of 3x3
    # Rank of 2x2 matmul (Strassen) = 7
    # Rank of 3x3 matmul (Bayer) = 23
    # Total rank = 7 * 23 = 161
    
    block_size = 3
    blocks_per_dim = 2
    
    rank = 7 * 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 block multiplication
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
    
    strassen_coeffs = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: {(I,J): val}, B_coeffs: {(I,J): val}, C_coeffs: {(I,J): val}
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1}), # Wait, C10 gets M4. C10 is (1,0).
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1}), # C00 gets -M5.
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # C11 gets M6.
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # C00 gets M7, C11 gets M7?
    ]
    
    # Correct Strassen C coefficients:
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    
    # Let's redefine properly
    # Term 0 (M1): A: {(0,0):1, (1,1):1}, B: {(0,0):1, (1,1):1}, C: {(0,0):1, (1,1):1}
    # Term 1 (M2): A: {(1,0):1, (1,1):1}, B: {(0,0):1}, C: {(1,0):1, (1,1):-1}
    # Term 2 (M3): A: {(0,0):1}, B: {(0,1):1, (1,1):-1}, C: {(0,1):1, (1,1):1}
    # Term 3 (M4): A: {(1,1):1}, B: {(1,0):1, (0,0):-1}, C: {(0,0):1, (1,0):1}
    # Term 4 (M5): A: {(0,0):1, (0,1):1}, B: {(1,1):1}, C: {(0,0):-1, (0,1):1}
    # Term 5 (M6): A: {(1,0):1, (0,0):-1}, B: {(0,0):1, (0,1):1}, C: {(1,1):1}
    # Term 6 (M7): A: {(0,1):1, (1,1):-1}, B: {(1,0):1, (1,1):1}, C: {(0,0):1, (1,1):1}
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    # Bayer 23 terms for 3x3
    # This is a placeholder for the 23 terms.
    # If not available, use 27 trivial terms.
    # For now, use 27 trivial terms to ensure correctness, rank = 7 * 27 = 189.
    # To improve, we need the 23 terms.
    # I will use 27 for safety.
    
    trivial_3x3_terms = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                u = [0]*9; v = [0]*9; w = [0]*9
                u[i*3+k] = 1
                v[k*3+j] = 1
                w[i*3+j] = 1
                trivial_3x3_terms.append((u, v, w))
                
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                if time.time() > deadline:
                    break
                # This loop structure is incorrect for Strassen.
                # Strassen applies to the 2x2 block grid, not per block.
                # The previous code was iterating I,J,K for 3x3 grid.
                # For 2x2 grid, we just iterate over the 7 Strassen terms.
                
    # Correct approach for 2x2 grid of 3x3 blocks:
    # We have 7 Strassen terms.
    # Each term involves linear combinations of blocks.
    # Each block multiplication is a 3x3 matmul.
    # We decompose each 3x3 matmul into 27 terms (or 23).
    
    U = [[0] * (7 * 27) for _ in range(n * m)]
    V = [[0] * (7 * 27) for _ in range(m * p)]
    W = [[0] * (7 * 27) for _ in range(p * n)]
    
    col_idx = 0
    for s_term in strassen_terms:
        a_coeffs, b_coeffs, c_coeffs = s_term
        for u_flat, v_flat, w_flat in trivial_3x3_terms:
            # Add to U
            for pi in range(2):
                for pj in range(2):
                    if (pi, pj) in a_coeffs:
                        val = a_coeffs[(pi, pj)]
                        # u_flat is for the 3x3 block at (pi, pj)
                        for li in range(3):
                            for lj in range(3):
                                if u_flat[li*3+lj] != 0:
                                    r = pi*3 + li
                                    c = pj*3 + lj
                                    U[r*m + c][col_idx] += val * u_flat[li*3+lj]
            # Add to V
            for pi in range(2):
                for pj in range(2):
                    if (pi, pj) in b_coeffs:
                        val = b_coeffs[(pi, pj)]
                        for li in range(3):
                            for lj in range(3):
                                if v_flat[li*3+lj] != 0:
                                    r = pi*3 + li
                                    c = pj*3 + lj
                                    V[r*p + c][col_idx] += val * v_flat[li*3+lj]
            # Add to W
            for pi in range(2):
                for pj in range(2):
                    if (pi, pj) in c_coeffs:
                        val = c_coeffs[(pi, pj)]
                        for li in range(3):
                            for lj in range(3):
                                if w_flat[li*3+lj] != 0:
                                    r = pi*3 + li
                                    c = pj*3 + lj
                                    W[r*n + c][col_idx] += val * w_flat[li*3+lj]
            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
