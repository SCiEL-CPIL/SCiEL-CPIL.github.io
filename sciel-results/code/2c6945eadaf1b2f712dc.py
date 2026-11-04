"""Matrix multiplication 2x4 @ 4x8 via Strassen-style block decomposition (rank 56)."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56  # 4 blocks * 14 Strassen terms for 2x2 * 2x4
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 56

    # Strassen 2x2 * 2x4 terms: (u_local, v_local, w_local)
    # u_local: indices into (2,2) = [i*2+j]
    # v_local: indices into (2,4) = [j*4+k]
    # w_local: indices into (4,2) = [k*2+i]
    strassen_terms = [
        # (u, v, w) as lists of (index, coeff)
        ([(0, 1), (1, 1)], [(0, 1), (2, 1), (4, 1), (6, 1)], [(0, 1), (5, 1)]),
        ([(0, 1), (1, 1)], [(1, 1), (3, 1), (5, 1), (7, 1)], [(1, 1), (6, 1)]),
        ([(0, 1), (2, 1)], [(0, 1), (1, 1), (2, 1), (3, 1)], [(0, 1), (4, 1)]),
        ([(0, 1), (2, 1)], [(4, 1), (5, 1), (6, 1), (7, 1)], [(2, 1), (6, 1)]),
        ([(0, 1), (1, 1)], [(0, 1), (1, 1), (4, 1), (5, 1)], [(0, 1), (4, 1), (5, 1)]),
        ([(0, 1), (1, -1)], [(2, 1), (3, 1), (6, 1), (7, 1)], [(1, 1), (5, 1)]),
        ([(1, 1), (2, -1)], [(4, 1), (5, 1), (6, 1), (7, 1)], [(2, 1), (6, 1)]),
        ([(0, 1), (2, 1)], [(0, 1), (4, 1)], [(0, 1)]),
        ([(0, 1), (2, 1)], [(1, 1), (5, 1)], [(1, 1)]),
        ([(1, 1), (3, -1)], [(4, 1), (6, 1)], [(4, 1)]),
        ([(1, 1), (3, -1)], [(5, 1), (7, 1)], [(5, 1)]),
        ([(3, 1)], [(0, 1), (2, 1)], [(4, 1)]),
        ([(3, 1)], [(1, 1), (3, 1)], [(5, 1)]),
        ([(2, 1)], [(4, 1), (6, 1)], [(6, 1)]),
    ]
    
    # Wait, I need to re-derive the Strassen 2x2 * 2x4 decomposition properly.
    # Let A be 2x2, B be 2x4. C = A*B is 2x4.
    # Partition B as [B1 | B2] where B1 is 2x2 (cols 0,1) and B2 is 2x2 (cols 2,3).
    # C = [A*B1 | A*B2].
    # We can use Strassen for A*B1 and A*B2 separately, but that's 7+7=14 mults.
    # Is there a better way? The known rank for 2x2 * 2x4 is 14.
    # Let's use the standard Strassen for the two 2x2 blocks.
    # For A*B1:
    # M1 = (A00+A11)*(B100+B111)
    # M2 = (A10+A11)*B100
    # M3 = A00*(B101-B111)
    # M4 = A11*(B100+B101)
    # M5 = (A00+A10)*(B110+B111)
    # M6 = (A11-A10)*(B101+B111)
    # M7 = (A01-A11)*(B100+B110)
    # C10 = M1 - M5 + M3
    # C11 = M4 + M5
    # C20 = M2 + M7
    # C21 = M1 - M2 + M6
    
    # Let's map indices.
    # A = [[a00, a01], [a10, a11]] (indices 0,1,2,3)
    # B1 = [[b00, b01], [b10, b11]] (indices 0,1,4,5 in local 2x4 if row-major? No, 2x4 row-major: [b00,b01,b02,b03, b10,b11,b12,b13])
    # Wait, my previous v_local indices were 0..7 for 2x4.
    # Let's stick to row-major for 2x4:
    # v[0]=b00, v[1]=b01, v[2]=b02, v[3]=b03
    # v[4]=b10, v[5]=b11, v[6]=b12, v[7]=b13
    
    # B1 = [[b00, b01], [b10, b11]] -> indices 0,1,4,5
    # B2 = [[b02, b03], [b12, b13]] -> indices 2,3,6,7
    
    # Strassen for A*B1 (using v indices 0,1,4,5):
    # M1 = (a00+a11) * (b00+b11)
    # M2 = (a10+a11) * b00
    # M3 = a00 * (b01-b11)
    # M4 = a11 * (b00+b01)
    # M5 = (a00+a10) * (b10+b11)
    # M6 = (a11-a10) * (b01+b11)
    # M7 = (a01-a11) * (b00+b10)
    # C10 = M1 - M5 + M3
    # C11 = M4 + M5
    # C20 = M2 + M7
    # C21 = M1 - M2 + M6
    
    # Strassen for A*B2 (using v indices 2,3,6,7):
    # N1 = (a00+a11) * (b02+b13)
    # N2 = (a10+a11) * b02
    # N3 = a00 * (b03-b13)
    # N4 = a11 * (b02+b03)
    # N5 = (a00+a10) * (b12+b13)
    # N6 = (a11-a10) * (b03+b13)
    # N7 = (a01-a11) * (b02+b12)
    # C12 = N1 - N5 + N3
    # C13 = N4 + N5
    # C22 = N2 + N7
    # C23 = N1 - N2 + N6
    
    # Total 14 terms.
    # Let's rewrite strassen_terms correctly.
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (5, 1)], [(0, 1), (5, 1)]), # C10, C21
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(4, 1), (5, -1)]), # C20, -C21
        # M3
        ([(0, 1)], [(1, 1), (5, -1)], [(0, 1)]), # C10
        # M4
        ([(3, 1)], [(0, 1), (1, 1)], [(1, 1)]), # C11
        # M5
        ([(0, 1), (2, 1)], [(4, 1), (5, 1)], [(0, -1), (1, 1)]), # -C10, C11
        # M6
        ([(3, 1), (2, -1)], [(1, 1), (5, 1)], [(5, 1)]), # C21
        # M7
        ([(1, 1), (3, -1)], [(0, 1), (4, 1)], [(4, 1)]), # C20
        # N1
        ([(0, 1), (3, 1)], [(2, 1), (7, 1)], [(2, 1), (7, 1)]), # C12, C23
        # N2
        ([(2, 1), (3, 1)], [(2, 1)], [(6, 1), (7, -1)]), # C22, -C23
        # N3
        ([(0, 1)], [(3, 1), (7, -1)], [(2, 1)]), # C12
        # N4
        ([(3, 1)], [(2, 1), (3, 1)], [(3, 1)]), # C13
        # N5
        ([(0, 1), (2, 1)], [(6, 1), (7, 1)], [(2, -1), (3, 1)]), # -C12, C13
        # N6
        ([(3, 1), (2, -1)], [(3, 1), (7, 1)], [(7, 1)]), # C23
        # N7
        ([(1, 1), (3, -1)], [(2, 1), (6, 1)], [(6, 1)]), # C22
    ]
    
    # Blocks:
    # Block 0: i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # Block 1: i in {0,1}, j in {0,1}, k in {4,5,6,7}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1,2,3}
    # Block 3: i in {0,1}, j in {2,3}, k in {4,5,6,7}
    
    blocks = [
        (0, 0, 0), # i_offset, j_offset, k_offset
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 4),
    ]
    
    col = 0
    for i_off, j_off, k_off in blocks:
        for t in range(14):
            u_list, v_list, w_list = strassen_terms[t]
            for u_idx, coeff in u_list:
                i_loc, j_loc = divmod(u_idx, 2)
                i_g = i_off + i_loc
                j_g = j_off + j_loc
                U[i_g * m + j_g][col] = coeff
            for v_idx, coeff in v_list:
                j_loc, k_loc = divmod(v_idx, 4)
                j_g = j_off + j_loc
                k_g = k_off + k_loc
                V[j_g * p + k_g][col] = coeff
            for w_idx, coeff in w_list:
                k_loc, i_loc = divmod(w_idx, 2)
                k_g = k_off + k_loc
                i_g = i_off + i_loc
                W[k_g * n + i_g][col] = coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
