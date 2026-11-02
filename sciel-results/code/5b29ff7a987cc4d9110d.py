# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Constructs a rank-42 exact tensor decomposition for 2x2 * 2x12 matrix multiplication.
    
    The rank-42 decomposition is formed by applying Strassen's algorithm (rank 7)
    to each of the 6 column pairs of the 2x12 matrix. Since the column pairs
    occupy disjoint slices of the B and C matrices, the ranks add up.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 7 products for 2x2 * 2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = a00 * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = (a00 + a01) * b11
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a11) * (b00 + b11) -- Wait, standard Strassen:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = (a00 + a01)b11
    # M4 = a11(b10 - b00)
    # M5 = a00(b01 - b11)
    # M6 = (a10 + a00)b01
    # M7 = (a01 + a11)(b10 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's use the standard coefficients:
    # U (A factors):
    # M1: a00, a11
    # M2: a10, a11
    # M3: a00, a01
    # M4: a11
    # M5: a00
    # M6: a10, a00
    # M7: a01, a11
    
    # V (B factors):
    # M1: b00, b11
    # M2: b00
    # M3: b11
    # M4: b10, -b00
    # M5: b01, -b11
    # M6: b01
    # M7: b10, b11
    
    # W (C factors):
    # C00: M1, M4, -M5, M7
    # C01: M3, M5
    # C10: M2, M4
    # C11: M1, -M2, M3, M6

    # Indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i

    strassen_products = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: list of (idx, coeff)
        # V: list of (idx, coeff)
        # W: list of (idx, coeff)
        
        # M1
        {
            'U': [((0,0), 1), ((1,1), 1)],
            'V': [((0,0), 1), ((1,1), 1)],
            'W': [((0,0), 1), ((1,0), 1), ((0,1), -1), ((1,1), 1)] # C00: M1, C10: M1? No.
            # C00 = M1 + M4 - M5 + M7 -> W for M1 at C00 is 1
            # C11 = M1 - M2 + M3 + M6 -> W for M1 at C11 is 1
            # So W for M1: C00=1, C11=1
            # Let's re-verify Strassen output contributions.
        }
    ]
    
    # Let's hardcode the 7 Strassen terms carefully.
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    # Term 2: M2 = (A10 + A11)B00
    # Term 3: M3 = (A00 + A01)B11
    # Term 4: M4 = A11(B10 - B00)
    # Term 5: M5 = A00(B01 - B11)
    # Term 6: M6 = (A10 + A00)B01
    # Term 7: M7 = (A01 + A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    terms = [
        { # M1
            'U': [((0,0), 1), ((1,1), 1)],
            'V': [((0,0), 1), ((1,1), 1)],
            'W': [((0,0), 1), ((1,1), 1)] # C00, C11
        },
        { # M2
            'U': [((1,0), 1), ((1,1), 1)],
            'V': [((0,0), 1)],
            'W': [((1,0), 1), ((1,1), -1)] # C10, C11
        },
        { # M3
            'U': [((0,0), 1), ((0,1), 1)],
            'V': [((1,1), 1)],
            'W': [((0,1), 1), ((1,1), 1)] # C01, C11
        },
        { # M4
            'U': [((1,1), 1)],
            'V': [((1,0), 1), ((0,0), -1)],
            'W': [((0,0), 1), ((1,0), 1)] # C00, C10
        },
        { # M5
            'U': [((0,0), 1)],
            'V': [((0,1), 1), ((1,1), -1)],
            'W': [((0,0), -1), ((0,1), 1)] # C00, C01
        },
        { # M6
            'U': [((1,0), 1), ((0,0), 1)],
            'V': [((0,1), 1)],
            'W': [((1,1), 1)] # C11
        },
        { # M7
            'U': [((0,1), 1), ((1,1), 1)],
            'V': [((1,0), 1), ((1,1), 1)],
            'W': [((0,0), 1)] # C00
        }
    ]
    
    col = 0
    for t in range(num_pairs):
        base_k = 2 * t
        for s in range(7):
            term = terms[s]
            for (ai, aj), coeff in term['U']:
                U[ai * m + aj, col] = coeff
            for (bj, bk_rel), coeff in term['V']:
                V[bj * p + base_k + bk_rel, col] = coeff
            for (ck_rel, ci), coeff in term['W']:
                # W index is k*n + i
                # k is base_k + ck_rel
                # i is ci
                W[(base_k + ck_rel) * n + ci, col] = coeff
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
