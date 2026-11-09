import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Total rank 30
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x8 using 4 blocks of 2x2x2 (Strassen)
    # j indices 0, 1
    # k indices 0..7
    # i indices 0, 1
    
    strassen_coeffs = [
        # M0: A00 * (B00 + B11)
        [(0, 0), (0, 0), (0, 0), 1],
        [(0, 0), (1, 1), (0, 0), 1],
        # M1: A10 * (B00 + B01)
        [(1, 0), (0, 0), (0, 0), 1],
        [(1, 0), (0, 1), (0, 0), 1],
        # M2: (A00 + A01) * B11
        [(0, 0), (1, 1), (1, 1), 1],
        [(0, 1), (1, 1), (1, 1), 1],
        # M3: (A10 + A11) * B00
        [(1, 0), (0, 0), (0, 0), 1], # Wait, M3 is (A10+A11)B00. 
        # Let's check Strassen formula carefully.
        # C00 = M0 - M1 + M3 + M5
        # C01 = M2 + M4
        # C10 = M1 + M4 - M6
        # C11 = M0 - M2 + M3 + M6
        
        # M0 = A00(B00+B11)
        # M1 = A10(B00+B01)
        # M2 = (A00+A01)B11
        # M3 = (A10+A11)B00
        # M4 = (A00+A11)(B10+B11)
        # M5 = (A01-A10)(B00+B10)
        # M6 = (A01-A11)(B01+B11)
        
        # We need to decompose each multiplication into rank-1 terms.
        # A multiplication X * Y where X is sum of A terms, Y is sum of B terms.
        # Each product term A_ij * B_kl is a rank-1 tensor.
    ]
    
    # It is easier to just iterate over the 7 Strassen multiplications
    # and for each, expand the sums.
    
    # Define the 7 multiplications as (list of A terms, list of B terms)
    # A terms are (i, j)
    # B terms are (j, k)
    
    mults = [
        # M0
        ([ (0,0) ], [ (0,0), (1,1) ]),
        # M1
        ([ (1,0) ], [ (0,0), (0,1) ]),
        # M2
        ([ (0,0), (0,1) ], [ (1,1) ]),
        # M3
        ([ (1,0), (1,1) ], [ (0,0) ]),
        # M4
        ([ (0,0), (1,1) ], [ (1,0), (1,1) ]),
        # M5
        ([ (0,1), (-1, 1) ], [ (0,0), (1,0) ]), # A01 - A10
        # M6
        ([ (0,1), (-1, 1) ], [ (0,1), (1,1) ])  # A01 - A11 -> (0,1) - (1,1) is wrong indices for A. A is 2x2. A01 is (0,1). A11 is (1,1).
    ]
    
    # Correction for M5, M6 A terms:
    # M5: (A01 - A10). A01 is (0,1). A10 is (1,0).
    # So terms are (0,1) with coeff 1, (1,0) with coeff -1.
    # M6: (A01 - A11). A01 is (0,1). A11 is (1,1).
    # So terms are (0,1) with coeff 1, (1,1) with coeff -1.
    
    # Let's redefine mults with coefficients
    mults = [
        # M0
        ([ ((0,0), 1) ], [ ((0,0), 1), ((1,1), 1) ]),
        # M1
        ([ ((1,0), 1) ], [ ((0,0), 1), ((0,1), 1) ]),
        # M2
        ([ ((0,0), 1), ((0,1), 1) ], [ ((1,1), 1) ]),
        # M3
        ([ ((1,0), 1), ((1,1), 1) ], [ ((0,0), 1) ]),
        # M4
        ([ ((0,0), 1), ((1,1), 1) ], [ ((1,0), 1), ((1,1), 1) ]),
        # M5
        ([ ((0,1), 1), ((1,0), -1) ], [ ((0,0), 1), ((1,0), 1) ]),
        # M6
        ([ ((0,1), 1), ((1,1), -1) ], [ ((0,1), 1), ((1,1), 1) ])
    ]
    
    col = 0
    for b in range(4):
        for a_terms, b_terms in mults:
            for (i_a, j_a), c_a in a_terms:
                for (j_b, k_b), c_b in b_terms:
                    # Check consistency j_a == j_b
                    if j_a != j_b:
                        continue
                    # Global indices
                    i = i_a
                    j = j_a
                    k = 2 * b + k_b
                    
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    coeff = c_a * c_b
                    U[u_idx][col] += coeff
                    V[v_idx][col] += coeff
                    W[w_idx][col] += coeff
                    col += 1
    # col should be 28 here
    
    # Part 2: 2x1x8 using rank 2
    # j = 2
    # r = 0 (local) -> i=0
    # r = 1 (local) -> i=1
    
    # Term 1: i=0, j=2
    # U[0*3+2, col] = 1
    # V[2*8+k, col] = 1 for all k
    # W[k*2+0, col] = 1 for all k
    for k in range(p):
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
    col += 1
    
    # Term 2: i=1, j=2
    # U[1*3+2, col] = 1
    # V[2*8+k, col] = 1 for all k
    # W[k*2+1, col] = 1 for all k
    for k in range(p):
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
