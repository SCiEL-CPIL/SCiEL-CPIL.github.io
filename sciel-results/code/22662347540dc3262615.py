import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen algorithm for 2x2 blocks.
    # 6x6 matrix is divided into 3x3 blocks of 2x2 matrices.
    # Each 2x2 multiplication uses 7 rank-1 terms.
    # Total rank = 3 * 3 * 3 * 7 = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = AB
    #
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B11 + B01)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each entry is a list of (local_index, coefficient) for U, V, W
    # local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # M1
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        # M2
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, -1)]
        },
        # M3
        {
            'U': [(0, 1)],
            'V': [(2, 1), (3, -1)],
            'W': [(1, 1), (3, 1)]
        },
        # M4
        {
            'U': [(3, 1)],
            'V': [(1, 1), (0, -1)],
            'W': [(0, 1), (2, 1)]
        },
        # M5
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)]
        },
        # M6
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (2, 1)],
            'W': [(3, 1)]
        },
        # M7
        {
            'U': [(1, 1), (3, -1)],
            'V': [(3, 1), (1, 1)],
            'W': [(0, 1)]
        }
    ]
    
    col = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for term in strassen_terms:
                    # U: A_{ab} block
                    # local (u, v) -> global i = 2a+u, j = 2b+v
                    for (lu, lv), coeff in term['U']:
                        i = 2 * a + lu
                        j = 2 * b + lv
                        U[i * m + j][col] += coeff
                    
                    # V: B_{bc} block
                    # local (u, v) -> global j = 2b+u, k = 2c+v
                    for (lu, lv), coeff in term['V']:
                        j = 2 * b + lu
                        k = 2 * c + lv
                        V[j * p + k][col] += coeff
                    
                    # W: C_{ac} block
                    # local (u, v) -> global i = 2a+u, k = 2c+v
                    # Note: W index is k*n + i
                    for (lu, lv), coeff in term['W']:
                        i = 2 * a + lu
                        k = 2 * c + lv
                        W[k * n + i][col] += coeff
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
