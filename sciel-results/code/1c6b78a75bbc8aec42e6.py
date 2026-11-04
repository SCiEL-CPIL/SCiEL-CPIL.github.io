import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size 2x2, 3x3 blocks
    # Total rank 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1: (x00+x11)(y00+y11)
    # M2: (x10+x11)y00
    # M3: x00(y01-y11)
    # M4: x11(y10-y00)
    # M5: (x00+x01)y11
    # M6: (x10-x00)(y00+y01)
    # M7: (x01-x11)(y10+y11)
    
    # Trace term: tr(P Z^T)
    # P00 = M1+M4-M5+M7
    # P01 = M3+M5
    # P10 = M2+M4
    # P11 = M1-M2+M3+M6
    # tr = P00 Z00 + P01 Z10 + P10 Z01 + P11 Z11
    # = M1(Z00+Z11) + M2(Z01-Z11) + M3(Z10+Z11) + M4(Z00+Z01) + M5(-Z00+Z10) + M6(Z11) + M7(Z00)
    
    # Define the 7 terms as lists of (u_idx, v_idx, coeff) for U, (v_idx, w_idx, coeff) for V, (w_idx, u_idx, coeff) for W
    # Note: U index is (I*2+u, J*2+v). V index is (J*2+v, K*2+w). W index is (K*2+w, I*2+u).
    
    terms = [
        # M1
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # M2
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(0, 1, 1), (1, 1, -1)] # Z01 - Z11
        },
        # M3
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)], # Y01 - Y11
            'W': [(1, 0, 1), (1, 1, 1)] # Z10 + Z11
        },
        # M4
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)], # Y10 - Y00
            'W': [(0, 0, 1), (0, 1, 1)] # Z00 + Z01
        },
        # M5
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (1, 0, 1)] # -Z00 + Z10
        },
        # M6
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)] # Z11
        },
        # M7
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)] # Z00
        }
    ]
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for t in terms:
                    # Add U contributions
                    for u, v, c in t['U']:
                        i = I * 2 + u
                        j = J * 2 + v
                        U[i * m + j][col] += c
                    
                    # Add V contributions
                    for v, w, c in t['V']:
                        j = J * 2 + v
                        k = K * 2 + w
                        V[j * p + k][col] += c
                    
                    # Add W contributions
                    for w, u, c in t['W']:
                        k = K * 2 + w
                        i = I * 2 + u
                        W[k * n + i][col] += c
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
