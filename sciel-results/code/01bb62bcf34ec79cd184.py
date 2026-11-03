import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (8, rank), V: (32, rank), W: (16, rank)
    
    # Strassen rank for 2x2x2 is 7
    # 8 sub-problems (2 J-groups x 4 K-groups)
    # Total rank = 8 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Predefine Strassen coefficients for A, B, W
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Local indices in 2x2 A: a(0,0), b(0,1), c(1,0), d(1,1)
    # Local indices in 2x2 B: e(0,0), f(0,1), g(1,0), h(1,1)
    # Local indices in 2x2 C: C00(0,0), C01(0,1), C10(1,0), C11(1,1)
    
    # Structure:
    # A_local[i, j_loc] -> Global A index: i*m + (2*J + j_loc)
    # B_local[j_loc, k_loc] -> Global B index: (2*J + j_loc)*p + (2*K + k_loc)
    # C_local[i, k_loc] -> Global W index: (2*K + k_loc)*n + i
    
    # Strassen steps definition:
    # Each step has:
    # u_coeffs: list of (i, j_loc, coeff)
    # v_coeffs: list of (j_loc, k_loc, coeff)
    # w_coeffs: list of (i, k_loc, coeff)
    
    steps = [
        { # M1
            'u': [(0, 0, 1), (1, 1, 1)], # a, d
            'v': [(0, 0, 1), (1, 1, 1)], # e, h
            'w': [(0, 0, 1), (1, 1, 1)]  # C00, C11
        },
        { # M2
            'u': [(1, 0, 1), (1, 1, 1)], # c, d
            'v': [(0, 0, 1)],            # e
            'w': [(1, 0, 1)]             # C10
        },
        { # M3
            'u': [(0, 0, 1)],            # a
            'v': [(0, 1, 1), (1, 1, -1)],# f, -h
            'w': [(0, 1, 1), (1, 1, 1)]  # C01, C11
        },
        { # M4
            'u': [(1, 1, 1)],            # d
            'v': [(1, 0, 1), (0, 0, -1)],# g, -e
            'w': [(0, 0, 1), (1, 0, 1)]  # C00, C10
        },
        { # M5
            'u': [(0, 0, 1), (0, 1, 1)], # a, b
            'v': [(1, 1, 1)],            # h
            'w': [(0, 0, -1), (0, 1, 1)] # -C00, C01
        },
        { # M6
            'u': [(1, 0, 1), (0, 0, -1)],# c, -a
            'v': [(0, 0, 1), (0, 1, 1)], # e, f
            'w': [(1, 1, 1)]             # C11
        },
        { # M7
            'u': [(0, 1, 1), (1, 1, -1)],# b, -d
            'v': [(1, 0, 1), (1, 1, 1)], # g, h
            'w': [(0, 0, 1)]             # C00
        }
    ]
    
    for J in range(2):
        for K in range(4):
            base_col = (J * 4 + K) * 7
            for s in range(7):
                col = base_col + s
                step = steps[s]
                
                # Fill U
                for i, j_loc, coeff in step['u']:
                    u_idx = i * m + (2 * J + j_loc)
                    U[u_idx][col] = coeff
                    
                # Fill V
                for j_loc, k_loc, coeff in step['v']:
                    v_idx = (2 * J + j_loc) * p + (2 * K + k_loc)
                    V[v_idx][col] = coeff
                    
                # Fill W
                for i, k_loc, coeff in step['w']:
                    w_idx = (2 * K + k_loc) * n + i
                    W[w_idx][col] = coeff
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
