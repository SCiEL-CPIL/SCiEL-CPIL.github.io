import time
import sys

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define blocks
    # Block q: 
    # A cols: a_start + {0,1}
    # B rows: b_row_start + {0,1}
    # B cols: b_col_start + {0,1}
    # C rows: {0,1} (since n=2)
    # C cols: c_col_start + {0,1}
    
    blocks = [
        # (a_col_start, b_row_start, b_col_start, c_col_start)
        (0, 0, 0, 0), # A_L, B_LL -> C_L
        (2, 2, 0, 0), # A_R, B_RL -> C_L
        (0, 0, 2, 2), # A_L, B_LR -> C_R
        (2, 2, 2, 2), # A_R, B_RR -> C_R
    ]
    
    col = 0
    for a_cs, brs, bcs, ccs in blocks:
        # Local 2x2 indices
        # a = A[0, a_cs], b = A[0, a_cs+1], c = A[1, a_cs], d = A[1, a_cs+1]
        # e = B[brs, bcs], f = B[brs, bcs+1], g = B[brs+1, bcs], h = B[brs+1, bcs+1]
        
        # M1
        U[0 * m + a_cs][col] += 1
        U[1 * m + a_cs + 1][col] += 1 # d is A[1, a_cs+1]
        V[brs * p + bcs][col] += 1
        V[(brs+1) * p + bcs + 1][col] += 1 # h is B[brs+1, bcs+1]
        # C00 (i=0, k=ccs) -> W[ccs * n + 0]
        # C11 (i=1, k=ccs+1) -> W[(ccs+1) * n + 1]
        W[ccs * n + 0][col] += 1
        W[(ccs+1) * n + 1][col] += 1
        col += 1
        
        # M2
        U[1 * m + a_cs][col] += 1 # c
        U[1 * m + a_cs + 1][col] += 1 # d
        V[brs * p + bcs][col] += 1 # e
        # C10 (i=1, k=ccs) -> W[ccs * n + 1]
        # C11 (i=1, k=ccs+1) -> W[(ccs+1) * n + 1]
        W[ccs * n + 1][col] += 1
        W[(ccs+1) * n + 1][col] += -1
        col += 1
        
        # M3
        U[0 * m + a_cs][col] += 1 # a
        V[brs * p + bcs + 1][col] += 1 # f
        V[(brs+1) * p + bcs + 1][col] += -1 # h
        # C01 (i=0, k=ccs+1) -> W[(ccs+1) * n + 0]
        # C11 (i=1, k=ccs+1) -> W[(ccs+1) * n + 1]
        W[(ccs+1) * n + 0][col] += 1
        W[(ccs+1) * n + 1][col] += 1
        col += 1
        
        # M4
        U[1 * m + a_cs + 1][col] += 1 # d
        V[(brs+1) * p + bcs][col] += 1 # g
        V[brs * p + bcs][col] += -1 # e
        # C00 (i=0, k=ccs) -> W[ccs * n + 0]
        # C10 (i=1, k=ccs) -> W[ccs * n + 1]
        W[ccs * n + 0][col] += 1
        W[ccs * n + 1][col] += 1
        col += 1
        
        # M5
        U[0 * m + a_cs][col] += 1 # a
        U[0 * m + a_cs + 1][col] += 1 # b
        V[(brs+1) * p + bcs + 1][col] += 1 # h
        # C00 (i=0, k=ccs) -> W[ccs * n + 0]
        # C01 (i=0, k=ccs+1) -> W[(ccs+1) * n + 0]
        W[ccs * n + 0][col] += -1
        W[(ccs+1) * n + 0][col] += 1
        col += 1
        
        # M6
        U[1 * m + a_cs][col] += 1 # c
        U[0 * m + a_cs][col] += -1 # a
        V[brs * p + bcs][col] += 1 # e
        V[brs * p + bcs + 1][col] += 1 # f
        # C11 (i=1, k=ccs+1) -> W[(ccs+1) * n + 1]
        W[(ccs+1) * n + 1][col] += 1
        col += 1
        
        # M7
        U[0 * m + a_cs + 1][col] += 1 # b
        U[1 * m + a_cs + 1][col] += -1 # d
        V[(brs+1) * p + bcs][col] += 1 # g
        V[(brs+1) * p + bcs + 1][col] += 1 # h
        # C00 (i=0, k=ccs) -> W[ccs * n + 0]
        # C11 (i=1, k=ccs+1) -> W[(ccs+1) * n + 1]
        W[ccs * n + 0][col] += 1
        W[(ccs+1) * n + 1][col] += -1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
