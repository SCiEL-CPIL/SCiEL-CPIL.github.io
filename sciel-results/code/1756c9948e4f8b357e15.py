"""Matrix multiplication 2x4x8 rank-minimal decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    
    # Strassen 2x2x2 terms
    strassen = [
        # M1: (a00+a11)(b00+b11) -> c00+=M1, c11+=M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (a10+a11)b00 -> c10+=M2, c11-=M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: a00(b01-b11) -> c01+=M3, c11+=M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4: a11(b10-b00) -> c00+=M4, c10+=M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (a00+a01)b11 -> c00-=M5, c01+=M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (a10-a00)(b00+b01) -> c11+=M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (a01-a11)(b10+b11) -> c00+=M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for a_block in range(2):       # A has 2 blocks of 2 columns
        for b_block in range(4):   # B has 4 blocks of 2 columns
            for A_c, B_c, C_c in strassen:
                # Fill U
                for (li, lj), coeff in A_c.items():
                    U[li * m + 2*a_block + lj, col] += coeff
                # Fill V
                for (lj, lk), coeff in B_c.items():
                    V[(2*a_block + lj) * p + 2*b_block + lk, col] += coeff
                # Fill W
                for (li, lk), coeff in C_c.items():
                    W[(2*b_block + lk) * n + li, col] += coeff
                col += 1
    
    # Verify
    U_arr = U
    V_arr = V
    W_arr = W
    ok = True
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # Fallback: schoolbook rank 64
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
