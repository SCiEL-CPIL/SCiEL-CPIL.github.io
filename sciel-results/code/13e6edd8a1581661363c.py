"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_u(i, j, c, val):
        U[i * m + j][c] += val
        
    def add_v(j, k, c, val):
        V[j * p + k][c] += val
        
    def add_w(k, i, c, val):
        W[k * n + i][c] += val

    # Block 1: k in {0, 1, 2}
    # Sub-block 1: 2x2x2 Strassen for j in {0,1}, k in {0,1}
    c = 0
    # M1 = (A00 + A11)(B00 + B11)
    add_u(0, 0, c, 1); add_u(1, 1, c, 1)
    add_v(0, 0, c, 1); add_v(1, 1, c, 1)
    add_w(0, 0, c, 1); add_w(1, 1, c, 1)
    c += 1
    
    # M2 = (A10 + A11)B00
    add_u(1, 0, c, 1); add_u(1, 1, c, 1)
    add_v(0, 0, c, 1)
    add_w(0, 1, c, 1); add_w(1, 1, c, -1)
    c += 1
    
    # M3 = A00(B01 - B11)
    add_u(0, 0, c, 1)
    add_v(0, 1, c, 1); add_v(1, 1, c, -1)
    add_w(0, 1, c, 1); add_w(1, 1, c, 1)
    c += 1
    
    # M4 = A11(B10 + B11)
    add_u(1, 1, c, 1)
    add_v(1, 0, c, 1); add_v(1, 1, c, 1)
    add_w(0, 0, c, 1); add_w(0, 1, c, 1)
    c += 1
    
    # M5 = (A00 + A01)B11
    add_u(0, 0, c, 1); add_u(0, 1, c, 1)
    add_v(1, 1, c, 1)
    add_w(0, 0, c, -1); add_w(0, 1, c, 1)
    c += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    add_u(1, 0, c, 1); add_u(0, 0, c, -1)
    add_v(0, 0, c, 1); add_v(0, 1, c, 1)
    add_w(1, 1, c, 1)
    c += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    add_u(0, 1, c, 1); add_u(1, 1, c, -1)
    add_v(1, 0, c, 1); add_v(1, 1, c, 1)
    add_w(0, 0, c, 1)
    c += 1
    
    # Sub-block 2: j=2, k in {0, 1}
    # Term: A_i2 B_2k
    add_u(0, 2, c, 1); add_u(1, 2, c, 1)
    add_v(2, 0, c, 1); add_v(2, 1, c, 1)
    add_w(0, 0, c, 1); add_w(0, 1, c, 1)
    add_w(1, 0, c, 1); add_w(1, 1, c, 1)
    c += 1
    
    # Sub-block 3: k=2
    # Term 1: A_i0 B_02
    add_u(0, 0, c, 1); add_u(1, 0, c, 1)
    add_v(0, 2, c, 1)
    add_w(2, 0, c, 1); add_w(2, 1, c, 1)
    c += 1
    
    # Term 2: A_i1 B_12
    add_u(0, 1, c, 1); add_u(1, 1, c, 1)
    add_v(1, 2, c, 1)
    add_w(2, 0, c, 1); add_w(2, 1, c, 1)
    c += 1
    
    # Term 3: A_i2 B_22
    add_u(0, 2, c, 1); add_u(1, 2, c, 1)
    add_v(2, 2, c, 1)
    add_w(2, 0, c, 1); add_w(2, 1, c, 1)
    c += 1
    
    # Block 2: k in {3, 4, 5}
    # Sub-block 1: 2x2x2 Strassen for j in {0,1}, k in {3,4}
    # M1
    add_u(0, 0, c, 1); add_u(1, 1, c, 1)
    add_v(0, 3, c, 1); add_v(1, 4, c, 1)
    add_w(3, 0, c, 1); add_w(4, 1, c, 1)
    c += 1
    
    # M2
    add_u(1, 0, c, 1); add_u(1, 1, c, 1)
    add_v(0, 3, c, 1)
    add_w(3, 1, c, 1); add_w(4, 1, c, -1)
    c += 1
    
    # M3
    add_u(0, 0, c, 1)
    add_v(0, 4, c, 1); add_v(1, 4, c, -1)
    add_w(3, 1, c, 1); add_w(4, 1, c, 1)
    c += 1
    
    # M4
    add_u(1, 1, c, 1)
    add_v(1, 3, c, 1); add_v(1, 4, c, 1)
    add_w(3, 0, c, 1); add_w(3, 1, c, 1)
    c += 1
    
    # M5
    add_u(0, 0, c, 1); add_u(0, 1, c, 1)
    add_v(1, 4, c, 1)
    add_w(3, 0, c, -1); add_w(3, 1, c, 1)
    c += 1
    
    # M6
    add_u(1, 0, c, 1); add_u(0, 0, c, -1)
    add_v(0, 3, c, 1); add_v(0, 4, c, 1)
    add_w(4, 1, c, 1)
    c += 1
    
    # M7
    add_u(0, 1, c, 1); add_u(1, 1, c, -1)
    add_v(1, 3, c, 1); add_v(1, 4, c, 1)
    add_w(3, 0, c, 1)
    c += 1
    
    # Sub-block 2: j=2, k in {3, 4}
    add_u(0, 2, c, 1); add_u(1, 2, c, 1)
    add_v(2, 3, c, 1); add_v(2, 4, c, 1)
    add_w(3, 0, c, 1); add_w(3, 1, c, 1)
    add_w(4, 0, c, 1); add_w(4, 1, c, 1)
    c += 1
    
    # Sub-block 3: k=5
    # Term 1: A_i0 B_05
    add_u(0, 0, c, 1); add_u(1, 0, c, 1)
    add_v(0, 5, c, 1)
    add_w(5, 0, c, 1); add_w(5, 1, c, 1)
    c += 1
    
    # Term 2: A_i1 B_15
    add_u(0, 1, c, 1); add_u(1, 1, c, 1)
    add_v(1, 5, c, 1)
    add_w(5, 0, c, 1); add_w(5, 1, c, 1)
    c += 1
    
    # Term 3: A_i2 B_25
    add_u(0, 2, c, 1); add_u(1, 2, c, 1)
    add_v(2, 5, c, 1)
    add_w(5, 0, c, 1); add_w(5, 1, c, 1)
    c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
