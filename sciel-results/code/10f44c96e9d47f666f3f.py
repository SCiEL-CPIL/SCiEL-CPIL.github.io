import time
import sys

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add to U
    def add_U(i, j, col, val):
        U[i * m + j][col] += val
        
    # Helper to add to V
    def add_V(j, k, col, val):
        V[j * p + k][col] += val
        
    # Helper to add to W
    def add_W(k, i, col, val):
        W[k * n + i][col] += val
        
    for g in range(3):
        col_base = g * 11
        
        # Strassen Part (7 terms)
        # M1: (a00 + a11)(b00 + b11)
        col = col_base
        add_U(0, 0, col, 1)
        add_U(1, 1, col, 1)
        add_V(0, 2*g, col, 1)
        add_V(1, 2*g+1, col, 1)
        add_W(2*g, 0, col, 1)
        add_W(2*g+1, 1, col, 1)
        
        # M2: (a10 + a11)b00
        col = col_base + 1
        add_U(1, 0, col, 1)
        add_U(1, 1, col, 1)
        add_V(0, 2*g, col, 1)
        add_W(2*g, 1, col, 1)
        add_W(2*g+1, 1, col, -1)
        
        # M3: a00(b01 - b11)
        col = col_base + 2
        add_U(0, 0, col, 1)
        add_V(0, 2*g+1, col, 1)
        add_V(1, 2*g+1, col, -1)
        add_W(2*g+1, 0, col, 1)
        add_W(2*g+1, 1, col, 1)
        
        # M4: a11(b10 - b00)
        col = col_base + 3
        add_U(1, 1, col, 1)
        add_V(1, 2*g, col, 1)
        add_V(0, 2*g, col, -1)
        add_W(2*g, 0, col, 1)
        add_W(2*g, 1, col, 1)
        
        # M5: (a00 + a01)b11
        col = col_base + 4
        add_U(0, 0, col, 1)
        add_U(0, 1, col, 1)
        add_V(1, 2*g+1, col, 1)
        add_W(2*g, 0, col, -1)
        add_W(2*g+1, 0, col, 1)
        
        # M6: (a10 - a00)(b00 + b01)
        col = col_base + 5
        add_U(1, 0, col, 1)
        add_U(0, 0, col, -1)
        add_V(0, 2*g, col, 1)
        add_V(0, 2*g+1, col, 1)
        add_W(2*g+1, 1, col, 1)
        
        # M7: (a01 - a11)(b10 + b11)
        col = col_base + 6
        add_U(0, 1, col, 1)
        add_U(1, 1, col, -1)
        add_V(1, 2*g, col, 1)
        add_V(1, 2*g+1, col, 1)
        add_W(2*g, 0, col, 1)
        add_W(2*g+1, 1, col, 1)
        
        # Rank-4 Part (4 terms)
        # T1: a02 * b20 -> C00
        col = col_base + 7
        add_U(0, 2, col, 1)
        add_V(2, 2*g, col, 1)
        add_W(2*g, 0, col, 1)
        
        # T2: a02 * b21 -> C01
        col = col_base + 8
        add_U(0, 2, col, 1)
        add_V(2, 2*g+1, col, 1)
        add_W(2*g+1, 0, col, 1)
        
        # T3: a12 * b20 -> C10
        col = col_base + 9
        add_U(1, 2, col, 1)
        add_V(2, 2*g, col, 1)
        add_W(2*g, 1, col, 1)
        
        # T4: a12 * b21 -> C11
        col = col_base + 10
        add_U(1, 2, col, 1)
        add_V(2, 2*g+1, col, 1)
        add_W(2*g+1, 1, col, 1)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
