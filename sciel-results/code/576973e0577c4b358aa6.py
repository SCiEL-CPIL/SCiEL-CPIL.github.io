import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0

    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add Strassen decomposition for 2x2 * 2x2 -> 2x2.
        A at rows [a_r0, a_r0+1], cols [a_c0, a_c0+1]
        B at rows [b_r0, b_r0+1], cols [b_c0, b_c0+1]
        C at rows [c_r0, c_r0+1], cols [c_c0, c_c0+1]
        """
        nonlocal col
        if time.time() > deadline:
            return
        
        # U indices: row*m + col
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        # V indices: row*p + col
        v00 = b_r0 * p + b_c0
        v01 = b_r0 * p + b_c0 + 1
        v10 = (b_r0 + 1) * p + b_c0
        v11 = (b_r0 + 1) * p + b_c0 + 1
        
        # W indices: col*n + row (trace convention W[k*n+i, r])
        w00 = c_c0 * n + c_r0
        w01 = (c_c0 + 1) * n + c_r0
        w10 = c_c0 * n + c_r0 + 1
        w11 = (c_c0 + 1) * n + c_r0 + 1
        
        # M1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00 -> c10 += 1, c11 -= 1
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M3 = a00(b01-b11) -> c01 += 1, c11 += 1
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = a11(b10-b00) -> c00 += 1, c10 += 1
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11 -> c00 -= 1, c01 += 1
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01) -> c11 += 1
        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11) -> c00 += 1
        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    def add_schoolbook_2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add schoolbook decomposition for 2x2 * 2x1 -> 2x1.
        A at rows [a_r0, a_r0+1], cols [a_c0, a_c0+1]
        B at rows [b_r0, b_r0+1], col b_c0
        C at rows [c_r0, c_r0+1], col c_c0
        """
        nonlocal col
        if time.time() > deadline:
            return
        
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        v0 = b_r0 * p + b_c0
        v1 = (b_r0 + 1) * p + b_c0
        
        w0 = c_c0 * n + c_r0
        w1 = c_c0 * n + c_r0 + 1
        
        # A00*B0 -> C0
        U[u00][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        col += 1
        # A01*B1 -> C0
        U[u01][col] = 1; V[v1][col] = 1; W[w0][col] = 1
        col += 1
        # A10*B0 -> C1
        U[u10][col] = 1; V[v0][col] = 1; W[w1][col] = 1
        col += 1
        # A11*B1 -> C1
        U[u11][col] = 1; V[v1][col] = 1; W[w1][col] = 1
        col += 1

    # Decompose C = A*B where A is 2x4, B is 4x5, C is 2x5.
    # Split B into B1 (4x2, cols 0-1) and B2 (4x3, cols 2-4).
    # C1 = A*B1 (2x2): Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    #   C1 = A1*B1_top + A2*B1_bot
    #   A1*B1_top: A rows 0-1 cols 0-1, B rows 0-1 cols 0-1 -> C rows 0-1 cols 0-1
    add_strassen(0, 0, 0, 0, 0, 0)
    #   A2*B1_bot: A rows 0-1 cols 2-3, B rows 2-3 cols 0-1 -> C rows 0-1 cols 0-1
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # C2 = A*B2 (2x3): Split B2 into B2a (4x1, col 2) and B2b (4x2, cols 3-4).
    # C2a = A*B2a (2x1): Split A into A1 (cols 0-1) and A2 (cols 2-3).
    #   C2a = A1*B2a_top + A2*B2a_bot
    #   A1*B2a_top: A rows 0-1 cols 0-1, B rows 0-1 col 2 -> C rows 0-1 col 2
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
    #   A2*B2a_bot: A rows 0-1 cols 2-3, B rows 2-3 col 2 -> C rows 0-1 col 2
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
    
    # C2b = A*B2b (2x2): Split A into A1 (cols 0-1) and A2 (cols 2-3).
    #   C2b = A1*B2b_top + A2*B2b_bot
    #   A1*B2b_top: A rows 0-1 cols 0-1, B rows 0-1 cols 3-4 -> C rows 0-1 cols 3-4
    add_strassen(0, 0, 0, 3, 0, 3)
    #   A2*B2b_bot: A rows 0-1 cols 2-3, B rows 2-3 cols 3-4 -> C rows 0-1 cols 3-4
    add_strassen(0, 2, 2, 3, 0, 3)
    
    # If deadline exceeded, we might have incomplete matrices.
    # However, the construction is static and fast, so this shouldn't happen.
    # We return the constructed matrices.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
