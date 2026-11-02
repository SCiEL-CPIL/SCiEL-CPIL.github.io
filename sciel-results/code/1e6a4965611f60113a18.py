import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    # A (2x4) = [A1, A2] (2x2 each)
    # B (4x5) = [B1; B2] (2x5 each)
    # B1 = [B1a, B1b, B1c] (2x2, 2x2, 2x1)
    # B2 = [B2a, B2b, B2c] (2x2, 2x2, 2x1)
    # C = A1 B1 + A2 B2
    # C = [A1 B1a + A2 B2a, A1 B1b + A2 B2b, A1 B1c + A2 B2c]
    # C is 2x5.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(u_offset, v_offset, w_offset):
        nonlocal col
        # M1
        U[u_offset + 0][col] = 1
        U[u_offset + 3][col] = 1
        V[v_offset + 0][col] = 1
        V[v_offset + 3][col] = 1
        W[w_offset + 0][col] = 1
        W[w_offset + 1][col] = 1
        W[w_offset + 2][col] = 1
        W[w_offset + 3][col] = 1
        col += 1
        
        # M2
        U[u_offset + 2][col] = 1
        U[u_offset + 3][col] = 1
        V[v_offset + 0][col] = 1
        W[w_offset + 0][col] = 1
        W[w_offset + 2][col] = -1
        col += 1
        
        # M3
        U[u_offset + 0][col] = 1
        V[v_offset + 2][col] = 1
        V[v_offset + 3][col] = -1
        W[w_offset + 1][col] = 1
        W[w_offset + 3][col] = 1
        col += 1
        
        # M4
        U[u_offset + 3][col] = 1
        V[v_offset + 1][col] = 1
        V[v_offset + 0][col] = -1
        W[w_offset + 0][col] = -1
        W[w_offset + 2][col] = -1
        col += 1
        
        # M5
        U[u_offset + 0][col] = 1
        U[u_offset + 1][col] = 1
        V[v_offset + 3][col] = 1
        W[w_offset + 1][col] = -1
        W[w_offset + 3][col] = -1
        col += 1
        
        # M6
        U[u_offset + 2][col] = 1
        U[u_offset + 0][col] = -1
        V[v_offset + 0][col] = 1
        V[v_offset + 2][col] = 1
        W[w_offset + 0][col] = 1
        W[w_offset + 2][col] = 1
        col += 1
        
        # M7
        U[u_offset + 1][col] = 1
        U[u_offset + 3][col] = -1
        V[v_offset + 1][col] = 1
        V[v_offset + 3][col] = 1
        W[w_offset + 1][col] = 1
        W[w_offset + 3][col] = 1
        col += 1

    def add_2x2x1(u_offset, v_offset, w_offset):
        nonlocal col
        # Schoolbook 2x2x1 (Rank 4)
        # M1: A00 B0
        U[u_offset + 0][col] = 1
        V[v_offset + 0][col] = 1
        W[w_offset + 0][col] = 1
        col += 1
        
        # M2: A01 B1
        U[u_offset + 1][col] = 1
        V[v_offset + 1][col] = 1
        W[w_offset + 0][col] = 1
        col += 1
        
        # M3: A10 B0
        U[u_offset + 2][col] = 1
        V[v_offset + 0][col] = 1
        W[w_offset + 1][col] = 1
        col += 1
        
        # M4: A11 B1
        U[u_offset + 3][col] = 1
        V[v_offset + 1][col] = 1
        W[w_offset + 1][col] = 1
        col += 1

    # A1 B1a (2x2x2)
    # A1: U 0-3. B1a: V 0-3. C_L: W 0-3.
    add_strassen(0, 0, 0)
    
    # A2 B2a (2x2x2)
    # A2: U 4-7. B2a: V 10-13. C_L: W 0-3.
    add_strassen(4, 10, 0)
    
    # A1 B1b (2x2x2)
    # A1: U 0-3. B1b: V 4-7. C_M: W 4-7.
    add_strassen(0, 4, 4)
    
    # A2 B2b (2x2x2)
    # A2: U 4-7. B2b: V 14-17. C_M: W 4-7.
    add_strassen(4, 14, 4)
    
    # A1 B1c (2x2x1)
    # A1: U 0-3. B1c: V 8-9. C_R: W 8-9.
    add_2x2x1(0, 8, 8)
    
    # A2 B2c (2x2x1)
    # A2: U 4-7. B2c: V 18-19. C_R: W 8-9.
    add_2x2x1(4, 18, 8)
    
    return {"U": U, "V": V, "W": W}
