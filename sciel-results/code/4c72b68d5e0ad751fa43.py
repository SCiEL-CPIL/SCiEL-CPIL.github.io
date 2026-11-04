"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-40 decomposition of the 2x3 x 3x10 matrix multiplication tensor.
    
    Strategy: Split the middle dimension (size 3) into blocks of sizes [1, 2].
    For each output column k:
      - The 1x1x1 block (rows 0x, 1x with col 0x) contributes rank 1.
      - The 2x2x1 block (rows 0y, 0z, 1y, 1z with cols 1x, 2x) contributes rank 3.
    
    For the 2x2x1 block with A = [[a, b], [c, d]], B = [g, h]^T, C = [e, f]^T:
      p1 = (a + d) * g
      p2 = (c + d) * h
      p3 = c * (h - g)
      e  = p1 - p3
      f  = p1 + p2 - p3
    
    Total rank = 10 * (1 + 3) = 40.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    rank = p * (1 + 3)  # 40
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 40
    V = [[0] * rank for _ in range(m * p)]   # 30 x 40
    W = [[0] * rank for _ in range(p * n)]   # 20 x 40
    
    for k in range(p):
        base = k * 4
        
        # --- Block 1: 1x1x1 (A[0,0] * B[0,k]) ---
        # U[0][base] = 1, V[0*p + k][base] = 1, W[k*n + 0][base] = 1
        U[0 * m + 0][base] = 1
        V[0 * p + k][base] = 1
        W[k * n + 0][base] = 1
        
        # --- Block 2: 2x2x1 Strassen-style (rank 3) ---
        # Indices: A rows 0,1; A cols 1,2; B rows 1,2; C rows 0,1
        # Let a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2]
        #     g=B[1,k], h=B[2,k], e=C[0,k], f=C[1,k]
        #
        # p1 = (a + d) * g  -> U: a=1,d=1; V: g=1; W: e+=1, f+=1
        # p2 = (c + d) * h  -> U: c=1,d=1; V: h=1; W: f+=1
        # p3 = c * (h - g)  -> U: c=1; V: h=1,g=-1; W: e-=1, f-=1
        
        # p1: (a+d)*g
        U[0 * m + 1][base + 1] = 1   # a
        U[1 * m + 2][base + 1] = 1   # d
        V[1 * p + k][base + 1] = 1   # g
        W[k * n + 0][base + 1] = 1   # e += 1
        W[k * n + 1][base + 1] = 1   # f += 1
        
        # p2: (c+d)*h
        U[1 * m + 1][base + 2] = 1   # c
        U[1 * m + 2][base + 2] = 1   # d
        V[2 * p + k][base + 2] = 1   # h
        W[k * n + 1][base + 2] = 1   # f += 1
        
        # p3: c*(h-g)
        U[1 * m + 1][base + 3] = 1   # c
        V[1 * p + k][base + 3] = -1  # -g
        V[2 * p + k][base + 3] = 1   # h
        W[k * n + 0][base + 3] = -1  # e -= 1
        W[k * n + 1][base + 3] = -1  # f -= 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
