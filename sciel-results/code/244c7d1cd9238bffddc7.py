"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 is the theoretical minimum for 2x2 x 2x8 using 4 blocks of 2x2
    # Each 2x2 block requires rank 7 (Strassen)
    num_blocks = p // 2  # 4 blocks
    rank = num_blocks * 7  # 28
    
    # U: (n*m, rank) = (4, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (16, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen formulas for 2x2:
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
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
    
    # Indices for A: 0:a, 1:b, 2:c, 3:d
    # Indices for B block i: 
    #   e: 0*p + 2i
    #   f: 0*p + 2i + 1
    #   g: 1*p + 2i
    #   h: 1*p + 2i + 1
    # Indices for C block i (row 0 and 1):
    #   C[0, 2i]: 2i * n + 0 = 2i
    #   C[0, 2i+1]: 2i * n + 1 = 2i + 1
    #   C[1, 2i]: (2i+1) * n + 0 = 2i + 2
    #   C[1, 2i+1]: (2i+1) * n + 1 = 2i + 3
    
    for b in range(num_blocks):
        base = b * 7
        
        # M1 = (A[0]+A[3]) * (B_e + B_h)
        # U: A[0]+=1, A[3]+=1
        U[0][base + 0] += 1
        U[3][base + 0] += 1
        # V: B_e+=1, B_h+=1
        V[0 * p + 2 * b][base + 0] += 1
        V[1 * p + 2 * b + 1][base + 0] += 1
        # W: C00+=1, C11+=1
        W[2 * b * n + 0][base + 0] += 1
        W[(2 * b + 1) * n + 1][base + 0] += 1
        
        # M2 = (A[2]+A[3]) * B_e
        U[2][base + 1] += 1
        U[3][base + 1] += 1
        V[0 * p + 2 * b][base + 1] += 1
        # W: C10+=1, C11-=1
        W[2 * b * n + 1][base + 1] += 1
        W[(2 * b + 1) * n + 1][base + 1] -= 1
        
        # M3 = A[0] * (B_f - B_h)
        U[0][base + 2] += 1
        V[0 * p + 2 * b + 1][base + 2] += 1
        V[1 * p + 2 * b + 1][base + 2] -= 1
        # W: C01+=1, C11+=1
        W[(2 * b) * n + 1][base + 2] += 1
        W[(2 * b + 1) * n + 1][base + 2] += 1
        
        # M4 = A[3] * (B_g - B_e)
        U[3][base + 3] += 1
        V[1 * p + 2 * b][base + 3] += 1
        V[0 * p + 2 * b][base + 3] -= 1
        # W: C00+=1, C10+=1
        W[2 * b * n + 0][base + 3] += 1
        W[2 * b * n + 1][base + 3] += 1
        
        # M5 = (A[0]+A[1]) * B_h
        U[0][base + 4] += 1
        U[1][base + 4] += 1
        V[1 * p + 2 * b + 1][base + 4] += 1
        # W: C00-=1, C10+=1
        W[2 * b * n + 0][base + 4] -= 1
        W[2 * b * n + 1][base + 4] += 1
        
        # M6 = (A[2]-A[0]) * (B_e + B_f)
        U[2][base + 5] += 1
        U[0][base + 5] -= 1
        V[0 * p + 2 * b][base + 5] += 1
        V[0 * p + 2 * b + 1][base + 5] += 1
        # W: C11+=1
        W[(2 * b + 1) * n + 1][base + 5] += 1
        
        # M7 = (A[1]-A[3]) * (B_g + B_h)
        U[1][base + 6] += 1
        U[3][base + 6] -= 1
        V[1 * p + 2 * b][base + 6] += 1
        V[1 * p + 2 * b + 1][base + 6] += 1
        # W: C00+=1
        W[2 * b * n + 0][base + 6] += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
