"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A1 B1 using Strassen
    # A1 is 2x2 (indices i in 0,1; j in 0,1)
    # B1 is 2x10 (indices j in 0,1; k in 0..9)
    # Split B1 into 5 blocks of 2x2
    
    # Strassen U coefficients for A
    # m=0: a+d
    # m=1: c+d
    # m=2: a
    # m=3: d
    # m=4: a-b
    # m=5: c-a
    # m=6: b-d
    
    U_coeffs = [
        [(0,0,1), (1,0,1)],       # a+d
        [(1,0,1), (1,1,1)],       # c+d
        [(0,0,1)],                # a
        [(1,1,1)],                # d
        [(0,0,1), (0,1,-1)],      # a-b
        [(1,0,1), (0,0,-1)],      # c-a
        [(0,1,1), (1,1,-1)]       # b-d
    ]
    
    # Strassen W coefficients for C
    # m=0: C00 += 1, C11 += 1
    # m=1: C10 += 1
    # m=2: C01 += 1, C11 += -1
    # m=3: C00 += -1, C01 += 1, C10 += 1
    # m=4: C00 += 1
    # m=5: C11 += 1
    # m=6: C00 += 1, C11 += 1
    
    W_coeffs = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1)],                # M2
        [(0,1,1), (1,1,-1)],      # M3
        [(0,0,-1), (0,1,1), (1,0,1)], # M4
        [(0,0,1)],                # M5
        [(1,1,1)],                # M6
        [(0,0,1), (1,1,1)]        # M7
    ]
    
    col = 0
    for g in range(5):
        k1 = 2 * g
        k2 = 2 * g + 1
        
        # V coefficients for B
        # m=0: b11+b22 -> V[0,k1]+=1, V[1,k2]+=1
        # m=1: b11 -> V[0,k1]+=1
        # m=2: b12-b22 -> V[0,k2]+=1, V[1,k2]+=-1
        # m=3: b21-b22 -> V[1,k1]+=1, V[1,k2]+=-1
        # m=4: b21+b22 -> V[1,k1]+=1, V[1,k2]+=1
        # m=5: b11+b21 -> V[0,k1]+=1, V[1,k1]+=1
        # m=6: b12+b21 -> V[0,k2]+=1, V[1,k1]+=1
        
        V_coeffs = [
            [(0, k1, 1), (1, k2, 1)],
            [(0, k1, 1)],
            [(0, k2, 1), (1, k2, -1)],
            [(1, k1, 1), (1, k2, -1)],
            [(1, k1, 1), (1, k2, 1)],
            [(0, k1, 1), (1, k1, 1)],
            [(0, k2, 1), (1, k1, 1)]
        ]
        
        for m in range(7):
            # Fill U
            for i, j, val in U_coeffs[m]:
                U[i * m + j][col] += val
            
            # Fill V
            for j, k, val in V_coeffs[m]:
                V[j * p + k][col] += val
            
            # Fill W
            for i, k, val in W_coeffs[m]:
                # W index is k * n + i
                W[k * n + i][col] += val
            
            col += 1
    
    # Part 2: A2 B2
    # A2 is 2x1 (indices i in 0,1; j=2)
    # B2 is 1x10 (indices j=2; k in 0..9)
    
    for k in range(10):
        # Term 1: A[0,2] * B[2,k] -> C[0,k]
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: A[1,2] * B[2,k] -> C[1,k]
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
