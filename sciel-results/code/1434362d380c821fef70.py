"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 28 (4 blocks of 2x2x2 Strassen)
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u[i,j,r], v[j,k,r], w[i,k,r]
    # r is 0-indexed 0..6
    # i, j, k are 0..1
    
    # P1: (a+d)(e+h)
    # u: (1,0,1,0) for a,b,c,d -> u[0,0]=1, u[1,1]=1
    # v: (1,0,0,1) for e,f,g,h -> v[0,0]=1, v[1,1]=1
    # w: C00+=1, C11+=1 -> w[0,0]=1, w[1,1]=1
    # P2: (c+d)e
    # u: c=1, d=1 -> u[1,0]=1, u[1,1]=1
    # v: e=1 -> v[0,0]=1
    # w: C10+=1, C11-=-1 -> w[1,0]=1, w[1,1]=-1
    # P3: a(f-h)
    # u: a=1 -> u[0,0]=1
    # v: f=1, h=-1 -> v[0,1]=1, v[1,1]=-1
    # w: C01+=1, C11+=1 -> w[0,1]=1, w[1,1]=1
    # P4: d(g-e)
    # u: d=1 -> u[1,1]=1
    # v: g=1, e=-1 -> v[1,0]=1, v[0,0]=-1
    # w: C00+=1, C10+=1 -> w[0,0]=1, w[1,0]=1
    # P5: (a+b)h
    # u: a=1, b=1 -> u[0,0]=1, u[0,1]=1
    # v: h=1 -> v[1,1]=1
    # w: C00-=-1, C01+=1 -> w[0,0]=-1, w[0,1]=1
    # P6: (c-a)(e+f)
    # u: c=1, a=-1 -> u[1,0]=1, u[0,0]=-1
    # v: e=1, f=1 -> v[0,0]=1, v[0,1]=1
    # w: C11+=1 -> w[1,1]=1
    # P7: (b-d)(g+h)
    # u: b=1, d=-1 -> u[0,1]=1, u[1,1]=-1
    # v: g=1, h=1 -> v[1,0]=1, v[1,1]=1
    # w: C00+=1 -> w[0,0]=1

    # Store in lists for easy access
    # u_coeffs[r][i][j]
    u_coeffs = [
        [[1, 0], [0, 1]], # r0 (P1)
        [[0, 0], [1, 1]], # r1 (P2)
        [[1, 0], [0, 0]], # r2 (P3)
        [[0, 0], [0, 1]], # r3 (P4)
        [[1, 1], [0, 0]], # r4 (P5)
        [[-1, 0], [1, 0]],# r5 (P6)
        [[0, 1], [0, -1]] # r6 (P7)
    ]
    
    # v_coeffs[r][j][k]
    v_coeffs = [
        [[1, 0], [0, 1]], # r0
        [[1, 0], [0, 0]], # r1
        [[0, 1], [0, -1]],# r2
        [[-1, 0], [1, 0]],# r3
        [[0, 0], [0, 1]], # r4
        [[1, 1], [0, 0]], # r5
        [[0, 0], [1, 1]]  # r6
    ]
    
    # w_coeffs[r][i][k]
    w_coeffs = [
        [[1, 0], [0, 1]], # r0
        [[0, 0], [1, -1]],# r1
        [[0, 1], [0, 1]], # r2
        [[1, 0], [1, 0]], # r3
        [[-1, 1], [0, 0]],# r4
        [[0, 0], [0, 1]], # r5
        [[1, 0], [0, 0]]  # r6
    ]

    num_blocks = 4
    
    for block in range(num_blocks):
        for r in range(7):
            col = block * 7 + r
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    val = u_coeffs[r][i][j]
                    if val != 0:
                        U[i * m + j][col] = val
                        
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    val = v_coeffs[r][j][k_local]
                    if val != 0:
                        k_global = block * 2 + k_local
                        V[j * p + k_global][col] = val
                        
            # Fill W
            for i in range(2):
                for k_local in range(2):
                    val = w_coeffs[r][i][k_local]
                    if val != 0:
                        k_global = block * 2 + k_local
                        W[k_global * n + i][col] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
