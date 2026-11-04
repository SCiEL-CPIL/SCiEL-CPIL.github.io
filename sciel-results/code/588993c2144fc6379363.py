import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 27 = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 blocks
    # mu[a][b][c] is coeff of A_ab in X_c
    # nu[a][b][c] is coeff of B_ab in Y_c
    # a, b in {0, 1}
    # c in {0..6}
    
    mu = [[0]*7 for _ in range(2)]
    nu = [[0]*7 for _ in range(2)]
    
    # c=0: (A00+A11)(B00+B11)
    mu[0][0] = 1; mu[1][1] = 1
    nu[0][0] = 1; nu[1][1] = 1
    
    # c=1: (A10+A11)B00
    mu[1][0] = 1; mu[1][1] = 1
    nu[0][0] = 1
    
    # c=2: A00(B01-B11)
    mu[0][0] = 1
    nu[0][1] = 1; nu[1][1] = -1
    
    # c=3: A11(B10-B00)
    mu[1][1] = 1
    nu[1][0] = 1; nu[0][0] = -1
    
    # c=4: (A00+A01)B11
    mu[0][0] = 1; mu[0][1] = 1
    nu[1][1] = 1
    
    # c=5: (A10-A00)(B00+B01)
    mu[1][0] = 1; mu[0][0] = -1
    nu[0][0] = 1; nu[0][1] = 1
    
    # c=6: (A01-A11)(B10+B11)
    mu[0][1] = 1; mu[1][1] = -1
    nu[1][0] = 1; nu[1][1] = 1
    
    # 3x3 schoolbook decomposition factors
    # Size 9x27
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                d = i * 9 + j * 3 + k
                U3[i * 3 + j][d] = 1
                V3[j * 3 + k][d] = 1
                W3[k * 3 + i][d] = 1
                
    # Construct global U, V, W
    # Global index for A: i * 6 + j
    # i = 3*a + r, j = 3*b + s
    # Global index for B: j * 6 + k
    # j = 3*b + s, k = 3*e + t
    # Global index for C: k * 6 + i
    # k = 3*e + t, i = 3*a + r
    
    for c in range(7):
        for d in range(27):
            col = c * 27 + d
            
            # Iterate over all global elements to fill U, V, W
            # It's more efficient to iterate over local indices and compute global
            for a in range(2):
                for b in range(2):
                    for r in range(3):
                        for s in range(3):
                            # Global index for A_ab,rs
                            i_global = 3 * a + r
                            j_global = 3 * b + s
                            idx_U = i_global * 6 + j_global
                            
                            U[idx_U][col] = mu[a][b][c] * U3[r * 3 + s][d]
                            
            for b in range(2):
                for e in range(2):
                    for s in range(3):
                        for t in range(3):
                            # Global index for B_be,st
                            j_global = 3 * b + s
                            k_global = 3 * e + t
                            idx_V = j_global * 6 + k_global
                            
                            V[idx_V][col] = nu[b][e][c] * V3[s * 3 + t][d]
                            
            for e in range(2):
                for a in range(2):
                    for t in range(3):
                        for r in range(3):
                            # Global index for C_ea,tr (which is C_ik with i=3a+r, k=3e+t)
                            k_global = 3 * e + t
                            i_global = 3 * a + r
                            idx_W = k_global * 6 + i_global
                            
                            W[idx_W][col] = W3[t * 3 + r][d]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
