"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition:
    # 7 terms for Strassen on columns 0,1 of B
    # 4 terms for Schoolbook on column 2 of B
    rank = 11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U: i*m + j (i in 0,1; j in 0,1)
    # V: j*p + k (j in 0,1; k in 0,1,2)
    # W: k*n + i (k in 0,1,2; i in 0,1)
    
    # Strassen part (cols 0, 1)
    # A = [[a, b], [c, d]] -> U indices: a=0, b=1, c=2, d=3
    # B = [[e, f], [g, h]] -> V indices: e=0, f=1, g=3, h=4 (since p=3, j=0 -> 0,1; j=1 -> 3,4,5)
    # Note: V index for j=1, k=0 is 1*3+0 = 3. V index for j=1, k=1 is 1*3+1 = 4.
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    # C00 += M1, C11 += M1
    W[0][0] = 1 # k=0, i=0
    W[2][1] = 1 # k=1, i=1 (Index k*n+i: 1*2+1=3? No. W index is k*n+i. k=1, i=1 -> 3. k=0, i=0 -> 0.)
    # Wait, W index: k*n + i.
    # k=0, i=0 -> 0
    # k=1, i=1 -> 1*2+1 = 3
    # So W[0][0] = 1, W[3][0] = 1.
    
    # Let's re-map W indices carefully.
    # W[k*n + i, c]
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    
    # Reset W[0]
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += M2
    W[2][1] = 1 # k=0, i=1 -> 1? No. C10 is i=1, k=0. W index 0*2+1 = 1.
    W[1][1] = 1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    # C01 += M3
    W[1][2] = 1 # k=1, i=0 -> 1*2+0 = 2? No. C01 is i=0, k=1. W index 1*2+0 = 2.
    W[2][2] = 1
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    # C00 -= M4, C10 += M4
    W[0][3] = -1
    W[1][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    # C01 += M5
    W[2][4] = 1
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C00 += M6
    W[0][5] = 1
    
    # M7 = (d-c)(g+h)
    U[3][6] = 1; U[2][6] = -1
    V[3][6] = 1; V[4][6] = 1
    # C00 -= M7, C11 += M7
    W[0][6] = -1
    W[3][6] = 1
    
    # Schoolbook part (col 2)
    # C02 = a*g2 + b*j2 (g2 is B[0,2], j2 is B[1,2])
    # V indices for k=2: j=0 -> 2, j=1 -> 5.
    # W indices for k=2: i=0 -> 4, i=1 -> 5.
    
    # Term 7: a * B[0,2]
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Term 8: b * B[1,2]
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Term 9: c * B[0,2]
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Term 10: d * B[1,2]
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
