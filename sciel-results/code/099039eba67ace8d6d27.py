"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
        
    col = 0
        
    # Strassen coefficients for 2x2x2
    # m1: A00:1, A11:1; B00:1, B11:1; C00:1, C11:1
    # m2: A10:1, A11:1; B00:1; C10:1
    # m3: A00:1; B01:1, B11:-1; C01:1
    # m4: A11:1; B10:1, B00:-1; C10:1, C00:-1 (Wait, C00 has +m4? No, C00=m1+m4-m5+m7. C10=m2+m4. C11=m1-m2+m3-m6. C01=m3+m5)
    # Let's use the verified correct coefficients for Strassen:
    # m1: (a+d)(x+w) -> A: [1,1,0,0] (a,d), B: [1,1,0,0] (x,w), C: [1,0,0,1] (c00,c11)
    # m2: (c+d)x -> A: [0,0,1,1] (c,d), B: [1,0,0,0] (x), C: [0,0,1,0] (c10)
    # m3: a(y-w) -> A: [1,0,0,0] (a), B: [0,1,0,-1] (y,w), C: [0,1,0,0] (c01)
    # m4: d(z-x) -> A: [0,0,0,1] (d), B: [0,0,1,-1] (z,x), C: [0,0,1,0] (c10)
    # m5: (a+b)w -> A: [1,1,0,0] (a,b), B: [0,0,0,1] (w), C: [0,1,0,0] (c01)
    # m6: (c-a)(x+z) -> A: [0,0,1,-1] (c,a), B: [1,0,1,0] (x,z), C: [0,0,0,1] (c11)
    # m7: (b-d)(y+w) -> A: [0,1,0,-1] (b,d), B: [0,1,0,1] (y,w), C: [1,0,0,0] (c00)
        
    # Wait, my previous check failed. Let's use a safe 8-term for 2x2x2?
    # No, I'll trust the standard Strassen and assume my manual check had a typo.
    # I will implement the 66-rank structure.
        
    strassen_U = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 1, 0, 0], # m5
        [0, 0, 1, -1], # m6
        [0, 1, 0, -1]  # m7
    ]
    strassen_V = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 1, 0, -1], # m3
        [0, 0, 1, -1], # m4
        [0, 0, 0, 1], # m5
        [1, 0, 1, 0], # m6
        [0, 1, 0, 1]  # m7
    ]
    strassen_W = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 0], # m2
        [0, 1, 0, 0], # m3
        [0, 0, 1, 0], # m4
        [0, 1, 0, 0], # m5
        [0, 0, 0, 1], # m6
        [1, 0, 0, 0]  # m7
    ]
        
    # C indices: 00, 01, 10, 11
    # W coefficients for each term:
    # m1: C00=1, C11=1
    # m2: C10=1
    # m3: C01=1
    # m4: C10=1
    # m5: C01=1
    # m6: C11=1
    # m7: C00=1
        
    # This W matrix above is NOT the combination coefficients.
    # The W matrix in the tensor decomposition represents the coefficient of the output.
    # For a rank-1 term, W is a vector.
    # Here, each Strassen term is a rank-1 tensor? No, Strassen terms are products of linear forms.
    # So each term k is a rank-1 tensor in the decomposition.
    # U_k, V_k, W_k are the vectors.
        
    # Let's define the 7 rank-1 terms for M_{2,2,2}.
    # Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # Term 2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    # Term 3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    # Term 4: U=[0,0,0,1], V=[0,0,1,-1], W=[0,0,1,0]
    # Term 5: U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,0]
    # Term 6: U=[0,0,1,-1], V=[1,0,1,0], W=[0,0,0,1]
    # Term 7: U=[0,1,0,-1], V=[0,1,0,1], W=[1,0,0,0]
        
    # Let's verify Term 1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1. Correct.
    # Term 2: (A10+A11)B00 -> C10 += 1. Correct.
    # Term 3: A00(B01-B11) -> C01 += 1. Correct.
    # Term 4: A11(B10-B00) -> C10 += 1. Correct.
    # Term 5: (A00+A01)B11 -> C01 += 1. Correct.
    # Term 6: (A10-A00)(B00+B10) -> C11 += 1. Correct.
    # Term 7: (A01-A11)(B01+B11) -> C00 += 1. Correct.
        
    # Now sum them up:
    # C00 = T1 + T7 = (A00+A11)(B00+B11) + (A01-A11)(B01+B11)
    # This is NOT ax+bz.
        
    # I will use the 8-term decomposition for M_{2,2,2} to be safe.
    # Rank 8 * 2 * 3 + 8 * 3 = 48 + 24 = 72.
        
    # I will return the schoolbook 72.
        
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
