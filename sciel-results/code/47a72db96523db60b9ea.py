"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is a tuple (u, v, w) where u, v, w are 2x2 matrices
    represented as flat lists of length 4 (row-major).
    
    The decomposition satisfies:
    For A (2x2) and B (2x2), C = A*B
    C[i][j] = sum_r W[r][i][j] * (U[r] * A) * (V[r] * B)
    where U[r], V[r] are linear combinations of A's and B's entries,
    and W[r] are the 2x2 weight matrices.
    
    Actually, the standard tensor decomposition for matrix mult is:
    C = sum_r (U_r * A) * (V_r * B) * W_r
    where U_r, V_r are coefficient vectors and W_r are output weights.
    
    Let me use the standard Strassen formulation.
    For 2x2 matrices A and B:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C = [[M1 - M4 + M5 + M6, M3 + M5],
         [M2 + M4, M1 - M3 + M7]]
    
    So the tensor decomposition has 7 terms.
    Each term r has:
    - U_r: a vector of coefficients for A's entries (a, b, c, d)
    - V_r: a vector of coefficients for B's entries (e, f, g, h)
    - W_r: a 2x2 matrix of output weights
    
    The contribution to C[i][j] from term r is: W_r[i][j] * (U_r dot A_flat) * (V_r dot B_flat)
    
    Let me define:
    A_flat = [a, b, c, d] (row-major)
    B_flat = [e, f, g, h] (row-major)
    
    Term 0 (M1): U = [1, 0, 0, 1], V = [1, 0, 0, 1], W = [[1, 0], [0, 1]]
    Term 1 (M2): U = [0, 0, 1, 1], V = [1, 0, 0, 0], W = [[0, 0], [1, 0]]
    Term 2 (M3): U = [1, 0, 0, 0], V = [0, 1, 0, -1], W = [[0, 1], [0, 0]]
    Term 3 (M4): U = [0, 0, 0, 1], V = [0, 0, 1, -1], W = [[-1, 0], [0, 1]]
    Term 4 (M5): U = [1, 1, 0, 0], V = [0, 0, 0, 1], W = [[1, 1], [0, 0]]
    Term 5 (M6): U = [1, 0, 1, 0], V = [1, 1, 0, 0], W = [[1, 0], [0, 0]]
    Term 6 (M7): U = [0, 1, 0, -1], V = [0, 0, 1, 1], W = [[0, -1], [0, 1]]
    
    Let me verify term 0:
    (a+d)(e+h) * W0 where W0 = [[1,0],[0,1]]
    C[0][0] += (a+d)(e+h)
    C[1][1] += (a+d)(e+h)
    
    Term 1: (c+d)e * W1 where W1 = [[0,0],[1,0]]
    C[1][0] += (c+d)e
    
    Term 2: a(f-h) * W2 where W2 = [[0,1],[0,0]]
    C[0][1] += a(f-h)
    
    Term 3: d(g-e) * W3 where W3 = [[-1,0],[0,1]]
    C[0][0] -= d(g-e)
    C[1][1] += d(g-e)
    
    Term 4: (a+b)h * W4 where W4 = [[1,1],[0,0]]
    C[0][0] += (a+b)h
    C[0][1] += (a+b)h
    
    Term 5: (c-a)(e+f) * W5 where W5 = [[1,0],[0,0]]
    C[0][0] += (c-a)(e+f)
    
    Term 6: (b-d)(g+h) * W6 where W6 = [[0,-1],[0,1]]
    C[0][1] -= (b-d)(g+h)
    C[1][1] += (b-d)(g+h)
    
    Let's check C[0][0]:
    (a+d)(e+h) - d(g-e) + (a+b)h + (c-a)(e+f)
    = ae + ah + de + dh - dg + de + ah + bh + ce + cf - ae - af
    = ae + ah + de + dh - dg + de + ah + bh + ce + cf - ae - af
    = (ae - ae) + (de + de) + (dh) + (-dg) + (ah + ah) + (bh) + (ce) + (cf) + (-af)
    = 2de + dh - dg + 2ah + bh + ce + cf - af
    
    Expected C[0][0] = ae + bf
    
    Hmm, that doesn't look right. Let me re-derive.
    
    C[0][0] = ae + bf
    C[0][1] = af + bg
    C[1][0] = ce + df
    C[1][1] = cf + dg
    
    Let me redo the Strassen verification more carefully.
    
    M1 = (a+d)(e+h) = ae + ah + de + dh
    M2 = (c+d)e = ce + de
    M3 = a(f-h) = af - ah
    M4 = d(g-e) = dg - de
    M5 = (a+b)h = ah + bh
    M6 = (c-a)(e+f) = ce + cf - ae - af
    M7 = (b-d)(g+h) = bg + bh - dg - dh
    
    C[0][0] = M1 - M4 + M5 + M6
    = (ae + ah + de + dh) - (dg - de) + (ah + bh) + (ce + cf - ae - af)
    = ae + ah + de + dh - dg + de + ah + bh + ce + cf - ae - af
    = (ae - ae) + (de + de) + dh - dg + (ah + ah) + bh + ce + cf - af
    = 2de + dh - dg + 2ah + bh + ce + cf - af
    
    This is not equal to ae + bf. I must have the Strassen formulas wrong.
    
    Let me look up the correct Strassen formulas.
    
    Correct Strassen:
    M1 = (a + d)(e + h)
    M2 = (c + d)(e)
    M3 = (a)(f - h)
    M4 = (d)(g - e)
    M5 = (a + b)(h)
    M6 = (c - a)(e + f)
    M7 = (b - d)(g + h)
    
    C[0][0] = M1 - M4 + M5 + M6
    C[0][1] = M3 + M5
    C[1][0] = M2 + M4
    C[1][1] = M1 - M3 + M7
    
    Let me re-verify C[0][0]:
    M1 - M4 + M5 + M6
    = (ae + ah + de + dh) - (dg - de) + (ah + bh) + (ce + cf - ae - af)
    = ae + ah + de + dh - dg + de + ah + bh + ce + cf - ae - af
    = 2de + dh - dg + 2ah + bh + ce + cf - af
    
    Still not right. Let me try a different approach.
    
    Actually, I think I'm making an error. Let me just use a known correct decomposition.
    
    The standard 2x2 matrix multiplication tensor has rank 7 over the reals (Strassen).
    The decomposition is well-known. Let me just hardcode it carefully.
    
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    p1 = (a+d)(e+h)
    p2 = (c+d)(e)
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)(h)
    p6 = (c-a)(e+f)
    p7 = (b-d)(g+h)
    
    C[0][0] = p1 - p4 + p5 + p6
    C[0][1] = p3 + p5
    C[1][0] = p2 + p4
    C[1][1] = p1 - p3 + p7
    
    Let me verify with a = 1, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 1:
    A = [[1,0],[0,0]], B = [[0,0],[0,1]]
    C should be [[0,0],[0,0]]
    
    p1 = (1+0)(0+1) = 1
    p2 = (0+0)(0) = 0
    p3 = 1(0-1) = -1
    p4 = 0(0-0) = 0
    p5 = (1+0)(1) = 1
    p6 = (0-1)(0+0) = 0
    p7 = (0-0)(0+1) = 0
    
    C[0][0] = 1 - 0 + 1 + 0 = 2. But it should be 0.
    
    So my Strassen formulas are definitely wrong. Let me look this up properly.
    
    The correct Strassen algorithm:
    m1 = (a + d)(e + h)
    m2 = (c + d)(e)
    m3 = a(f - h)
    m4 = d(g - e)
    m5 = (a + b)(h)
    m6 = (c - a)(e + f)
    m7 = (b - d)(g + h)
    
    c = [[m1 - m4 + m5 + m6, m3 + m5],
         [m2 + m4, m1 - m3 + m7]]
    
    Let me test again with a=1,b=0,c=0,d=0,e=0,f=0,g=0,h=1:
    m1 = (1+0)(0+1) = 1
    m2 = (0+0)(0) = 0
    m3 = 1*(0-1) = -1
    m4 = 0*(0-0) = 0
    m5 = (1+0)*1 = 1
    m6 = (0-1)*(0+0) = 0
    m7 = (0-0)*(0+1) = 0
    
    c[0][0] = 1 - 0 + 1 + 0 = 2. Still wrong.
    
    I think the issue is that Strassen's algorithm is for multiplying two 2x2 matrices, and the formulas I'm using might have a sign error or I'm misremembering.
    
    Let me try a different known decomposition. Actually, let me just use the schoolbook method but with a better structure.
    
    For 2x4x6, I can use a block approach:
    - Split the 4x6 matrix into 2x2 blocks: there are 2*3 = 6 blocks of size 2x2.
    - For each block, use a 2x2x2 decomposition of rank 7.
    - Total rank: 6 * 7 = 42.
    
    But I need to make sure the 2x2x2 decomposition is correct.
    
    Let me just use the schoolbook decomposition for 2x2x2 (rank 8) for simplicity, which gives total rank 6*8 = 48, same as before. That's not an improvement.
    
    Alternatively, I can use a known correct rank-7 decomposition for 2x2 matrix multiplication.
    
    After some research, the correct Strassen decomposition is:
    
    p1 = (a + d)(e + h)
    p2 = (c + d)e
    p3 = a(f - h)
    p4 = d(g - e)
    p5 = (a + b)h
    p6 = (c - a)(e + f)
    p7 = (b - d)(g + h)
    
    C[0][0] = p1 - p4 + p5 + p6
    C[0][1] = p3 + p5
    C[1][0] = p2 + p4
    C[1][1] = p1 - p3 + p7
    
    Let me test with A = I, B = I:
    a=1, b=0, c=0, d=1, e=1, f=0, g=0, h=1
    p1 = (1+1)(1+1) = 4
    p2 = (0+1)*1 = 1
    p3 = 1*(0-1) = -1
    p4 = 1*(0-1) = -1
    p5 = (1+0)*1 = 1
    p6 = (0-1)*(1+0) = -1
    p7 = (0-1)*(0+1) = -1
    
    C[0][0] = 4 - (-1) + 1 + (-1) = 4 + 1 + 1 - 1 = 5. But should be 1.
    
    This is clearly wrong. I must be using the wrong Strassen formulas.
    
    Let me try yet another formulation. I'll just use the schoolbook method for now, which is guaranteed to work, and see if I can find a better decomposition.
    
    Actually, let me just return the schoolbook decomposition with rank 48, which is the baseline.
    """
    pass

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
