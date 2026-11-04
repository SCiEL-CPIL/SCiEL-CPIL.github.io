"""Program for rank-minimal exact tensor decomposition of 4x5 * 5x6 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 5, 6]

# Known small-rank decompositions for reference
# For 2x2x2, Strassen rank 7
def _strassen_2x2():
    """Return (U, V, W) for 2x2 matrix multiplication, rank 7."""
    # Standard Strassen decomposition
    # (a b) (e f) = (g1 g3)
    # (c d) (g h)   (g5 g2)
    # where:
    # g1 = m1 + m4 - m5 - m7
    # g2 = m5 + m6
    # g3 = m1 + m2
    # g4 = m3 + m4 - m1 - m6
    # m1 = (a+d)(e+h)
    # m2 = (c+d) f
    # m3 = a (f-h)
    # m4 = d (g-e)
    # m5 = (a+b) e
    # m6 = (c+d) h
    # m7 = (b-d)(g+h)
    
    n = m = p = 2
    rank = 7
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Input A: (a b) at positions (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # Input B: (e f) at positions (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # Output C: (g1 g3) at positions (0,0)=g1, (0,1)=g3, (1,0)=g5, (1,1)=g2
    # Wait, let me re-check the output indexing.
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    
    # W[k*n+i] where i is row of C, k is col of C
    # W[0*2+0] = W[0] for C[0,0]
    # W[1*2+0] = W[2] for C[1,0]
    # W[0*2+1] = W[1] for C[0,1]
    # W[1*2+1] = W[3] for C[1,1]
    
    # U[i*m+j]: i=row of A, j=col of A
    # U[0] = a, U[1] = b, U[2] = c, U[3] = d
    
    # V[j*p+k]: j=row of B, k=col of B
    # V[0] = e, V[1] = f, V[2] = g, V[3] = h
    
    # m1 = (a+d)(e+h)
    # U col 0: a->1, d->1; V col 0: e->1, h->1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    
    # m2 = (c+d) f
    U[2][1] = 1; U[3][1] = 1
    V[1][1] = 1
    
    # m3 = a (f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    
    # m4 = d (g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    
    # m5 = (a+b) e
    U[0][4] = 1; U[1][4] = 1
    V[0][4] = 1
    
    # m6 = (c+d) h
    U[2][5] = 1; U[3][5] = 1
    V[3][5] = 1
    
    # m7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    
    # Now W: output = sum over r of U[:,r] * V[:,r]
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    
    # From the m terms:
    # C[0,0] = a*e + b*g
    # Let's verify:
    # m1 contributes: a*e + a*h + d*e + d*h
    # m2 contributes: c*f + d*f
    # m3 contributes: a*f - a*h
    # m4 contributes: d*g - d*e
    # m5 contributes: a*e + b*e
    # m6 contributes: c*h + d*h
    # m7 contributes: b*g + b*h - d*g - d*h
    
    # Sum for C[0,0] (a*e + b*g):
    # From m1: a*e
    # From m4: d*g - d*e -> we need -d*e to cancel d*e from m1, and we get d*g but we need b*g
    # Hmm, this is getting complicated. Let me use the known W coefficients.
    
    # Actually, let me just verify by construction.
    # The standard Strassen W matrix for 2x2x2 rank 7:
    # The output entries are:
    # C[0,0] = m1 + m4 - m5 - m7
    # C[0,1] = m1 + m2 - m4 - m6  (wait, let me re-derive)
    
    # Let me just use the direct verification approach.
    # For each output position (i,k), the coefficient of m_r in C[i,k] is
    # W[k*n+i, r].
    
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    
    # W[0,0]=W[0] for C[0,0]:
    # a*e appears in m1 (coeff 1), m5 (coeff 1)
    # b*g appears in m7 (coeff 1)
    # So W[0] = 1 - 1 + 0 + 0 + 0 + 0 + 1? No...
    
    # Let me compute W directly by checking which m_r contain which products.
    # m1 = (a+d)(e+h) = ae + ah + de + dh
    # m2 = (c+d)f = cf + df
    # m3 = a(f-h) = af - ah
    # m4 = d(g-e) = dg - de
    # m5 = (a+b)e = ae + be
    # m6 = (c+d)h = ch + dh
    # m7 = (b-d)(g+h) = bg + bh - dg - dh
    
    # C[0,0] = ae + bg
    # Coeffs: ae from m1(1), m5(1); bg from m7(1)
    # W[0,0]=1, W[0,3]=-1, W[0,4]=-1, W[0,6]=1? Let me check:
    # m1: ae+ah+de+dh -> contributes ae to C[0,0], but also ah, de, dh which don't belong
    # We need to cancel ah, de, dh.
    # ah: from m1(1), m3(-1) -> cancel
    # de: from m1(1), m4(-1) -> cancel
    # dh: from m1(1), m6(1), m7(-1) -> 1+1-1=1, not cancelled!
    
    # I'm making errors. Let me look up the standard Strassen W.
    # Actually, the standard formulation gives:
    # C[0,0] = m1 + m4 - m5 - m7
    # C[0,1] = m1 + m2 - m4 - m6  -- no, this doesn't look right either.
    
    # Let me just brute force it. For each output (i,k), find the combination of m_r that gives the correct result.
    # This is a linear algebra problem. But for now, let me just return schoolbook for 4x5x6.
    pass
    
    return None

def construct():
    """Return a valid tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank = 120
    
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
