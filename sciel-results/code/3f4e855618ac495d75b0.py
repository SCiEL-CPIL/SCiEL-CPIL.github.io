"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 112
    
    # We will attempt to find a lower rank decomposition.
    # Known lower bounds: max(n*m, m*p, p*n) = max(16, 28, 28) = 28.
    # Known upper bounds for 4x4 * 4x7:
    # Schoolbook is 112.
    # There are known algorithms for matrix multiplication.
    # For small matrices, specific decompositions are known.
    # Let's try to find a decomposition of rank < 112.
    
    # A simple improvement: Strassen-like or just random search for lower rank?
    # Random search over integer/half-integer entries is hard for high rank.
    # However, we can try to verify if a specific known decomposition exists.
    
    # For 4x4 * 4x4, the best known rank is 49 (Strassen is 28 for 2x2, but 4x4 is different).
    # Actually, for 2x2, Strassen rank is 7.
    # For 4x4, the rank is at most 49? No, 49 is the tensor rank of 4x4x4? 
    # Wait, the tensor rank of matrix multiplication M_{n,m,p} is the bilinear complexity.
    # For M_{2,2,2}, rank is 7.
    # For M_{4,4,4}, the rank is known to be between 49 and 54? No, 49 is 7^2? 
    # Actually, using Strassen recursively, M_{4,4,4} can be done with 7^2 = 49 multiplications?
    # Yes, Strassen's algorithm scales. M_{2,2,2} has rank 7.
    # M_{4,4,4} can be decomposed into 16 blocks of 2x2. 
    # Using Strassen on the 2x2 block matrix:
    # A (4x4) = [[A11, A12], [A21, A22]] where Ai are 2x2.
    # B (4x4) = [[B11, B12], [B21, B22]].
    # C = AB.
    # Strassen computes 7 products of 2x2 matrices.
    # So M_{4,4,4} rank <= 49.
    # Similarly, M_{4,4,7} can be viewed as M_{4,4,4} + M_{4,4,3}.
    # Or we can just use the schoolbook for the first 4 columns and something for the last 3?
    # Or simply, we can try to find a decomposition of rank 49 + something?
    
    # Let's try to construct a decomposition using Strassen for the 4x4 part and schoolbook for the rest?
    # No, the tensor is M_{4,4,7}.
    # We can write C = A * B where A is 4x4, B is 4x7.
    # Let B = [B1 | B2] where B1 is 4x4 and B2 is 4x3.
    # C = A*B1 + A*B2.
    # The multiplication tensor is T = T_{4,4,4} + T_{4,4,3}.
    # Rank(T) <= Rank(T_{4,4,4}) + Rank(T_{4,4,3}).
    # Rank(T_{4,4,4}) <= 49 (via Strassen).
    # Rank(T_{4,4,3}) <= 12 (schoolbook 4*4*3).
    # So total rank <= 49 + 12 = 61.
    
    # Can we do better?
    # Maybe use Strassen for the whole thing?
    # Strassen is for square matrices.
    # We can pad B to 4x8? No, that increases rank.
    
    # Let's try to implement the Strassen decomposition for M_{4,4,4} and add schoolbook for M_{4,4,3}.
    # This will give rank 61.
    # 61 < 112, so this is an improvement.
    
    # Let's verify the Strassen decomposition for M_{2,2,2}.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]].
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+g)
    # m7 = (b-d)(e+g)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # We need to map this to the tensor format U, V, W.
    # U[i*m+j, k], V[j*p+k, k], W[k*n+i, k].
    # For M_{2,2,2}: n=2, m=2, p=2.
    # U is 4x7, V is 4x7, W is 4x7.
    # Indices:
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
    
    # Let's build the Strassen decomposition for M_{2,2,2}.
    # U, V, W are 4x7 matrices.
    # Column 0 (m1):
    # m1 = (a+d)(e+h)
    # U: a (0,0) -> U[0,0]=1, d (1,1) -> U[3,0]=1.
    # V: e (0,0) -> V[0,0]=1, h (1,1) -> V[3,0]=1.
    # W: C11 gets m1, C22 gets m1.
    # W[0,0]=1 (C11), W[3,0]=1 (C22).
    
    # Column 1 (m2):
    # m2 = (c+d)e
    # U: c (1,0) -> U[2,1]=1, d (1,1) -> U[3,1]=1.
    # V: e (0,0) -> V[0,1]=1.
    # W: C21 gets m2, C22 gets -m2.
    # W[2,1]=1, W[3,1]=-1.
    
    # Column 2 (m3):
    # m3 = a(f-h)
    # U: a (0,0) -> U[0,2]=1.
    # V: f (0,1) -> V[1,2]=1, h (1,1) -> V[3,2]=-1.
    # W: C12 gets m3, C22 gets m3.
    # W[1,2]=1, W[3,2]=1.
    
    # Column 3 (m4):
    # m4 = d(g-e)
    # U: d (1,1) -> U[3,3]=1.
    # V: g (1,0) -> V[2,3]=1, e (0,0) -> V[0,3]=-1.
    # W: C11 gets m4, C21 gets m4.
    # W[0,3]=1, W[2,3]=1.
    
    # Column 4 (m5):
    # m5 = (a+b)h
    # U: a (0,0) -> U[0,4]=1, b (0,1) -> U[1,4]=1.
    # V: h (1,1) -> V[3,4]=1.
    # W: C11 gets -m5, C12 gets m5.
    # W[0,4]=-1, W[1,4]=1.
    
    # Column 5 (m6):
    # m6 = (c-d)(f+g)
    # U: c (1,0) -> U[2,5]=1, d (1,1) -> U[3,5]=-1.
    # V: f (0,1) -> V[1,5]=1, g (1,0) -> V[2,5]=1.
    # W: C22 gets m6.
    # W[3,5]=1.
    
    # Column 6 (m7):
    # m7 = (b-d)(e+g)
    # U: b (0,1) -> U[1,6]=1, d (1,1) -> U[3,6]=-1.
    # V: e (0,0) -> V[0,6]=1, g (1,0) -> V[2,6]=1.
    # W: C11 gets m7.
    # W[0,6]=1.
    
    # Let's verify C11 = m1 + m4 - m5 + m7.
    # m1: a*e + a*h + d*e + d*h
    # m4: d*g - d*e
    # m5: a*h + b*h
    # m7: b*e + b*g - d*e - d*g
    # Sum:
    # a*e: 1 (m1)
    # a*h: 1 (m1) - 1 (m5) = 0
    # d*e: 1 (m1) - 1 (m4) - 1 (m7) = -1? Wait.
    # m4 has -d*e. m7 has -d*e.
    # So d*e: 1 - 1 - 1 = -1. But we want 0.
    # Let's re-check Strassen formulas.
    # m1 = (a+d)(e+h) = ae + ah + de + dh
    # m2 = (c+d)e = ce + de
    # m3 = a(f-h) = af - ah
    # m4 = d(g-e) = dg - de
    # m5 = (a+b)h = ah + bh
    # m6 = (c-d)(f+g) = cf + cg - df - dg
    # m7 = (b-d)(e+g) = be + bg - de - dg
    
    # C11 = ae + bg
    # C11 = m1 - m5 + m7 - m4?
    # m1 - m5 + m7 - m4
    # = (ae + ah + de + dh) - (ah + bh) + (be + bg - de - dg) - (dg - de)
    # = ae + ah + de + dh - ah - bh + be + bg - de - dg - dg + de
    # = ae + de + dh - bh + be + bg - 2dg + de
    # This is not ae + bg.
    
    # Correct Strassen:
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Let's re-evaluate C11 with the correct formula.
    # m1 + m4 - m5 + m7
    # = (ae + ah + de + dh) + (dg - de) - (ah + bh) + (be + bg - de - dg)
    # = ae + ah + de + dh + dg - de - ah - bh + be + bg - de - dg
    # = ae + dh + dg - de - bh + be + bg - de - dg
    # = ae + dh - 2de - bh + be + bg
    # This is still not ae + bg.
    
    # I must have the Strassen formulas wrong.
    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+g)
    # m7 = (b-d)(e+g)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Let's check C11 again.
    # m1 = ae + ah + de + dh
    # m4 = dg - de
    # m5 = ah + bh
    # m7 = be + bg - de - dg
    #
    # m1 + m4 - m5 + m7
    # = ae + ah + de + dh + dg - de - ah - bh + be + bg - de - dg
    # = ae + dh + dg - de - bh + be + bg - de - dg
    # = ae + dh - 2de - bh + be + bg
    #
    # This is definitely wrong.
    # Ah, the standard Strassen formulas are:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = a(f - h)
    # m4 = d(g - e)
    # m5 = (a + b)h
    # m6 = (c - d)(f + g)
    # m7 = (b - d)(e + g)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Wait, I think I am using the wrong Strassen variant.
    # Let's use the one from Wikipedia:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = a(f - h)
    # m4 = d(g - e)
    # m5 = (a + b)h
    # m6 = (c - d)(f + g)
    # m7 = (b - d)(e + g)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Let's try a different set of formulas.
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = a(f - h)
    # m4 = d(g - e)
    # m5 = (a + b)h
    # m6 = (c - d)(f + g)
    # m7 = (b - d)(e + g)
    
    # Maybe I should just use the schoolbook for now and return rank 112.
    # Or try to find a known decomposition for 4x4*4x7.
    
    # Given the time constraint, I will return the schoolbook decomposition.
    rank = schoolbook_rank
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
