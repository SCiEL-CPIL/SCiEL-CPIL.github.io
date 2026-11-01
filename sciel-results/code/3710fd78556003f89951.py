import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition: Strassen for 2x2x2 part + 4 dot products for 2x2x1 part
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x2): 0=a, 1=b, 2=c, 3=d
    # Indices for B (2x3): 0=e, 1=f, 2=g, 3=h, 4=i, 5=j
    # Indices for C (2x3): 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
    
    # Strassen products for C00, C01, C10, C11
    # p1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (c+d)(e+h)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1; V[3][1] = 1
    W[3][1] = 1; W[0][1] = -1 # C10 += p2, C00 -= p2? No.
    # Let's trace carefully.
    
    # Standard Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e+h)
    # p3 = a(f-h)
    # p4 = d(i-f)
    # p5 = (a+b)i
    # p6 = (c+d)f
    # p7 = (a-b)h
    #
    # C00 = p1 - p4 + p7
    # C01 = p5 + p4
    # C10 = p2 - p7
    # C11 = p1 - p3 + p2
    
    # Reset W for Strassen part
    for r in range(7):
        for i in range(6):
            W[i][r] = 0
            
    # p1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    W[4][0] = 1  # C11
    
    # p2 = (c+d)(e+h)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1; V[3][1] = 1
    W[3][1] = 1  # C10
    W[4][1] = 1  # C11
    
    # p3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[4][2] = -1 # C11 -= p3
    
    # p4 = d(i-f)
    U[3][3] = 1
    V[4][3] = 1; V[1][3] = -1
    W[0][3] = -1 # C00 -= p4
    W[1][3] = 1  # C01 += p4
    
    # p5 = (a+b)i
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1  # C01 += p5
    
    # p6 = (c+d)f
    U[2][5] = 1; U[3][5] = 1
    V[1][5] = 1
    # Not used in standard C00-C11? 
    # Wait, standard Strassen p6 is (c+d)f? 
    # Let's re-verify Strassen.
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e+h)
    # p3 = a(f-h)
    # p4 = d(i-g)  <-- Note: B = [[e,f],[g,h]] in some notations, [[e,f],[h,i]] in others.
    # My B indices: 0=e, 1=f, 2=g, 3=h, 4=i, 5=j.
    # So B = [[e, f, g], [h, i, j]].
    # Submatrix B12 = [[e, f], [h, i]].
    # So "g" in Strassen formula corresponds to h (index 3).
    # "h" in Strassen formula corresponds to i (index 4).
    # "f" is f (index 1).
    # "e" is e (index 0).
    
    # So:
    # p1 = (a+d)(e+h) -> U[0,3], V[0,3]
    # p2 = (c+d)(e+h) -> U[2,3], V[0,3]
    # p3 = a(f-h)     -> U[0], V[1,-3]
    # p4 = d(i-f)     -> U[3], V[4,-1]
    # p5 = (a+b)i     -> U[0,1], V[4]
    # p6 = (c+d)f     -> U[2,3], V[1]
    # p7 = (a-b)h     -> U[0,-1], V[3]
    
    # C00 = p1 - p4 + p7
    # C01 = p5 + p4
    # C10 = p2 - p7
    # C11 = p1 - p3 + p2
    
    # Let's rebuild W based on this.
    # p1: C00(1), C11(1)
    # p2: C10(1), C11(1)
    # p3: C11(-1)
    # p4: C00(-1), C01(1)
    # p5: C01(1)
    # p6: Unused? 
    # p7: C00(1), C10(-1)
    
    # Check C11: p1 - p3 + p2 = (a+d)(e+h) - a(f-h) + (c+d)(e+h)
    # = ae+ah+de+dh - af+ah + ce+ch+de+dh
    # = 2ae + 2ah + 2de + 2dh - af + ce + ch
    # Target C11 = cf + di.
    # This is not matching. My Strassen mapping is wrong or incomplete.
    
    # Correct Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e+h)
    # p3 = a(f-h)
    # p4 = d(i-g)  -> d(i-h) in my indices? No.
    # Let's use the explicit 7 products that work.
    
    # I will just use the Rank 12 schoolbook to be safe, or Rank 11 with verified products.
    # Actually, Rank 11 with Strassen is complex to verify manually.
    # I will return Rank 12 for correctness guarantee.
    # Wait, Rank 11 is definitely better if I can get it right.
    
    # Let's just use Schoolbook Rank 12.
    rank = 12
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
