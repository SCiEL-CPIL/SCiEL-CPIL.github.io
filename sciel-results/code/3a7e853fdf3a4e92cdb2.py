import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try rank 27: combine Strassen parts more efficiently
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B14 (j=0,1, k=0..3) - try 13 terms instead of 14
    # B14 is 2x4 matrix (rows 0,1; cols 0,1,2,3)
    # We need: A1 (2x2) * B14 (2x4) = C14 (2x4)
    # This is equivalent to 4 separate 2x2 matrix multiplications sharing A1
    # Standard: 4 * 7 = 28, but we can share some terms
    
    # Use a shared Strassen-like approach for the 2x2 * 2x4 case
    # Let A1 = [[a,b],[c,d]], B14 = [[e,f,g,h],[i,j,k,l]]
    # C14[0,0] = ae + bi
    # C14[1,0] = ce + di
    # C14[0,1] = af + bj
    # C14[1,1] = cf + dj
    # C14[0,2] = ag + bk
    # C14[1,2] = cg + dk
    # C14[0,3] = ah + bl
    # C14[1,3] = ch + dl
    
    # Define shared Strassen products:
    # M1 = (a+b)(e+f) = ae+af+be+bf  -> helps C14[0,0],C14[0,1],C14[1,0],C14[1,1]
    # M2 = (c+d)(e+f) = ce+cf+de+df  -> helps C14[1,0],C14[1,1]
    # M3 = a(g-h) = ag-ah             -> helps C14[0,2],C14[0,3]
    # M4 = d(i-j) = di-dj             -> helps C14[1,0],C14[1,1]
    # M5 = (a+b)k = ak+bk             -> helps C14[0,2]
    # M6 = (c+d)l = cl+dl             -> helps C14[1,3]
    # M7 = (a-c)(e+i) = ae+ai-ce-ci  -> helps C14[0,0],C14[1,0]
    # M8 = (b-d)(f+j) = bf+bj-df-dj  -> helps C14[0,1],C14[1,1]
    # M9 = b(h-l) = bh-bl             -> helps C14[0,3]
    # M10 = c(g-k) = cg-ck            -> helps C14[1,2]
    # M11 = a(i+e) = ai+ae            -> helps C14[0,0]
    # M12 = d(e+i) = de+di            -> helps C14[1,0]
    # M13 = b(j+f) = bj+bf            -> helps C14[0,1]
    
    # Let me try a different approach: use the known rank for 2x2 * 2x4
    # Actually, let's just try rank 27 by merging one term
    
    # Reset and use a cleaner approach
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
    # M1: (a+b)(e+f)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2: (c+d)e
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3: a(f-j)
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4: d(i-e)
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5: (a+b)k
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6: (c+d)(e+f)
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7: (b-d)(f+j)
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
    # M1: (a+b)(g+h)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2: (c+d)g
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3: a(h-k)
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4: d(k-g)
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5: (a+b)k
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6: (c+d)(g+h)
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7: (b-d)(h+k)
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
