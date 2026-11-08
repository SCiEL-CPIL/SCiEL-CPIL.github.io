import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    # Rank 25 solution:
    # Split p=5 into p1=2, p2=3.
    # T_{2,3,2} rank 11 (4 + 7)
    # T_{2,3,3} rank 14 (6 + 8)
    # Total rank 25.
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # --- Part 1: T_{2,3,2} (Columns 0,1) ---
    # Split m=3 into m1=1 (row 2), m2=2 (rows 0,1)
    
    # Subpart 1a: T_{2,1,2} (A row 2, B cols 0,1 rows 0,1? No, B rows corresponding to A col 2)
    # A[0:2, 2] * B[2:3, 0:2]
    # A is 2x3, B is 3x5.
    # A[:, 2] is 2x1. B[2, 0:2] is 1x2.
    # Product is 2x2.
    # Rank 4.
    for i in range(n):
        for k in range(2):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # r = 4
    
    # Subpart 1b: T_{2,2,2} (A rows 0,1 cols 0,1; B rows 0,1 cols 0,1)
    # A[0:2, 0:2] * B[0:2, 0:2]
    # Rank 7 (Strassen).
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # T1: (a+b)(e+g) -> C00 += 1, C11 += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[0 * p + 0][r] = 1; V[1 * p + 0][r] = 1
    W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    
    # T2: (c+d)(e+g) -> C10 += 1, C11 -= 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1; V[1 * p + 0][r] = 1
    W[1 * n + 0][r] = 1; W[1 * n + 1][r] = -1
    r += 1
    
    # T3: (c+d)(f-h) -> C10 += 1, C11 += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    
    # T4: (a+b)(h-f) -> C00 += 1, C01 += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[0 * p + 1][r] = -1; V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1
    r += 1
    
    # T5: a*e -> C00 += 1
    U[0 * m + 0][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # T6: d*h -> C11 += 1
    U[1 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # T7: (a-b)(g-h) -> C01 += 1, C10 -= 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = -1
    V[1 * p + 0][r] = 1; V[1 * p + 1][r] = -1
    W[0 * n + 1][r] = 1; W[1 * n + 0][r] = -1
    r += 1
    # r = 4 + 7 = 11
    
    # --- Part 2: T_{2,3,3} (Columns 2,3,4) ---
    # Split m=3 into m1=1 (row 2), m2=2 (rows 0,1)
    
    # Subpart 2a: T_{2,1,3} (A row 2, B cols 2,3,4 rows 0,1? No, B row 2)
    # A[:, 2] * B[2, 2:5]
    # Rank 6.
    for i in range(n):
        for k in range(2, 5):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # r = 11 + 6 = 17
    
    # Subpart 2b: T_{2,2,3} (A rows 0,1 cols 0,1; B rows 0,1 cols 2,3,4)
    # Rank 8.
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]] (cols 2,3,4)
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    # T8: (a+b)(e+h) -> C00 += 1, C11 += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[0 * p + 2][r] = 1; V[1 * p + 2][r] = 1
    W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1
    r += 1
    
    # T9: (c+d)(e+h) -> C10 += 1, C11 -= 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1; V[1 * p + 2][r] = 1
    W[3 * n + 0][r] = 1; W[3 * n + 1][r] = -1
    r += 1
    
    # T10: (c+d)(f-i) -> C10 += 1, C11 += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1
    r += 1
    
    # T11: (a+b)(i-f) -> C00 += 1, C01 += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[0 * p + 3][r] = -1; V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1
    r += 1
    
    # T12: a*e -> C00 += 1
    U[0 * m + 0][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # T13: d*h -> C11 += 1
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # T14: (a-b)(h-i) -> C01 += 1, C10 -= 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = -1
    V[1 * p + 2][r] = 1; V[1 * p + 3][r] = -1
    W[2 * n + 1][r] = 1; W[3 * n + 0][r] = -1
    r += 1
    
    # T15: (g-j) term? 
    # We need to handle column 4 (g, j).
    # The above 8 terms handle cols 2,3.
    # We need 4 more for col 4? No, rank 8 total for 2x2x3.
    # The 8 terms above only cover 2 columns of B.
    # We need to cover 3 columns.
    # The standard rank 8 decomposition for 2x2x3 covers all 3 columns.
    # My T8-T14 only used V indices 2,3.
    # I need to include V index 4.
    
    # Let's restart Subpart 2b with a correct rank 8 decomposition for 2x2x3.
    # The rank 8 decomposition is not simply Strassen + 1.
    # It is known to be 8.
    # I will use a known decomposition.
    
    # Correction: I will just use naive rank 12 for this part to be safe, 
    # but that would make total rank 11 + 6 + 12 = 29.
    # Better to use rank 8.
    
    # I will implement the rank 8 decomposition for 2x2x3.
    # Reference: "Rank of 2x2x3 tensor"
    
    # This is complex to write from scratch.
    # I will fall back to Rank 26 solution (10 + 16) or Rank 28.
    # Actually, I will just use the Rank 28 solution but optimized?
    # No, I will use the Rank 25 solution but replace Subpart 2b with Rank 12 (Naive) -> Total 29.
    # This is worse than 28.
    
    # Let's try Rank 26:
    # T_{2,3,2} rank 11.
    # T_{2,3,3} rank 15? (6 + 9?)
    # If I use Naive for 2x2x3, it is 12.
    # 6 + 12 = 18.
    # 11 + 18 = 29.
    
    # I will stick to the current Rank 28 solution as it is valid.
    # But I can improve the Strassen parts?
    # The current code uses Rank 7 for 2x2x2.
    # It uses Rank 4 for 2x2x1.
    # It uses Rank 10 for 2x1x5.
    # Total 28.
    
    # I will return the current solution.
    
    # Re-initialize to Rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - rank 4
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - rank 7
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1
    r += 1
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1
    r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1
    r += 1
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - rank 7
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1
    r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1
    r += 1
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1
    r += 1
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1
    r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1
    r += 1
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
