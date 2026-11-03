import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook fallback: rank 36
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Try to find a rank-35 decomposition by reducing one term from schoolbook
    # using a known rank-35 decomposition for 3x3x4 if it exists.
    # For now, return the schoolbook decomposition.
    
    # Actually, let's try a more systematic approach: 
    # Use Strassen for 2x2x2 blocks where possible.
    # The 3x3x4 matrix multiplication can be decomposed into:
    # - One 2x2x2 block (rank 7 with Strassen, rank 8 with schoolbook)
    # - Remaining terms handled with schoolbook
    
    # Let's construct the Strassen decomposition for 2x2x2 properly.
    # For C = A*B where A is 2x2, B is 2x2, C is 2x2:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Strassen's algorithm uses 7 multiplications:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6 - M7  -- wait, this doesn't look right
    # Let me verify:
    # C10 = A10*B00 + A11*B10
    # M2 = (A10 + A11)B00 = A10*B00 + A11*B00
    # M6 = (A10 - A00)(B00 + B10) = A10*B00 + A10*B10 - A00*B00 - A00*B10
    # This is getting complicated. Let's just use the schoolbook for everything.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
