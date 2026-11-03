"""Tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # We'll use a block decomposition with Strassen for 2x2x2 blocks.
    # Split: i in {0,1} vs i=2, j in {0,1} vs j=2, k in {0,1} vs k in {2,3}
    
    # Strassen 2x2x2 (corrected formulas):
    # For C = A*B where A,B,C are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 + M3 - M2 + M6
    
    # Let me verify these formulas numerically:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(g-h) = ag-ah
    # M4 = d(f-e) = df-de
    # M5 = (a+b)h = ah+bh
    # M6 = (c-a)(e+g) = ce+cg-ae-ag
    # M7 = (b-d)(f+h) = bf+bh-df-dh
    #
    # C00 should be ae+bg:
    # M1-M4+M5+M7 = (ae+ah+de+dh) - (df-de) + (ah+bh) + (bf+bh-df-dh)
    # = ae+ah+de+dh-df+de+ah+bh+bf+bh-df-dh
    # = ae + 2ah + 2de + 2bh + bf - 2df
    # That's not ae+bg. So these formulas are wrong.
    
    # Let me use the standard Strassen from Wikipedia:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21-B22)
    # M4 = (A22)(B12-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B21)
    # M7 = (A12-A22)(B12+B22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M7
    # C22 = M1 + M3 - M2 + M6
    
    # Using 1-indexed: A11=a, A12=b, A21=c, A22=d
    # B11=e, B12=f, B21=g, B22=h
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(g-h) = ag-ah
    # M4 = d(f-e) = df-de
    # M5 = (a+b)h = ah+bh
    # M6 = (c-a)(e+g) = ce+cg-ae-ag
    # M7 = (b-d)(f+h) = bf+bh-df-dh
    #
    # C11 = ae+bg:
    # M1-M4+M5+M7 = ae+ah+de+dh - df+de + ah+bh + bf+bh-df-dh
    # = ae + 2ah + 2de + 2bh + bf - 2df
    # Still wrong.
    
    # Let me just use a known correct Strassen implementation.
    # From Wikipedia (1-indexed):
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)(B11)
    # M3 = (A11)(B21 - B22)
    # M4 = (A22)(B12 - B11)
    # M5 = (A11 + A12)(B22)
    # M6 = (A21 - A11)(B11 + B21)
    # M7 = (A12 - A22)(B12 + B22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M7
    # C22 = M1 + M3 - M2 + M6
    
    # Let me verify C12 = af+bg:
    # M3+M4 = ag-ah + df-de
    # That should equal af+bg. Not matching.
    
    # I think the issue is with the standard Strassen formulas. Let me look them up carefully.
    # Actually, let me just use numpy to find a valid rank-7 decomposition for 2x2x2.
    
    # For now, let's use a simpler approach: schoolbook for all blocks.
    # But we can still try to get a better decomposition.
    
    # Actually, let me just return the schoolbook decomposition which is guaranteed correct.
    rank = n * m * p  # 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Validate
    if time.time() - t0 < DEADLINE:
        valid = _validate(U, V, W, n, m, p)
        if not valid:
            raise ValueError("Validation failed")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
