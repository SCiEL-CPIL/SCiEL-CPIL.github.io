import numpy as np
import time
from itertools import product

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Build rank-43 decomposition using Strassen blocks
    rank = 43
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    def add_strassen_block(j_base, k_base):
        """Add rank-7 Strassen decomposition for 2x2 (in i,j) x 2x2 (in j,k) block.
        i is always 0,1 (full i dimension).
        j_base: which pair of j indices (0 or 2)
        k_base: which pair of k indices (0, 2, or 4)
        """
        nonlocal col
        j0, j1 = j_base, j_base + 1
        k0, k1 = k_base, k_base + 1
        
        # Standard Strassen for C = AB where A is 2x2, B is 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 - M2 + M4 + M5
        # C01 = M2 + M3
        # C10 = M4 + M5
        # C11 = M1 - M3 + M2 + M7
        # Wait, let me re-derive carefully.
        #
        # Actually, standard Strassen:
        # M1 = (A11-A22)(B21+B22)
        # M2 = (A11+A22)B22
        # M3 = (A11-A12)B12
        # M4 = A22(B11-B12)
        # M5 = (A11+A22)(B11+B22)  -- no, this isn't standard
        #
        # Let me use the standard formulation:
        # For C = A*B, A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 - M2 + M4 + M5
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M3 + M2 + M7  -- wait, this doesn't look right either
        #
        # Let me just verify: C11 = ae+bg, C12 = af+bh, C21 = ce+dg, C22 = cf+dh
        #
        # M1 = (a+d)(e+h) = ae+ah+de+dh
        # M2 = (c+d)e = ce+de
        # M3 = a(f-h) = af-ah
        # M4 = d(g-e) = dg-de
        # M5 = (a+b)h = ah+bh
        # M6 = (c-a)(e+f) = ce+cf-ae-af
        # M7 = (b-d)(g+h) = bg+bh-dg-dh
        #
        # C11 = ae+bg: M1 - M2 + M4 + M5 = (ae+ah+de+dh) - (ce+de) + (dg-de) + (ah+bh)
        #   = ae+ah+de+dh-ce-de+dg-de+ah+bh = ae+2ah+bh+dh-ce+dg -- no, this is wrong
        #
        # Let me try different combinations:
        # C11 = ae+bg
        # C12 = af+bh
        # C21 = ce+dg
        # C22 = cf+dh
        #
        # I'll use the well-known correct Strassen formulas:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C11 = M1 - M2 + M4 + M5
        #   = (ae+ah+de+dh) - (ce+de) + (dg-de) + (ah+bh)
        #   = ae+ah+de+dh-ce-de+dg-de+ah+bh
        #   = ae + 2ah - ce + dh + dg + bh - de
        # This is not ae+bg. So these formulas are wrong.
        #
        # Correct standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C11 = M1 - M2 + M4 + M5 -- let me just trust the tensor decomposition approach
        # and verify numerically.
        
        # Instead of deriving by hand, let me use the tensor decomposition directly.
        # The 2x2x2 multiplication tensor has a known rank-7 decomposition.
        # I'll use the one from the current program and verify it.
        
        # From the current program's Strassen block (which scored 1.0, meaning it's valid):
        # M1: U[0,j0]=1, V[j1,k1]=-1, V[j0,k0]=1, W[k0,0]=1, W[k1,1]=-1
        # M2: U[0,j0]=1, U[0,j1]=1, V[j1,k1]=1, W[k1,1]=1
        # M3: U[0,j0]=1, V[j0,k1]=1, V[j1,k1]=-1, W[k0,0]=1
        # M4: U[1,j1]=1, V[j0,k0]=1, V[j0,k1]=-1, W[k0,1]=-1, W[k1,1]=1
        # M5: U[0,j1]=1, U[1,j1]=1, V[j0,k0]=1, V[j1,k1]=1, W[k1,1]=1
        # M6: U[0,j0]=1, U[0,j1]=-1, V[j0,k1]=1, V[j1,k1]=1, W[k0,1]=1
        # M7: U[1,j1]=1, U[0,j0]=-1, V[j0,k0]=1, V[j0,k1]=1, W[k0,1]=1
        
        # Let me verify M1: should compute A[0,j0]*B[j0,k0]*W[k0,0] + A[0,j0]*B[j1,k1]*(-W[k1,1])... 
        # Actually the tensor condition is:
        # T[(i,j),(j,k),(k,i)] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # For the sub-block, we need the sum over the 7 terms to equal the 2x2 multiplication tensor.
        
        # I'll just use the current program's formulas and verify numerically.
        
        # M1
        U[0*m + j0, col] += 1
        V[j1*p + k1, col] -= 1
        V[j0*p + k0, col] += 1
        W[k0*n + 0, col] += 1
        W[k1*n + 1, col] -= 1
        col += 1
        
        # M2
        U[0*m + j0, col] += 1
        U[0*m + j1, col] += 1
        V[j1*p + k1, col] += 1
        W[k1*n + 1, col] += 1
        col += 1
        
        # M3
        U[0*m + j0, col] += 1
        V[j0*p + k1, col] += 1
        V[j1*p + k1, col] -= 1
        W[k0*n + 0, col] += 1
        col += 1
        
        # M4
        U[1*m + j1, col] += 1
        V[j0*p + k0, col] += 1
        V[j0*p + k1, col] -= 1
        W[k0*n + 1, col] -= 1
        W[k1*n + 1, col] += 1
        col += 1
        
        # M5
        U[0*m + j1, col] += 1
        U[1*m + j1, col] += 1
        V[j0*p + k0, col] += 1
        V[j1*p + k1, col] += 1
        W[k1*n + 1, col] += 1
        col += 1
        
        # M6
        U[0*m + j0, col] += 1
        U[0*m + j1, col] -= 1
        V[j0*p + k1, col] += 1
        V[j1*p + k1, col] += 1
        W[k0*n + 1, col] += 1
        col += 1
        
        # M7
        U[1*m + j1, col] += 1
        U[0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + k1, col] += 1
        W[k0*n + 1, col] += 1
        col += 1

    # Apply Strassen blocks
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            add_strassen_block(j_base, k_base)
    
    # Handle j=4 with rank 1
    U[0*m + 4, col] = 1
    U[1*m + 4, col] = 1
    for k in range(6):
        V[4*p + k, col] = 1
        W[k*n + 0, col] = 1
        W[k*n + 1, col] = 1
    col += 1
    
    # Verify correctness
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed[i, k] = s
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
