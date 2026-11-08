"""Rewritten program for 6x6x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_DEADLINE = time.time() + 145.0

# EVOLVE-BLOCK-START
def construct():
    # Use a 2x2x2 block decomposition with Strassen's algorithm for each
    # 2x2 matrix multiplication. This yields a valid rank-189 decomposition
    # for 6x6x6 matrix multiplication (3x3 = 9 blocks per dimension, 7 per
    # 2x2 block => 9*7 = 63? Wait, 3*3*3 = 27 block-mults * 7 = 189).
    
    n, m, p = DIMENSIONS
    
    # Strassen formulas for 2x2 matrix multiplication:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M2 + M5
    # C10 = M3 + M6
    # C11 = M1 - M2 + M4 + M7
    # where:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Each M_i is a rank-1 tensor with:
    # U factor: coefficients for A submatrix
    # V factor: coefficients for B submatrix
    # W factor: coefficients for C submatrix
    
    # For a 2x2 block at global position (I, J, K) where I is row-block of A (and C),
    # J is col-block of A (and row-block of B), K is col-block of B (and C):
    # A block is at rows [2I:2I+2], cols [2J:2J+2]
    # B block is at rows [2J:2J+2], cols [2K:2K+2]
    # C block is at rows [2I:2I+2], cols [2K:2K+2]
    
    # Define the 7 Strassen rank-1 components as (u_coeffs, v_coeffs, w_coeffs)
    # where coeffs are lists of (local_row, local_col, value)
    
    strassen_components = [
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: contributes to C00 with +1, C11 with +1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: contributes to C01 with +1, C10 with -1 (wait, let me re-derive)
        
        # Let me use the standard Strassen decomposition carefully:
        # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = AB
        # C00 = (a+d)(e+h) - (c+d)e + a(f-h) - d(g-e) + (a+b)h - (c-a)(e+f) + (b-d)(g+h)
        # Hmm, let me use a cleaner formulation.
        
        # Standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)*e
        # M3 = a*(f-h)
        # M4 = d*(g-e)
        # M5 = (a+b)*h
        # M6 = (c-a)*(e+f)
        # M7 = (b-d)*(g+h)
        
        # C00 = M1 - M2 + M3 + M4
        # C01 = M5 + M7
        # C10 = M6 - M7
        # C11 = M1 - M3 + M5 - M4
        
        # Let me verify:
        # C00 = ae + bg = (a+d)(e+h) - (c+d)e + a(f-h) - d(g-e) + ... 
        # Actually: (a+d)(e+h) = ae+ah+de+dh
        # (c+d)e = ce+de
        # a(f-h) = af-ah
        # d(g-e) = dg-de
        # So M1 - M2 + M3 + M4 = (ae+ah+de+dh) - (ce+de) + (af-ah) - (dg-de)
        # = ae+ah+de+dh - ce-de + af-ah - dg+de = ae + dh + af - ce - dg + de
        # Hmm that's not right. Let me just use a known correct set.
        
        # I'll use a different well-known decomposition.
    ]
    
    # Let me just hardcode the known-correct Strassen decomposition.
    # Using the standard 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M2 + M5
    # C10 = M3 + M6
    # C11 = M1 - M2 + M4 + M7
    
    # Let me verify C00 = M1 + M4 - M5 - M7:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M4 = d(g-e) = dg-de
    # M5 = (a+b)h = ah+bh
    # M7 = (b-d)(g+h) = bg+bh-dg-dh
    # M1+M4-M5-M7 = (ae+ah+de+dh) + (dg-de) - (ah+bh) - (bg+bh-dg-dh)
    # = ae+ah+de+dh + dg-de - ah-bh - bg-bh+dg+dh
    # = ae + (ah-ah) + (de-de) + dh + dg - bh - bg - bh + dg + dh
    # = ae + 2dh + 2dg - 2bh - bg
    # That's not ae+bg. So my formulas are wrong.
    
    # Let me use a verified set. I'll use:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M7
    # C10 = M6 - M7
    # C11 = M1 - M3 + M5 - M4
    
    # Verify C00: M1-M2+M3+M4
    # = (ae+ah+de+dh) - (ce+de) + (af-ah) + (dg-de)
    # = ae+ah+de+dh-ce-de+af-ah+dg-de
    # = ae + (ah-ah) + (de-de-de) + dh + af + dg - ce
    # = ae - de + dh + af + dg - ce
    # Still not right.
    
    # OK let me just look this up properly. The standard Strassen:
    # M1 = (A11 - A22)(B21 + B22)
    # M2 = (A11 + A12)B22
    # M3 = (A21 + A22)B11
    # M4 = (A22 - A11)(B11 + B12)
    # M5 = A11(B12 - B22)
    # M6 = A22(B21 - B11)
    # M7 = (A11 + A12)(B22 - B12)  -- wait this varies by source
    
    # Different sources use different numbering. Let me just use a simple correct one.
    
    # Simplest: use the fact that 2x2 matmul has tensor rank 7, and use a known
    # explicit decomposition. I'll use this verified one:
    
    # For 2x2: C = A*B
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # 7 scalar multiplications:
    # p1 = a*(e-f)
    # p2 = (a+b)*f
    # p3 = (c+d)*e
    # p4 = d*(g-e)
    # p5 = (a-d)*(f+h)
    # p6 = (b+c)*h
    # p7 = (a+b)*(e-f)  -- hmm
    
    # This is getting complicated. Let me just use a numpy-based verification
    # approach: build the 2x2 matmul tensor, then find a rank-7 decomposition
    # by known formulas. Actually, the simplest is to just use the original
    # code's approach which was verified to work (score 1.1429).
    
    # The original code used:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # with W coefficients:
    # M0: C00 += 1, C11 += 1
    # M1: C10 += 1, C11 += -1
    # M2: C01 += 1, C11 += 1
    # M3: C00 += 1, C10 += 1
    # M4: C00 += -1, C01 += 1
    # M5: C11 += 1
    # M6: C00 += 1
    
    # Let me verify this is correct by checking C00:
    # C00 contributions: M0(+1), M3(+1), M4(-1), M6(+1)
    # = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    # = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bg+bh-dg-dh)
    # = ae+ah+de+dh+dg-de-ah-bh+bg+bh-dg-dh
    # = ae + (ah-ah) + (de-de) + (dh-dh) + (dg-dg) + (-bh+bh) + bg
    # = ae + bg ✓
    
    # C01: M2(+1), M4(+1)
    # = a(f-h) + (a+b)h = af-ah+ah+bh = af+bh ✓
    
    # C10: M1(+1), M3(+1)
    # = (c+d)e + d(g-e) = ce+de+dg-de = ce+dg ✓
    
    # C11: M0(+1), M1(-1), M2(+1), M5(+1)
    # = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
    # = (ae+ah+de+dh) - (ce+de) + (af-ah) + (ce+cf-ae-af)
    # = ae+ah+de+dh-ce-de+af-ah+ce+cf-ae-af
    # = (ae-ae) + (ah-ah) + (de-de) + dh + (-ce+ce) + (af-af) + cf
    # = dh + cf ✓
    
    # Great, so the original decomposition is correct. Let me rebuild it cleanly.
    
    # The 7 Strassen components with verified W coefficients:
    strassen = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: list of (row, col, val) for 2x2 A block
        # v_coeffs: list of (row, col, val) for 2x2 B block
        # w_coeffs: list of (row, col, val) for 2x2 C block
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),      # M0
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),              # M1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),              # M2
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),              # M3
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),              # M4
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),              # M5
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),              # M6
    ]
    
    block = 2
    nb = n // block  # 3
    mb = m // block  # 3
    pb = p // block  # 3
    
    total_rank = nb * mb * pb * len(strassen)  # 3*3*3*7 = 189
    
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    col = 0
    for I in range(nb):
        for J in range(mb):
            for K in range(pb):
                for u_c, v_c, w_c in strassen:
                    for (ri, ci, val) in u_c:
                        U[I*block+ri, J*block+ci, ]  # can't do 3D index on 2D
                    # Need to use row index = I*block+ri, col index = J*block+ci for U
                    # U is (n*m, rank), so row = (I*block+ri)*m + (J*block+ci)
                    for (ri, ci, val) in u_c:
                        U[(I*block+ri)*m + (J*block+ci), col] += val
                    for (ri, ci, val) in v_c:
                        V[(J*block+ri)*p + (K*block+ci), col] += val
                    for (ri, ci, val) in w_c:
                        # W index: k*n + i where k is col of C, i is row of C
                        W[(K*block+ci)*n + (I*block+ri), col] += val
                    col += 1
    
    # Convert to lists for the expected return format
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
