import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 Strassen-based decomposition
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Local indices for 2x2 blocks
    # A: (0,0), (0,1), (1,0), (1,1) -> global i, j
    # B: (0,0), (0,1), (1,0), (1,1) -> global j, k
    # C: (0,0), (0,1), (1,0), (1,1) -> global k, i
    
    # Map local (row, col) to global index offset
    # A block (bm): i in {0,1}, j in {2*bm, 2*bm+1}
    # B block (bm, bp): j in {2*bm, 2*bm+1}, k in {2*bp, 2*bp+1}
    # C block (bp): k in {2*bp, 2*bp+1}, i in {0,1}
    
    for bm in range(2):
        for bp in range(4):
            base_rank = (bm * 4 + bp) * 7
            for t in range(7):
                r = base_rank + t
                
                # Determine coefficients for U, V, W
                # We add to U[i*m+j], V[j*p+k], W[k*n+i]
                
                # Helper to add contribution
                def add_u(i, j, val):
                    U[i*m+j][r] += val
                def add_v(j, k, val):
                    V[j*p+k][r] += val
                def add_w(k, i, val):
                    W[k*n+i][r] += val
                
                # Local indices
                # A00: i=0, j=0 -> global i=0, j=2*bm
                # A01: i=0, j=1 -> global i=0, j=2*bm+1
                # A10: i=1, j=0 -> global i=1, j=2*bm
                # A11: i=1, j=1 -> global i=1, j=2*bm+1
                
                # B00: j=0, k=0 -> global j=2*bm, k=2*bp
                # B01: j=0, k=1 -> global j=2*bm, k=2*bp+1
                # B10: j=1, k=0 -> global j=2*bm+1, k=2*bp
                # B11: j=1, k=1 -> global j=2*bm+1, k=2*bp+1
                
                # C00: k=0, i=0 -> global k=2*bp, i=0
                # C01: k=1, i=0 -> global k=2*bp+1, i=0
                # C10: k=0, i=1 -> global k=2*bp, i=1
                # C11: k=1, i=1 -> global k=2*bp+1, i=1
                
                if t == 0: # M1
                    add_u(0, 2*bm, 1)
                    add_u(1, 2*bm+1, 1)
                    add_v(2*bm, 2*bp, 1)
                    add_v(2*bm+1, 2*bp+1, 1)
                    add_w(2*bp, 0, 1)
                    add_w(2*bp+1, 1, 1)
                elif t == 1: # M2
                    add_u(1, 2*bm, 1)
                    add_u(1, 2*bm+1, 1)
                    add_v(2*bm, 2*bp, 1)
                    add_w(2*bp, 1, 1)
                elif t == 2: # M3
                    add_u(0, 2*bm, 1)
                    add_v(2*bm+1, 2*bp, 1)
                    add_v(2*bm+1, 2*bp+1, -1)
                    add_w(2*bp+1, 0, 1)
                elif t == 3: # M4
                    add_u(1, 2*bm+1, 1)
                    add_v(2*bm, 2*bp+1, 1)
                    add_v(2*bm, 2*bp, -1)
                    add_w(2*bp, 0, 1)
                    add_w(2*bp, 1, 1)
                elif t == 4: # M5
                    add_u(0, 2*bm, 1)
                    add_u(0, 2*bm+1, 1)
                    add_v(2*bm+1, 2*bp+1, 1)
                    add_w(2*bp, 0, -1)
                    add_w(2*bp+1, 0, 1)
                elif t == 5: # M6
                    add_u(1, 2*bm, 1)
                    add_u(0, 2*bm, -1)
                    add_v(2*bm, 2*bp, 1)
                    add_v(2*bm, 2*bp+1, 1)
                    add_w(2*bp+1, 1, 1)
                elif t == 6: # M7
                    add_u(0, 2*bm+1, 1)
                    add_u(1, 2*bm+1, -1)
                    add_v(2*bm+1, 2*bp, 1)
                    add_v(2*bm+1, 2*bp+1, 1)
                    add_w(2*bp, 0, 1)
    
    return {"U": U, "V": V, "W": W}
