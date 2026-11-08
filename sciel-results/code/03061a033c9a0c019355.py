import numpy as np
def construct():
    n, m, p = 5, 5, 5
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)
        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val
        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())
    
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})
    
    def strassen_4x4(i0, i1, j0, j1, k0, k1):
        # Split 4x4 into 2x2 blocks
        # Rows: i0,i0+1 | i0+2,i0+3
        # Cols of A: j0,j0+1 | j0+2,j0+3
        # Cols of B: k0,k0+1 | k0+2,k0+3
        a00, a01 = (i0, i0+1), (i0+2, i0+3)
        a10, a11 = (i0, i0+1), (i0+2, i0+3)
        b00, b01 = (k0, k0+1), (k0+2, k0+3)
        b10, b11 = (k0, k0+1), (k0+2, k0+3)
        
        # Actually, for 4x4, I need to split rows and columns
        # Row split: top 2, bottom 2
        # Col split of A: left 2, right 2
        # Col split of B: left 2, right 2
        
        r0, r1 = i0, i0+2
        c0, c1 = j0, j0+2
        d0, d1 = k0, k0+2
        
        # A00: rows r0:r1, cols c0:c1
        # A01: rows r0:r1, cols c1:c1+2
        # A10: rows r1:r1+2, cols c0:c1
        # A11: rows r1:r1+2, cols c1:c1+2
        # B00: rows c0:c1, cols d0:d1
        # B01: rows c0:c1, cols d1:d1+2
        # B10: rows c1:c1+2, cols d0:d1
        # B11: rows c1:c1+2, cols d1:d1+2
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each of these is a 2x2*2x2 multiplication, done with Strassen (7 terms)
        
        # M1 = (A00+A11)(B00+B11)
        # A00+A11: rows r0:r1 and r1:r1+2, cols c0:c1 and c1:c1+2
        # This is a 4x4 matrix, but we only need the 2x2 blocks
        # Actually, (A00+A11) is a 4x4 matrix with:
        #   top-left 2x2 = A00 (top 2 rows, left 2 cols)
        #   top-right 2x2 = A01 (top 2 rows, right 2 cols)
        #   bottom-left 2x2 = A10 (bottom 2 rows, left 2 cols)
        #   bottom-right 2x2 = A11 (bottom 2 rows, right 2 cols)
        # Wait, that's not right. A00 is 2x2, A11 is 2x2, so A00+A11 doesn't make sense as a 4x4.
        
        # I think I'm overcomplicating this. Let me use a different approach.
        # For 4x4*4x4, I'll just use 7 Strassen 2x2*2x2 multiplications directly.
        
        # Actually, the standard Strassen for 4x4 is:
        # Split into 2x2 blocks: A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # where each block is 2x2.
        # Then use 7 multiplications of 2x2*2x2 matrices.
        
        # So I need to call strassen_2x2 7 times with appropriate indices.
        
        # A11: rows i0:i0+2, cols j0:j0+2
        # A12: rows i0:i0+2, cols j0+2:j0+4
        # A21: rows i0+2:i0+4, cols j0:j0+2
        # A22: rows i0+2:i0+4, cols j0+2:j0+4
        # B11: rows j0:j0+2, cols k0:k0+2
        # B12: rows j0:j0+2, cols k0+2:k0+4
        # B21: rows j0+2:j0+4, cols k0:k0+2
        # B22: rows j0+2:j0+4, cols k0+2:k0+4
        
        # M1 = (A11+A22)(B11+B22)
        # This requires multiplying a 4x4 "matrix" (which is really two 2x2 blocks) by another.
        # But we can't directly do this with our strassen_2x2 function.
        
        # I think the issue is that Strassen's algorithm for 4x4 requires multiplying 2x2 matrices,
        # but the intermediate products involve sums of 2x2 matrices, which are still 2x2.
        # So I should be able to use strassen_2x2 for each of the 7 multiplications.
        
        # Let me define the 2x2 blocks:
        # A11: i in [i0, i0+1], j in [j0, j0+1]
        # A12: i in [i0, i0+1], j in [j0+2, j0+3]
        # A21: i in [i0+2, i0+3], j in [j0, j0+1]
        # A22: i in [i0+2, i0+3], j in [j0+2, j0+3]
        # B11: j in [j0, j0+1], k in [k0, k0+1]
        # B12: j in [j0, j0+1], k in [k0+2, k0+3]
        # B21: j in [j0+2, j0+3], k in [k0, k0+1]
        # B22: j in [j0+2, j0+3], k in [k0+2, k0+3]
        
        # For M1 = (A11+A22)(B11+B22):
        # (A11+A22) has entries:
        #   (i0,j0): A11[i0,j0] = 1
        #   (i0,j0+1): A11[i0,j0+1] = 1
        #   (i0+1,j0): A11[i0+1,j0] = 1
        #   (i0+1,j0+1): A11[i0+1,j0+1] = 1
        #   (i0+2,j0+2): A22[i0+2,j0+2] = 1
        #   (i0+2,j0+3): A22[i0+2,j0+3] = 1
        #   (i0+3,j0+2): A22[i0+3,j0+2] = 1
        #   (i0+3,j0+3): A22[i0+3,j0+3] = 1
        # But this is a 4x4 matrix, not a 2x2 matrix. So I can't use strassen_2x2 directly.
        
        # I think I need a different approach. Let me just use schoolbook for the 4x4*4x4 part,
        # or find a better way.
        
        # Actually, for the purpose of this problem, let me just use the 2+3 split approach
        # that's already in the code, and try to optimize it.
        pass
    
    # Let me just use the original 2+3 split approach and see if I can find any improvements.
    # For now, let me implement the 4+1 split with schoolbook for the 4x4 parts.
    
    # C44 = A44*B44 + A41*B14
    # A44*B44: 4x4*4x4 schoolbook = 64 terms
    for i in range(4):
        for j in range(4):
            for k in range(4):
                add_term({(i,j):1}, {(j,k):1}, {(k,i):1})
    # A41*B14: 4x1*1x4 = 16 terms
    for i in range(4):
        for k in range(4):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    
    # C41 = A44*B41 + A41*B11
    # A44*B41: 4x4*4x1 = 16 terms
    for i in range(4):
        for j in range(4):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A41*B11: 4x1*1x1 = 4 terms
    for i in range(4):
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
    
    # C14 = A14*B44 + A11*B14
    # A14*B44: 1x4*4x4 = 16 terms
    for j in range(4):
        for k in range(4):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A11*B14: 1x1*1x4 = 4 terms
    for k in range(4):
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    
    # C11 = A14*B41 + A11*B11
    # A14*B41: 1x4*4x1 = 4 terms
    for j in range(4):
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # A11*B11: 1x1*1x1 = 1 term
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
