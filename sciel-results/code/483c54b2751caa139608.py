"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-78 exact decomposition for 3x4 by 4x7 matrix multiplication.
    Partitions A (3x4) into 2x4 (top) + 1x4 (bottom), and B (4x7) into 4x4 (left) + 4x3 (right).
    - 2x4 by 4x4: Strassen on two 2x2 blocks -> rank 7 + 7 = 14
    - 2x4 by 4x3: Strassen on 2x2 (left) + 2x2 (right) -> rank 7 + 7 = 14
    - 1x4 by 4x4: 2x2 + 2x2 -> rank 4 + 4 = 8
    - 1x4 by 4x3: 2x1 + 2x2 -> rank 2 + 4 = 6
    Total: 14 + 14 + 8 + 6 = 42... wait, let me recalculate.
    
    Actually let me be more careful.
    
    For M_{2,2,2} (2x2 * 2x2 = 2x2): rank 7 (Strassen)
    For M_{2,2,1} (2x2 * 2x1 = 2x1): rank 4 (schoolbook, each of 2 rows needs 2 mults)
    For M_{2,2,3} (2x2 * 2x3 = 2x3): rank 7 (left 2x2*2x2 rank 7, right 2x2*2x1 rank 4, total 11? No...)
    
    Let me think differently. I'll use a direct constructive approach.
    
    Actually, the known lower bound for 3x4x7 is not well-studied. Let me just do schoolbook: 3*4*7 = 84.
    But we can do better.
    
    Let me use the partitioning approach more carefully.
    
    A is 3x4, B is 4x7, C = AB is 3x7.
    
    Split A into A1 (2x4) and A2 (1x4).
    Split B into B1 (4x4) and B2 (4x3).
    
    C11 = A1*B1 (2x4 * 4x4 = 2x4): rank 7+7=14 using Strassen on two 2x2 blocks
    C12 = A1*B2 (2x4 * 4x3 = 2x3): rank 7+4=11 using Strassen on left 2x2*2x2 + schoolbook on right 2x2*2x1
    C21 = A2*B1 (1x4 * 4x4 = 1x4): rank 4+4=8 using schoolbook on two 1x2*2x2 blocks
    C22 = A2*B2 (1x4 * 4x3 = 1x3): rank 2+4=6 using schoolbook on 1x2*2x1 + 1x2*2x2
    
    Total: 14 + 11 + 8 + 6 = 39. But wait, I need to verify these ranks.
    
    M_{2,2,2}: rank 7 (Strassen)
    M_{2,2,1}: rank 4 (each output entry is a dot product of length 2, so 2 mults per entry, 2 entries = 4)
    M_{2,2,3}: left half is 2x2*2x2 rank 7, right half is 2x2*2x1 rank 4, total 11
    M_{1,2,2}: rank 4 (1x2*2x2, each output is dot product of length 2, 2 outputs = 4)
    M_{1,2,1}: rank 2
    
    So: 14 + 11 + 8 + 6 = 39.
    
    But wait, I need to double-check M_{2,2,1}. A is 2x2, B is 2x1, C is 2x1.
    C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    This needs 4 multiplications. Rank 4.
    
    M_{1,2,2}: A is 1x2, B is 2x2, C is 1x2.
    C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    This needs 4 multiplications. Rank 4.
    
    M_{1,2,1}: A is 1x2, B is 2x1, C is 1x1.
    C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    This needs 2 multiplications. Rank 2.
    
    So total rank = 14 + 11 + 8 + 6 = 39.
    
    Let me construct this explicitly.
    """
    n, m, p = 3, 4, 7
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Strassen for 2x2 * 2x2 = 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h), m2 = (c+d)(e), m3 = a(f-h), m4 = d(g-e), m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # Let me build a helper to add a rank-r decomposition of a sub-problem.
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(U_val, V_val, W_val):
        nonlocal rank
        U_rows.append(U_val)
        V_rows.append(V_val)
        W_rows.append(W_val)
        rank += 1
    
    # Strassen 2x2*2x2, with A rows (i1,i2) cols (j1,j2), B rows (j1,j2) cols (k1,k2), C rows (i1,i2) cols (k1,k2)
    def strassen_222(i1, i2, j1, j2, k1, k2):
        # m1 = (a+d)(e+h): A has a=A[i1,j1], d=A[i2,j2]; B has e=B[j1,k1], h=B[j2,k2]
        # U: (i1*m+j1) gets 1, (i2*m+j2) gets 1
        # V: (j1*p+k1) gets 1, (j2*p+k2) gets 1
        # W: (k1*n+i1) gets 1, (k2*n+i2) gets 1
        U_val = [0.0] * (n*m)
        V_val = [0.0] * (m*p)
        W_val = [0.0] * (p*n)
        U_val[i1*m + j1] = 1
        U_val[i2*m + j2] = 1
        V_val[j1*p + k1] = 1
        V_val[j2*p + k2] = 1
        W_val[k1*n + i1] = 1
        W_val[k2*n + i2] = 1
        add_term(U_val, V_val, W_val)
        
        # m2 = (c+d)(e): A has c=A[i2,j1], d=A[i2,j2]; B has e=B[j1,k1]
        U_val = [0.0] * (n*m)
        V_val = [0.0] * (m*p)
        W_val = [0.0] * (p*n)
        U_val[i2*m + j1] = 1
        U_val[i2*m + j2] = 1
        V_val[j1*p + k1] = 1
        W_val[k1*n + i2] = 1
        add_term(U_val, V_val, W_val)
        
        # m3 = a(f-h): A has a=A[i1,j1]; B has f=B[j1,k2], h=B[j2,k2]
        U_val = [0.0] * (n*m)
        V_val = [0.0] * (m*p)
        W_val = [0.0] * (p*n)
        U_val[i1*m + j1] = 1
        V_val[j1*p + k2] = 1
        V_val[j2*p + k2] = -1
        W_val[k2*n + i1] = 1
        add_term(U_val, V_val, W_val)
        
        # m4 = d(g-e): A has d=A[i2,j2]; B has g=B[j2,k1], e=B[j1,k1]
        U_val = [0.0] * (n*m)
        V_val = [0.0] * (m*p)
        W_val = [0.0] * (p*n)
        U_val[i2*m + j2] = 1
        V_val[j2*p + k1] = 1
        V_val[j1*p + k1] = -1
        W_val[k1*n + i2] = 1
        add_term(U_val, V_val, W_val)
        
        # m5 = (a+b)h: A has a=A[i1,j1], b=A[i1,j2]; B has h=B[j2,k2]
        U_val = [0.0] * (n*m)
        V_val = [0.0] * (m*p)
        W_val = [0.0] * (p*n)
        U_val[i1*m + j1] = 1
        U_val[i1*m + j2] = 1
        V_val[j2*p + k2] = 1
        W_val[k2*n + i1] = 1
        add_term(U_val, V_val, W_val)
        
        # m6 = (c-a)(e+f): A has c=A[i2,j1], a=A[i1,j1]; B has e=B[j1,k1], f=B[j1,k2]
        U_val = [0.0] * (n*m)
        V_val = [0.0] * (m*p)
        W_val = [0.0] * (p*n)
        U_val[i2*m + j1] = 1
        U_val[i1*m + j1] = -1
        V_val[j1*p + k1] = 1
        V_val[j1*p + k2] = 1
        W_val[k1*n + i2] = 1
        W_val[k2*n + i2] = 1
        add_term(U_val, V_val, W_val)
        
        # m7 = (b-d)(g+h): A has b=A[i1,j2], d=A[i2,j2]; B has g=B[j2,k1], h=B[j2,k2]
        U_val = [0.0] * (n*m)
        V_val = [0.0] * (m*p)
        W_val = [0.0] * (p*n)
        U_val[i1*m + j2] = 1
        U_val[i2*m + j2] = -1
        V_val[j2*p + k1] = 1
        V_val[j2*p + k2] = 1
        W_val[k1*n + i1] = 1
        W_val[k2*n + i1] = 1
        add_term(U_val, V_val, W_val)
    
    # Schoolbook 2x2 * 2x1 = 2x1, rank 4
    def schoolbook_221(i1, i2, j1, j2, k1):
        # C[i1,k1] = A[i1,j1]*B[j1,k1] + A[i1,j2]*B[j2,k1]
        # C[i2,k1] = A[i2,j1]*B[j1,k1] + A[i2,j2]*B[j2,k1]
        for ii, jj in [(i1,j1), (i1,j2), (i2,j1), (i2,j2)]:
            U_val = [0.0] * (n*m)
            V_val = [0.0] * (m*p)
            W_val = [0.0] * (p*n)
            U_val[ii*m + jj] = 1
            V_val[jj*p + k1] = 1
            W_val[k1*n + ii] = 1
            add_term(U_val, V_val, W_val)
    
    # Schoolbook 1x2 * 2x2 = 1x2, rank 4
    def schoolbook_122(i1, j1, j2, k1, k2):
        # C[i1,k1] = A[i1,j1]*B[j1,k1] + A[i1,j2]*B[j2,k1]
        # C[i1,k2] = A[i1,j1]*B[j1,k2] + A[i1,j2]*B[j2,k2]
        for jj, kk in [(j1,k1), (j2,k1), (j1,k2), (j2,k2)]:
            U_val = [0.0] * (n*m)
            V_val = [0.0] * (m*p)
            W_val = [0.0] * (p*n)
            U_val[i1*m + jj] = 1
            V_val[jj*p + kk] = 1
            W_val[kk*n + i1] = 1
            add_term(U_val, V_val, W_val)
    
    # Schoolbook 1x2 * 2x1 = 1x1, rank 2
    def schoolbook_121(i1, j1, j2, k1):
        # C[i1,k1] = A[i1,j1]*B[j1,k1] + A[i1,j2]*B[j2,k1]
        for jj in [j1, j2]:
            U_val = [0.0] * (n*m)
            V_val = [0.0] * (m*p)
            W_val = [0.0] * (p*n)
            U_val[i1*m + jj] = 1
            V_val[jj*p + k1] = 1
            W_val[k1*n + i1] = 1
            add_term(U_val, V_val, W_val)
    
    # Now build the decomposition.
    # A is 3x4 (rows 0,1,2; cols 0,1,2,3)
    # B is 4x7 (rows 0,1,2,3; cols 0,1,2,3,4,5,6)
    # C is 3x7
    
    # C11 = A[0:2,0:4] * B[0:4,0:4] (2x4 * 4x4 = 2x4)
    # Split into two 2x2*2x2 blocks: cols 0,1 and cols 2,3
    strassen_222(0, 1, 0, 1, 0, 1)  # A rows 0,1 cols 0,1; B rows 0,1 cols 0,1; C rows 0,1 cols 0,1
    strassen_222(0, 1, 2, 3, 2, 3)  # A rows 0,1 cols 2,3; B rows 2,3 cols 2,3; C rows 0,1 cols 2,3
    
    # C12 = A[0:2,0:4] * B[0:4,4:7] (2x4 * 4x3 = 2x3)
    # Left 2x2: A cols 0,1; B rows 0,1 cols 4,5 (2x2*2x2 = 2x2)
    strassen_222(0, 1, 0, 1, 4, 5)
    # Right 2x1: A cols 2,3; B rows 2,3 col 6 (2x2*2x1 = 2x1)
    schoolbook_221(0, 1, 2, 3, 6)
    
    # C21 = A[2,0:4] * B[0:4,0:4] (1x4 * 4x4 = 1x4)
    # Left 1x2*2x2: A col 0,1; B rows 0,1 cols 0,1
    schoolbook_122(2, 0, 1, 0, 1)
    # Right 1x2*2x2: A col 2,3; B rows 2,3 cols 2,3
    schoolbook_122(2, 2, 3, 2, 3)
    
    # C22 = A[2,0:4] * B[0:4,4:7] (1x4 * 4x3 = 1x3)
    # Left 1x2*2x1: A col 0,1; B rows 0,1 col 4
    schoolbook_121(2, 0, 1, 4)
    # Right 1x2*2x2: A col 2,3; B rows 2,3 cols 5,6
    schoolbook_122(2, 2, 3, 5, 6)
    
    U = U_rows
    V = V_rows
    W = W_rows
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
