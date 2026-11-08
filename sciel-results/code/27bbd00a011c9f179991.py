"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W satisfy the bilinear constraints."""
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = np.dot(W[k*n+i], U[i*m+j1] * V[j2*p+k])
                        target = 1.0 if j1 == j2 else 0.0
                        max_err = max(max_err, abs(s - target))
        return max_err

    def schoolbook(n, m, p):
        """Return schoolbook decomposition for n x m times m x p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Strassen 2x2x2 (rank 7)
    def strassen_2x2x2():
        n2, m2, p2 = 2, 2, 2
        rank = 7
        U = np.zeros((n2*m2, rank))
        V = np.zeros((m2*p2, rank))
        W = np.zeros((p2*n2, rank))
        # m1 = (a11+a22)(b11+b22)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # m2 = (a21+a22)b11
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
        # m3 = a11(b12-b22)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=-1
        # m4 = a22(b21-b11)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=-1; W[2,3]=1
        # m5 = (a11+a12)b22
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[1,4]=1; W[3,4]=1
        # m6 = (a21-a11)(b11+b12)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[0,5]=1
        # m7 = (a12-a22)(b21+b22)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    # 4x4x4 via block Strassen: split into 2x2 blocks
    # A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]], C = AB
    # Use 7 Strassen products on the 2x2 block matrices, each of which is 2x2x2.
    # But actually we need to compute C = AB where A is 4x4, B is 4x4.
    # Block decomposition: A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # where each block is 2x2.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # We can apply Strassen at the block level: treat A00, A01, A10, A11 as 2x2 matrices.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M5
    #
    # Each Mi is a 2x2 matrix product computed via Strassen (rank 7 each).
    # Total rank = 7 * 7 = 49 for 4x4x4.
    #
    # Let's construct this carefully.
    def strassen_4x4x4():
        n4, m4, p4 = 4, 4, 4
        rank_total = 49
        U = np.zeros((n4*m4, rank_total))
        V = np.zeros((m4*p4, rank_total))
        W = np.zeros((p4*n4, rank_total))
        
        # For each Strassen block product Mi, we need a 2x2x2 Strassen decomposition.
        # The 2x2 block (i,j) of A corresponds to global rows (i*2+i2, j*2+j2) for i2,j2 in {0,1}
        # Similarly for B and C.
        
        # Strassen coefficients for the 7 block products:
        # Each block product M = X * Y where X, Y are 2x2.
        # The Strassen factors for M are defined by u, v, w for the 2x2x2 case.
        # We need to map the 2x2 indices to 4x4 indices.
        
        # For block product at (bi, bj) in A and (bk, bl) in B producing block (bi, bl) in C:
        # A block (bi, bj): rows bi*2+{0,1}, cols bj*2+{0,1}
        # B block (bk, bl): rows bk*2+{0,1}, cols bl*2+{0,1}
        # C block (bi, bl): rows bl*2+{0,1}, cols bi*2+{0,1}
        
        # Strassen 7 factors for a generic 2x2x2:
        # f1: X[[0,0],[1,1]] * Y[[0,0],[1,1]] -> C[[0,0],[1,1]]
        # f2: X[[1,0],[1,1]] * Y[[0,0]] -> C[[1,0],[1,1]]
        # f3: X[[0,0]] * Y[[0,1],[1,1]] with sign -> C[[0,1],[1,1]]
        # f4: X[[1,1]] * Y[[1,0],[0,0]] with sign -> C[[0,0],[1,0]]
        # f5: X[[0,0],[0,1]] * Y[[1,1]] -> C[[0,1],[1,1]]
        # f6: X[[1,0],[0,0]] with sign * Y[[0,0],[0,1]] -> C[[0,0]]
        # f7: X[[0,1],[1,1]] with sign * Y[[1,0],[1,1]] -> C[[0,0]]
        
        # Let's define the 7 Strassen patterns for 2x2x2:
        # pattern[r] = (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: list of (i,j, val) for the 2x2 X
        # V_coeffs: list of (j,k, val) for the 2x2 Y
        # W_coeffs: list of (k,i, val) for the 2x2 Z
        
        s2_patterns = [
            # f1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # f2
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),
            # f3
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,-1)]),
            # f4
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,-1),(1,0,1)]),
            # f5
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,1)]),
            # f6
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,0,1)]),
            # f7
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        # Block Strassen: 7 block products
        # For each block product s (0..6), and each 2x2 Strassen factor f (0..6):
        # Global factor index = s*7 + f
        # We need to determine which A block, B block, C block are involved.
        
        # Block Strassen definitions:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each M_s, we need:
        # - A blocks involved with signs
        # - B blocks involved with signs
        # - C blocks where result appears with signs
        
        block_strassen = [
            # (A_blocks: [(bi,bj,sign),...], B_blocks: [(bk,bl,sign),...], C_blocks: [(bi,bl,sign),...])
            # M1 = (A00+A11)(B00+B11)
            # A blocks: (0,0,1), (1,1,1)
            # B blocks: (0,0,1), (1,1,1)
            # C: M1 contributes to C00 (+1), C11 (+1)
            {
                'A': [(0,0,1),(1,1,1)],
                'B': [(0,0,1),(1,1,1)],
                'C': [(0,0,1),(1,1,1)]
            },
            # M2 = (A10+A11)B00
            {
                'A': [(1,0,1),(1,1,1)],
                'B': [(0,0,1)],
                'C': [(1,0,1),(1,1,1)]
            },
            # M3 = A00(B01-B11)
            {
                'A': [(0,0,1)],
                'B': [(0,1,1),(1,1,-1)],
                'C': [(0,1,1),(1,1,-1)]
            },
            # M4 = A11(B10-B00)
            {
                'A': [(1,1,1)],
                'B': [(1,0,1),(0,0,-1)],
                'C': [(0,0,-1),(1,0,1)]
            },
            # M5 = (A00+A01)B11
            {
                'A': [(0,0,1),(0,1,1)],
                'B': [(1,1,1)],
                'C': [(0,1,1),(1,1,1)]
            },
            # M6 = (A10-A00)(B00+B01)
            {
                'A': [(1,0,1),(0,0,-1)],
                'B': [(0,0,1),(0,1,1)],
                'C': [(0,0,1)]
            },
            # M7 = (A01-A11)(B10+B11)
            {
                'A': [(0,1,1),(1,1,-1)],
                'B': [(1,0,1),(1,1,1)],
                'C': [(0,0,1)]
            },
        ]
        
        # Now construct U, V, W for 4x4x4
        for s, bs in enumerate(block_strassen):
            for f, (U_pat, V_pat, W_pat) in enumerate(s2_patterns):
                col = s * 7 + f
                
                # For U: for each (bi, bj, sign_a) in bs['A'], and each (i2, j2, val_u) in U_pat:
                # global row = (bi*2 + i2) * 4 + (bj*2 + j2)
                # Wait, U row index is i*m + j where i is row of A, j is col of A.
                # For 4x4x4, n=m=4, so U row = i*4 + j.
                # Block A(bi,bj) has rows bi*2+{0,1}, cols bj*2+{0,1}
                # So global i = bi*2 + i2, global j = bj*2 + j2
                # U row = (bi*2+i2)*4 + (bj*2+j2)
                for (bi, bj, sign_a) in bs['A']:
                    for (i2, j2, val_u) in U_pat:
                        gi = bi * 2 + i2
                        gj = bj * 2 + j2
                        U[gi*4 + gj, col] += sign_a * val_u
                
                # For V: row = j*p + k = j*4 + k for 4x4x4
                # Block B(bk, bl) has rows bk*2+{0,1}, cols bl*2+{0,1}
                # So global j = bk*2 + j2, global k = bl*2 + k2
                # V row = (bk*2+j2)*4 + (bl*2+k2)
                for (bk, bl, sign_b) in bs['B']:
                    for (j2, k2, val_v) in V_pat:
                        gj = bk * 2 + j2
                        gk = bl * 2 + k2
                        V[gj*4 + gk, col] += sign_b * val_v
                
                # For W: row = k*n + i = k*4 + i for 4x4x4
                # Block C(bi, bl) has rows bl*2+{0,1}, cols bi*2+{0,1}
                # So global k = bl*2 + k2, global i = bi*2 + i2
                # W row = (bl*2+k2)*4 + (bi*2+i2)
                for (bi, bl, sign_c) in bs['C']:
                    for (k2, i2, val_w) in W_pat:
                        gk = bl * 2 + k2
                        gi = bi * 2 + i2
                        W[gk*4 + gi, col] += sign_c * val_w
        
        return U, V, W

    # Verify Strassen 4x4x4
    U4, V4, W4 = strassen_4x4x4()
    err4 = check_decomposition(U4, V4, W4, tol=1e-10)
    if err4 > 1e-8:
        print(f"Strassen 4x4x4 error: {err4}")
    
    # Schoolbook 4x4x1
    def schoolbook_4x4x1():
        n, m, p = 4, 4, 1
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U41, V41, W41 = schoolbook_4x4x1()
    err41 = check_decomposition(U41, V41, W41, tol=1e-10)
    if err41 > 1e-8:
        print(f"Schoolbook 4x4x1 error: {err41}")

    # Assemble 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # Total rank = 49 + 49 + 16 = 114
    rank_total = 114
    U = np.zeros((16, rank_total))
    V = np.zeros((36, rank_total))
    W = np.zeros((36, rank_total))
    
    # Block 1: k=0..3 (first 4x4x4)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4  # V rows 0..15 correspond to j=0..3, k=0..3
    W[0:16, 0:49] = W4  # W rows 0..15 correspond to k=0..3, i=0..3
    
    # Block 2: k=4..7 (second 4x4x4)
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4  # V rows 16..31 correspond to j=0..3, k=4..7
    W[16:32, 49:98] = W4  # W rows 16..31 correspond to k=4..7, i=0..3
    
    # Block 3: k=8 (4x4x1)
    U[:, 98:114] = U41
    V[32:36, 98:114] = V41  # V rows 32..35 correspond to j=0..3, k=8
    W[32:36, 98:114] = W41  # W rows 32..35 correspond to k=8, i=0..3
    
    # Verify full decomposition
    err = check_decomposition(U, V, W, tol=1e-8)
    if err > 1e-6:
        print(f"Full 4x4x9 error: {err}")
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = schoolbook(n, m, p)
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

