"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2_2x2(U, V, W, u_row_offset, v_row_offset, w_row_offset, col_start):
    """Add a Strassen 7-term decomposition for (2x2)(2x2) -> 2x2.
    U rows: u_row_offset..u_row_offset+1 (the 2 rows of the left 2x2 block)
    V rows: v_row_offset..v_row_offset+1 (the 2 rows of the right 2x2 block)
    W rows: w_row_offset..w_row_offset+1 (the 2 rows of the output 2x2 block)
    col_start: starting column index in U,V,W
    Returns new col_start.
    """
    # Left matrix P = [[a,b],[c,d]] indexed by U[u_row_offset+0, col], U[u_row_offset+1, col]
    # Right matrix Q = [[e,f],[g,h]] indexed by V[v_row_offset+0, col], V[v_row_offset+1, col]
    # Output R = P*Q, W[w_row_offset+0..1, col]
    
    # M1 = (a+b)(e+h)
    c = col_start
    U[u_row_offset+0][c] = 1
    U[u_row_offset+1][c] = 1
    V[v_row_offset+0][c] = 1
    V[v_row_offset+1][c] = 1
    W[w_row_offset+0][c] = 1
    W[w_row_offset+1][c] = 1
    col_start += 1
    
    # M2 = (c+d)(e)
    U[u_row_offset+0][c] = 1
    U[u_row_offset+1][c] = 1
    V[v_row_offset+0][c] = 1
    W[w_row_offset+1][c] = 1
    col_start += 1
    
    # M3 = (a)(f-h)
    U[u_row_offset+0][c] = 1
    V[v_row_offset+1][c] = 1
    V[v_row_offset+0][c] = -1
    W[w_row_offset+0][c] = 1
    W[w_row_offset+1][c] = 1
    col_start += 1
    
    # M4 = (d)(g-c)
    U[u_row_offset+1][c] = 1
    V[v_row_offset+1][c] = 1
    V[v_row_offset+0][c] = -1
    W[w_row_offset+0][c] = 1
    W[w_row_offset+1][c] = 1
    col_start += 1
    
    # M5 = (a+b)(h)
    U[u_row_offset+0][c] = 1
    U[u_row_offset+1][c] = 1
    V[v_row_offset+1][c] = 1
    W[w_row_offset+0][c] = -1
    W[w_row_offset+1][c] = 1
    col_start += 1
    
    # M6 = (c-d)(e+f)
    U[u_row_offset+0][c] = 1
    U[u_row_offset+1][c] = -1
    V[v_row_offset+0][c] = 1
    V[v_row_offset+1][c] = 1
    W[w_row_offset+1][c] = 1
    col_start += 1
    
    # M7 = (b-d)(g+h)
    U[u_row_offset+0][c] = 1
    U[u_row_offset+1][c] = -1
    V[v_row_offset+0][c] = 1
    V[v_row_offset+1][c] = 1
    W[w_row_offset+0][c] = 1
    col_start += 1
    
    return col_start


def _schoolbook_2x2_2x1(U, V, W, u_row_offset, v_row_offset, w_row_offset, col_start):
    """Add schoolbook for (2x2)(2x1) -> 2x1. Rank 4."""
    # P is 2x2 (U rows u_row_offset, u_row_offset+1), Q is 2x1 (V rows v_row_offset, v_row_offset+1)
    # Output is 2x1 (W rows w_row_offset, w_row_offset+1)
    # R[0] = P[0,0]*Q[0] + P[0,1]*Q[1]
    # R[1] = P[1,0]*Q[0] + P[1,1]*Q[1]
    for i in range(2):
        for j in range(2):
            U[u_row_offset+i][col_start] = 1
            V[v_row_offset+j][col_start] = 1
            W[w_row_offset+i][col_start] = 1
            col_start += 1
    return col_start


def _schoolbook_2x1_2x2(U, V, W, u_row_offset, v_row_offset, w_row_offset, col_start):
    """Add schoolbook for (2x1)(1x2) -> 2x2. Rank 2."""
    # P is 2x1 (U row u_row_offset), Q is 1x2 (V row v_row_offset)
    # Output is 2x2 (W rows w_row_offset, w_row_offset+1)
    # R[i,j] = P[i]*Q[j]
    for i in range(2):
        for j in range(2):
            U[u_row_offset+i][col_start] = 1
            V[v_row_offset+j][col_start] = 1
            W[w_row_offset+i][col_start] = 1
            col_start += 1
    return col_start


def _schoolbook_2x1_1x1(U, V, W, u_row_offset, v_row_offset, w_row_offset, col_start):
    """Add schoolbook for (2x1)(1x1) -> 2x1. Rank 2."""
    for i in range(2):
        U[u_row_offset+i][col_start] = 1
        V[v_row_offset][col_start] = 1
        W[w_row_offset+i][col_start] = 1
        col_start += 1
    return col_start


def _schoolbook_1x2_2x2(U, V, W, u_row_offset, v_row_offset, w_row_offset, col_start):
    """Add schoolbook for (1x2)(2x2) -> 1x2. Rank 4."""
    # P is 1x2 (U row u_row_offset), Q is 2x2 (V rows v_row_offset, v_row_offset+1)
    # Output is 1x2 (W rows w_row_offset, w_row_offset+1)
    # R[j] = P[0]*Q[0,j] + P[1]*Q[1,j]
    for j in range(2):
        for k in range(2):
            U[u_row_offset+k][col_start] = 1
            V[v_row_offset+k][col_start] = 1
            W[w_row_offset+j][col_start] = 1
            col_start += 1
    return col_start


def _schoolbook_1x2_2x1(U, V, W, u_row_offset, v_row_offset, w_row_offset, col_start):
    """Add schoolbook for (1x2)(2x1) -> 1x1. Rank 2."""
    for k in range(2):
        U[u_row_offset+k][col_start] = 1
        V[v_row_offset+k][col_start] = 1
        W[w_row_offset][col_start] = 1
        col_start += 1
    return col_start


def _schoolbook_1x1_1x2(U, V, W, u_row_offset, v_row_offset, w_row_offset, col_start):
    """Add schoolbook for (1x1)(1x2) -> 1x2. Rank 2."""
    for j in range(2):
        U[u_row_offset][col_start] = 1
        V[v_row_offset+j][col_start] = 1
        W[w_row_offset+j][col_start] = 1
        col_start += 1
    return col_start


def _schoolbook_1x1_1x1(U, V, W, u_row_offset, v_row_offset, w_row_offset, col_start):
    """Add schoolbook for (1x1)(1x1) -> 1x1. Rank 1."""
    U[u_row_offset][col_start] = 1
    V[v_row_offset][col_start] = 1
    W[w_row_offset][col_start] = 1
    col_start += 1
    return col_start


def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    # A is n x m = 3x5, B is m x p = 5x5, C is n x p = 3x5
    
    # Partition:
    # A rows: top = [0,1], bot = [2]
    # A cols / B rows: g1=[0,1], g2=[2,3], g3=[4]
    # B cols: g1=[0,1], g2=[2,3], g3=[4]
    
    rank = 67
    U = [[0] * rank for _ in range(n * m)]  # 15 x 67
    V = [[0] * rank for _ in range(m * p)]  # 25 x 67
    W = [[0] * rank for _ in range(p * n)]  # 15 x 67
    
    col = 0
    
    # A_top: rows 0,1 of A -> U rows 0..1 (cols 0-4 of A)
    # A_bot: row 2 of A -> U row 10 (col 0-4 of A, since A is stored row-major: row i, col j -> index i*m+j)
    
    # U index for A[i,j] = i*m + j = i*5 + j
    # V index for B[i,j] = i*p + j = i*5 + j
    # W index for C[i,j] = j*n + i (trace convention: W[k*n+i, r] where k is col, i is row)
    # So W index for C[i,j] = j*3 + i
    
    # A_top row 0: U indices 0,1,2,3,4
    # A_top row 1: U indices 5,6,7,8,9
    # A_bot row 2: U indices 10,11,12,13,14
    
    # B row 0: V indices 0,1,2,3,4
    # B row 1: V indices 5,6,7,8,9
    # B row 2: V indices 10,11,12,13,14
    # B row 3: V indices 15,16,17,18,19
    # B row 4: V indices 20,21,22,23,24
    
    # C row 0, col 0: W index 0*3+0 = 0
    # C row 0, col 1: W index 1*3+0 = 3
    # C row 0, col 2: W index 2*3+0 = 6
    # C row 0, col 3: W index 3*3+0 = 9
    # C row 0, col 4: W index 4*3+0 = 12
    # C row 1, col 0: W index 0*3+1 = 1
    # C row 1, col 1: W index 1*3+1 = 4
    # C row 1, col 2: W index 2*3+1 = 7
    # C row 1, col 3: W index 3*3+1 = 10
    # C row 1, col 4: W index 4*3+1 = 13
    # C row 2, col 0: W index 0*3+2 = 2
    # C row 2, col 1: W index 1*3+2 = 5
    # C row 2, col 2: W index 2*3+2 = 8
    # C row 2, col 3: W index 3*3+2 = 11
    # C row 2, col 4: W index 4*3+2 = 14
    
    # Helper to get U index for A[i,j]
    def u_idx(i, j):
        return i * m + j
    
    # Helper to get V index for B[i,j]
    def v_idx(i, j):
        return i * p + j
    
    # Helper to get W index for C[i,j]
    def w_idx(i, j):
        return j * n + i
    
    # ===== A_top (rows 0,1) x B =====
    # A_top[:,0:2] -> U rows u_idx(0,0)=0, u_idx(0,1)=1, u_idx(1,0)=5, u_idx(1,1)=6
    # A_top[:,2:4] -> U rows u_idx(0,2)=2, u_idx(0,3)=3, u_idx(1,2)=7, u_idx(1,3)=8
    # A_top[:,4:5] -> U rows u_idx(0,4)=4, u_idx(1,4)=9
    
    # B[0:2,0:2] -> V rows v_idx(0,0)=0, v_idx(0,1)=1, v_idx(1,0)=5, v_idx(1,1)=6
    # B[0:2,2:4] -> V rows v_idx(0,2)=2, v_idx(0,3)=3, v_idx(1,2)=7, v_idx(1,3)=8
    # B[0:2,4:5] -> V rows v_idx(0,4)=4, v_idx(1,4)=9
    # B[2:4,0:2] -> V rows v_idx(2,0)=10, v_idx(2,1)=11, v_idx(3,0)=15, v_idx(3,1)=16
    # B[2:4,2:4] -> V rows v_idx(2,2)=12, v_idx(2,3)=13, v_idx(3,2)=17, v_idx(3,3)=18
    # B[2:4,4:5] -> V rows v_idx(2,4)=14, v_idx(3,4)=19
    # B[4:5,0:2] -> V rows v_idx(4,0)=20, v_idx(4,1)=21
    # B[4:5,2:4] -> V rows v_idx(4,2)=22, v_idx(4,3)=23
    # B[4:5,4:5] -> V rows v_idx(4,4)=24
    
    # C[0:2,0:2] -> W rows w_idx(0,0)=0, w_idx(0,1)=3, w_idx(1,0)=1, w_idx(1,1)=4
    # C[0:2,2:4] -> W rows w_idx(0,2)=6, w_idx(0,3)=9, w_idx(1,2)=7, w_idx(1,3)=10
    # C[0:2,4:5] -> W rows w_idx(0,4)=12, w_idx(1,4)=13
    
    # For Strassen (2x2)(2x2):
    # The left 2x2 has rows [r0, r1] in U, where r0 is row 0 of the 2x2, r1 is row 1
    # The right 2x2 has rows [s0, s1] in V
    # The output 2x2 has rows [t0, t1] in W, where t0 is row 0, t1 is row 1
    
    # (g1,g1): A_top[:,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    # Left 2x2: U rows [0, 5] (row 0 and row 1 of A_top, cols 0)
    #   Actually, the 2x2 block A_top[:,0:2] has:
    #   Row 0: A[0,0], A[0,1] -> U indices 0, 1
    #   Row 1: A[1,0], A[1,1] -> U indices 5, 6
    # So for Strassen, "a" = U[0], "b" = U[1], "c" = U[5], "d" = U[6]
    # Right 2x2: B[0:2,0:2] has:
    #   Row 0: B[0,0], B[0,1] -> V indices 0, 1
    #   Row 1: B[1,0], B[1,1] -> V indices 5, 6
    # So "e" = V[0], "f" = V[1], "g" = V[5], "h" = V[6]
    # Output 2x2: C[0:2,0:2] has:
    #   Row 0: C[0,0], C[0,1] -> W indices 0, 3
    #   Row 1: C[1,0], C[1,1] -> W indices 1, 4
    # So "R[0,0]" = W[0], "R[0,1]" = W[3], "R[1,0]" = W[1], "R[1,1]" = W[4]
    
    # I need to modify my Strassen function to take explicit row indices rather than offsets.
    # Let me rewrite with explicit indices.
    
    def strassen_explicit(U, V, W, a, b, c, d, e, f, g, h, R00, R01, R10, R11, col):
        """Strassen with explicit U row indices for a,b,c,d and V row indices for e,f,g,h
        and W row indices for R00,R01,R10,R11."""
        # M1 = (a+b)(e+h)
        U[a][col] += 1
        U[b][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[R00][col] += 1
        W[R11][col] += 1
        col += 1
        
        # M2 = (c+d)(e)
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[R10][col] += 1
        col += 1
        
        # M3 = (a)(f-h)
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[R01][col] += 1
        W[R11][col] += 1
        col += 1
        
        # M4 = (d)(g-c)
        U[d][col] += 1
        V[g][col] += 1
        V[c][col] -= 1
        W[R00][col] += 1
        W[R11][col] += 1
        col += 1
        
        # M5 = (a+b)(h)
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[R00][col] -= 1
        W[R11][col] += 1
        col += 1
        
        # M6 = (c-d)(e+f)
        U[c][col] += 1
        U[d][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[R11][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[R00][col] += 1
        col += 1
        
        return col
    
    def schoolbook_explicit(U, V, W, u_rows, v_rows, w_rows, col):
        """Generic schoolbook: for each (u, v, w) triple, add a term.
        u_rows: list of U row indices
        v_rows: list of V row indices  
        w_rows: list of W row indices
        The product is sum over k of U[u_k] * V[v_k] contributing to W[w_k]
        Actually, for a (r x s)(s x t) product, we need r*s*t terms in schoolbook.
        But here I'm using it for specific small cases."""
        pass
    
    # Let me just do explicit terms for each small case.
    
    # (g1,g1): A_top[:,0:2] x B[0:2,0:2] -> C[0:2,0:2], Strassen
    # a=U[0], b=U[1], c=U[5], d=U[6]
    # e=V[0], f=V[1], g=V[5], h=V[6]
    # R00=W[0], R01=W[3], R10=W[1], R11=W[4]
    col = strassen_explicit(U, V, W, 0, 1, 5, 6, 0, 1, 5, 6, 0, 3, 1, 4, col)
    
    # (g1,g2): A_top[:,0:2] x B[0:2,2:4] -> C[0:2,2:4], Strassen
    # a=U[0], b=U[1], c=U[5], d=U[6]
    # e=V[2], f=V[3], g=V[7], h=V[8]
    # R00=W[6], R01=W[9], R10=W[7], R11=W[10]
    col = strassen_explicit(U, V, W, 0, 1, 5, 6, 2, 3, 7, 8, 6, 9, 7, 10, col)
    
    # (g1,g3): A_top[:,0:2] x B[0:2,4:5] -> C[0:2,4:5], schoolbook (2x2)(2x1) rank 4
    # A_top[:,0:2]: U[0],U[1] (row 0), U[5],U[6] (row 1)
    # B[0:2,4:5]: V[4] (row 0, col 4), V[9] (row 1, col 4)
    # C[0:2,4:5]: W[12] (row 0, col 4), W[13] (row 1, col 4)
    # R[0] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # R[1] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    U[0][col] += 1; V[4][col] += 1; W[12][col] += 1; col += 1
    U[1][col] += 1; V[9][col] += 1; W[12][col] += 1; col += 1
    U[5][col] += 1; V[4][col] += 1; W[13][col] += 1; col += 1
    U[6][col] += 1; V[9][col] += 1; W[13][col] += 1; col += 1
    
    # (g2,g1): A_top[:,2:4] x B[2:4,0:2] -> C[0:2,0:2], Strassen
    # a=U[2], b=U[3], c=U[7], d=U[8]
    # e=V[10], f=V[11], g=V[15], h=V[16]
    # R00=W[0], R01=W[3], R10=W[1], R11=W[4]
    col = strassen_explicit(U, V, W, 2, 3, 7, 8, 10, 11, 15, 16, 0, 3, 1, 4, col)
    
    # (g2,g2): A_top[:,2:4] x B[2:4,2:4] -> C[0:2,2:4], Strassen
    # a=U[2], b=U[3], c=U[7], d=U[8]
    # e=V[12], f=V[13], g=V[17], h=V[18]
    # R00=W[6], R01=W[9], R10=W[7], R11=W[10]
    col = strassen_explicit(U, V, W, 2, 3, 7, 8, 12, 13, 17, 18, 6, 9, 7, 10, col)
    
    # (g2,g3): A_top[:,2:4] x B[2:4,4:5] -> C[0:2,4:5], schoolbook (2x2)(2x1) rank 4
    # A_top[:,2:4]: U[2],U[3] (row 0), U[7],U[8] (row 1)
    # B[2:4,4:5]: V[14] (row 2, col 4), V[19] (row 3, col 4)
    # C[0:2,4:5]: W[12], W[13]
    U[2][col] += 1; V[14][col] += 1; W[12][col] += 1; col += 1
    U[3][col] += 1; V[19][col] += 1; W[12][col] += 1; col += 1
    U[7][col] += 1; V[14][col] += 1; W[13][col] += 1; col += 1
    U[8][col] += 1; V[19][col] += 1; W[13][col] += 1; col += 1
    
    # (g3,g1): A_top[:,4:5] x B[4:5,0:2] -> C[0:2,0:2], (2x1)(1x2) rank 2
    # A_top[:,4:5]: U[4] (row 0, col 4), U[9] (row 1, col 4)
    # B[4:5,0:2]: V[20] (row 4, col 0), V[21] (row 4, col 1)
    # C[0:2,0:2]: W[0], W[3], W[1], W[4]
    # R[0,0] = A[0,4]*B[4,0]
    # R[0,1] = A[0,4]*B[4,1]
    # R[1,0] = A[1,4]*B[4,0]
    # R[1,1] = A[1,4]*B[4,1]
    U[4][col] += 1; V[20][col] += 1; W[0][col] += 1; col += 1
    U[4][col] += 1; V[21][col] += 1; W[3][col] += 1; col += 1
    U[9][col] += 1; V[20][col] += 1; W[1][col] += 1; col += 1
    U[9][col] += 1; V[21][col] += 1; W[4][col] += 1; col += 1
    
    # (g3,g2): A_top[:,4:5] x B[4:5,2:4] -> C[0:2,2:4], (2x1)(1x2) rank 2
    # A_top[:,4:5]: U[4], U[9]
    # B[4:5,2:4]: V[22] (row 4, col 2), V[23] (row 4, col 3)
    # C[0:2,2:4]: W[6], W[9], W[7], W[10]
    U[4][col] += 1; V[22][col] += 1; W[6][col] += 1; col += 1
    U[4][col] += 1; V[23][col] += 1; W[9][col] += 1; col += 1
    U[9][col] += 1; V[22][col] += 1; W[7][col] += 1; col += 1
    U[9][col] += 1; V[23][col] += 1; W[10][col] += 1; col += 1
    
    # (g3,g3): A_top[:,4:5] x B[4:5,4:5] -> C[0:2,4:5], (2x1)(1x1) rank 2
    # A_top[:,4:5]: U[4], U[9]
    # B[4:5,4:5]: V[24] (row 4, col 4)
    # C[0:2,4:5]: W[12], W[13]
    U[4][col] += 1; V[24][col] += 1; W[12][col] += 1; col += 1
    U[9][col] += 1; V[24][col] += 1; W[13][col] += 1; col += 1
    
    # ===== A_bot (row 2) x B =====
    # A_bot[:,0:2]: U[10], U[11]
    # A_bot[:,2:4]: U[12], U[13]
    # A_bot[:,4:5]: U[14]
    
    # (g1,g1): A_bot[:,0:2] x B[0:2,0:2] -> C[2:3,0:2], (1x2)(2x2) rank 4
    # R[0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # R[1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    # C[2,0]=W[2], C[2,1]=W[5]
    U[10][col] += 1; V[0][col] += 1; W[2][col] += 1; col += 1
    U[11][col] += 1; V[5][col] += 1; W[2][col] += 1; col += 1
    U[10][col] += 1; V[1][col] += 1; W[5][col] += 1; col += 1
    U[11][col] += 1; V[6][col] += 1; W[5][col] += 1; col += 1
    
    # (g1,g2): A_bot[:,0:2] x B[0:2,2:4] -> C[2:3,2:4], (1x2)(2x2) rank 4
    # C[2,2]=W[8], C[2,3]=W[11]
    U[10][col] += 1; V[2][col] += 1; W[8][col] += 1; col += 1
    U[11][col] += 1; V[7][col] += 1; W[8][col] += 1; col += 1
    U[10][col] += 1; V[3][col] += 1; W[11][col] += 1; col += 1
    U[11][col] += 1; V[8][col] += 1; W[11][col] += 1; col += 1
    
    # (g1,g3): A_bot[:,0:2] x B[0:2,4:5] -> C[2:3,4:5], (1x2)(2x1) rank 2
    # C[2,4]=W[14]
    U[10][col] += 1; V[4][col] += 1; W[14][col] += 1; col += 1
    U[11][col] += 1; V[9][col] += 1; W[14][col] += 1; col += 1
    
    # (g2,g1): A_bot[:,2:4] x B[2:4,0:2] -> C[2:3,0:2], (1x2)(2x2) rank 4
    U[12][col] += 1; V[10][col] += 1; W[2][col] += 1; col += 1
    U[13][col] += 1; V[15][col] += 1; W[2][col] += 1; col += 1
    U[12][col] += 1; V[11][col] += 1; W[5][col] += 1; col += 1
    U[13][col] += 1; V[16][col] += 1; W[5][col] += 1; col += 1
    
    # (g2,g2): A_bot[:,2:4] x B[2:4,2:4] -> C[2:3,2:4], (1x2)(2x2) rank 4
    U[12][col] += 1; V[12][col] += 1; W[8][col] += 1; col += 1
    U[13][col] += 1; V[17][col] += 1; W[8][col] += 1; col += 1
    U[12][col] += 1; V[13][col] += 1; W[11][col] += 1; col += 1
    U[13][col] += 1; V[18][col] += 1; W[11][col] += 1; col += 1
    
    # (g2,g3): A_bot[:,2:4] x B[2:4,4:5] -> C[2:3,4:5], (1x2)(2x1) rank 2
    U[12][col] += 1; V[14][col] += 1; W[14][col] += 1; col += 1
    U[13][col] += 1; V[19][col] += 1; W[14][col] += 1; col += 1
    
    # (g3,g1): A_bot[:,4:5] x B[4:5,0:2] -> C[2:3,0:2], (1x1)(1x2) rank 2
    U[14][col] += 1; V[20][col] += 1; W[2][col] += 1; col += 1
    U[14][col] += 1; V[21][col] += 1; W[5][col] += 1; col += 1
    
    # (g3,g2): A_bot[:,4:5] x B[4:5,2:4] -> C[2:3,2:4], (1x1)(1x2) rank 2
    U[14][col] += 1; V[22][col] += 1; W[8][col] += 1; col += 1
    U[14][col] += 1; V[23][col] += 1; W[11][col] += 1; col += 1
    
    # (g3,g3): A_bot[:,4:5] x B[4:5,4:5] -> C[2:3,4:5], (1x1)(1x1) rank 1
    U[14][col] += 1; V[24][col] += 1; W[14][col] += 1; col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
