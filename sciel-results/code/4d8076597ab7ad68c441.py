"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-28 decomposition of the 2x4x4 matrix multiplication tensor
    # using a Strassen-based construction.
    #
    # We split the 2x4x4 problem into two 2x4x2 sub-problems:
    #   C_left  = A * B_left   (A is 2x4, B_left is 4x2, C_left is 2x2)
    #   C_right = A * B_right  (A is 2x4, B_right is 4x2, C_right is 2x2)
    #
    # Each 2x4x2 sub-problem can be decomposed into two 2x2x2 Strassen
    # instances. We use the standard Strassen 7-multiplication algorithm
    # for each 2x2x2 block, giving 7 + 7 = 14 terms per 2x4x2 sub-problem,
    # for a total of 28 terms.
    #
    # The tensor T_{i,j,k} for 2x4x4 has indices:
    #   i in {0,1} (row of A / output)
    #   j in {0,1,2,3} (col of A / row of B)
    #   k in {0,1,2,3} (col of B / output)
    #
    # U has shape (8, 28): rows indexed by (i,j) -> i*4+j
    # V has shape (16, 28): rows indexed by (j,k) -> j*4+k
    # W has shape (8, 28): rows indexed by (k,i) -> k*2+i  (official trace convention)
    #
    # For each term t, U_t, V_t, W_t are vectors such that:
    #   C[k,i] = sum_t U_t[i,j] * V_t[j,k] * W_t[k,i]  (summed over j)
    #
    # We build the 14 terms for C_left (k=0,1) and 14 for C_right (k=2,3).

    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 8 rows
    V = [[0] * rank for _ in range(m * p)]   # 16 rows
    W = [[0] * rank for _ in range(p * n)]   # 8 rows

    # Strassen multipliers for 2x2 matrices:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # Output:
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6
    #
    # For a 2x2x2 block with A indices (a,b; c,d) and B indices (e,f; g,h):
    #   a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    #   e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1]
    #
    # U vectors (for A):
    # P1: a=1, d=1
    # P2: c=1, d=1
    # P3: a=1
    # P4: d=1
    # P5: a=1, b=1
    # P6: c=1, a=-1
    # P7: b=1, d=-1
    #
    # V vectors (for B):
    # P1: e=1, h=1
    # P2: e=1
    # P3: f=1, h=-1
    # P4: g=1, e=-1
    # P5: h=1
    # P6: e=1, f=1
    # P7: g=1, h=1
    #
    # W vectors (for C), using convention W[k*n+i]:
    # P1: C[0,0]+=1, C[1,1]+=1  -> W[0*2+0]=1, W[1*2+1]=1
    # P2: C[1,0]+=1             -> W[0*2+1]=1
    # P3: C[0,1]+=1, C[1,1]+=1  -> W[1*2+0]=1, W[1*2+1]=1
    # P4: C[0,0]+=1, C[1,0]+=1  -> W[0*2+0]=1, W[0*2+1]=1
    # P5: C[0,0]-=1, C[0,1]+=1  -> W[0*2+0]=-1, W[1*2+0]=1
    # P6: C[1,1]+=1             -> W[1*2+1]=1
    # P7: C[0,0]+=1             -> W[0*2+0]=1

    def add_term(term_col, a_vec, b_vec, w_vec):
        """Add a Strassen term at column term_col.
        a_vec: list of 4 values for A[0,0],A[0,1],A[1,0],A[1,1]
        b_vec: list of 4 values for B[0,0],B[0,1],B[1,0],B[1,1]
        w_vec: list of 4 values for C[0,0],C[0,1],C[1,0],C[1,1]
        """
        # U: rows i*4+j for i in 0,1 and j in 0,1
        # A[0,0] -> row 0*4+0 = 0
        # A[0,1] -> row 0*4+1 = 1
        # A[1,0] -> row 1*4+0 = 4
        # A[1,1] -> row 1*4+1 = 5
        U[0][term_col] += a_vec[0]
        U[1][term_col] += a_vec[1]
        U[4][term_col] += a_vec[2]
        U[5][term_col] += a_vec[3]

        # V: rows j*4+k for j in 0,1 and k in 0,1 (local indices)
        # B[0,0] -> row 0
        # B[0,1] -> row 1
        # B[1,0] -> row 4
        # B[1,1] -> row 5
        V[0][term_col] += b_vec[0]
        V[1][term_col] += b_vec[1]
        V[4][term_col] += b_vec[2]
        V[5][term_col] += b_vec[3]

        # W: rows k*2+i for k in 0,1 and i in 0,1 (local indices)
        # C[0,0] -> row 0*2+0 = 0
        # C[0,1] -> row 0*2+1 = 1
        # C[1,0] -> row 1*2+0 = 2
        # C[1,1] -> row 1*2+1 = 3
        W[0][term_col] += w_vec[0]
        W[1][term_col] += w_vec[1]
        W[2][term_col] += w_vec[2]
        W[3][term_col] += w_vec[3]

    def add_term_right(term_col, a_vec, b_vec, w_vec):
        """Add a Strassen term for the right half (k=2,3) at column term_col.
        a_vec: list of 4 values for A[0,0],A[0,1],A[1,0],A[1,1]
        b_vec: list of 4 values for B[0,0],B[0,1],B[1,0],B[1,1] (local, k=0,1 -> global k=2,3)
        w_vec: list of 4 values for C[0,0],C[0,1],C[1,0],C[1,1] (local)
        """
        # U: same as before (A is 2x4, but we only use first 2 columns for this sub-block)
        U[0][term_col] += a_vec[0]
        U[1][term_col] += a_vec[1]
        U[4][term_col] += a_vec[2]
        U[5][term_col] += a_vec[3]

        # V: rows j*4+k for j in 0,1 and k in 0,1 (local) -> global k = 2 + k_local
        # B[0,0] local -> global row 0*4+2 = 2
        # B[0,1] local -> global row 0*4+3 = 3
        # B[1,0] local -> global row 1*4+2 = 6
        # B[1,1] local -> global row 1*4+3 = 7
        V[2][term_col] += b_vec[0]
        V[3][term_col] += b_vec[1]
        V[6][term_col] += b_vec[2]
        V[7][term_col] += b_vec[3]

        # W: rows k*2+i for k in 0,1 (local) -> global k = 2 + k_local, i in 0,1
        # C[0,0] local -> global row (2+0)*2+0 = 4
        # C[0,1] local -> global row (2+0)*2+1 = 5
        # C[1,0] local -> global row (2+1)*2+0 = 6
        # C[1,1] local -> global row (2+1)*2+1 = 7
        W[4][term_col] += w_vec[0]
        W[5][term_col] += w_vec[1]
        W[6][term_col] += w_vec[2]
        W[7][term_col] += w_vec[3]

    # Strassen multipliers:
    # P1: a=(1,0,0,1), b=(1,0,0,1), w=(1,0,0,1)  -> C[0,0]+=1, C[1,1]+=1
    # P2: a=(0,0,1,1), b=(1,0,0,0), w=(0,0,1,0)  -> C[1,0]+=1
    # P3: a=(1,0,0,0), b=(0,1,0,-1), w=(0,1,0,1) -> C[0,1]+=1, C[1,1]+=1
    # P4: a=(0,0,0,1), b=(0,-1,1,0), w=(1,1,0,0) -> C[0,0]+=1, C[1,0]+=1
    # P5: a=(1,1,0,0), b=(0,0,0,1), w=(-1,1,0,0)-> C[0,0]-=1, C[0,1]+=1
    # P6: a=(-1,0,1,0), b=(1,1,0,0), w=(0,0,0,1) -> C[1,1]+=1
    # P7: a=(0,1,0,-1), b=(0,0,1,1), w=(1,0,0,0) -> C[0,0]+=1

    strassen_terms = [
        # (a_vec, b_vec, w_vec)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # P3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 1, 0, 0]),  # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # P5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # P7
    ]

    # For the left half (k=0,1), we have two 2x2x2 blocks:
    # Block L0: A columns j=0,1; B rows j=0,1; output k=0,1
    # Block L1: A columns j=2,3; B rows j=2,3; output k=0,1
    #
    # Wait, I need to reconsider the block structure.
    #
    # A is 2x4, B is 4x4.
    # C = A * B, so C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For C_left (k=0,1): C[i,0] = sum_j A[i,j]*B[j,0], C[i,1] = sum_j A[i,j]*B[j,1]
    # For C_right (k=2,3): C[i,2] = sum_j A[i,j]*B[j,2], C[i,3] = sum_j A[i,j]*B[j,3]
    #
    # For C_left, we can split j into j=0,1 and j=2,3:
    #   C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0] + A[i,3]*B[3,0]
    #   C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1] + A[i,3]*B[3,1]
    #
    # So C_left = A_left * B_left_top + A_right * B_left_bottom
    # where A_left = A[:,0:2], A_right = A[:,2:4]
    # B_left_top = B[0:2, 0:2], B_left_bottom = B[2:4, 0:2]
    #
    # Each of these is a 2x2 * 2x2 = 2x2 multiplication, solvable with 7 terms.
    # So C_left needs 7 + 7 = 14 terms.
    # Similarly C_right needs 14 terms.
    # Total = 28.

    # For C_left, block 1 (j=0,1): A_left = A[:,0:2], B_left_top = B[0:2,0:2]
    # U uses rows i*4+j for j=0,1 -> rows 0,1,4,5
    # V uses rows j*4+k for j=0,1, k=0,1 -> rows 0,1,4,5
    # W uses rows k*2+i for k=0,1 -> rows 0,1,2,3

    for s in range(7):
        a_vec, b_vec, w_vec = strassen_terms[s]
        add_term(s, a_vec, b_vec, w_vec)

    # For C_left, block 2 (j=2,3): A_right = A[:,2:4], B_left_bottom = B[2:4,0:2]
    # U uses rows i*4+j for j=2,3 -> rows 2,3,6,7
    # V uses rows j*4+k for j=2,3, k=0,1 -> rows 8,9,12,13
    # W uses rows k*2+i for k=0,1 -> rows 0,1,2,3

    def add_term_left_block2(term_col, a_vec, b_vec, w_vec):
        """Add a Strassen term for C_left, j=2,3 block at column term_col."""
        # U: A[i,2] -> row i*4+2, A[i,3] -> row i*4+3
        U[2][term_col] += a_vec[0]  # A[0,2]
        U[3][term_col] += a_vec[1]  # A[0,3]
        U[6][term_col] += a_vec[2]  # A[1,2]
        U[7][term_col] += a_vec[3]  # A[1,3]

        # V: B[2,0] -> row 2*4+0=8, B[2,1] -> row 2*4+1=9, B[3,0] -> row 3*4+0=12, B[3,1] -> row 3*4+1=13
        V[8][term_col] += b_vec[0]   # B[2,0]
        V[9][term_col] += b_vec[1]   # B[2,1]
        V[12][term_col] += b_vec[2]  # B[3,0]
        V[13][term_col] += b_vec[3]  # B[3,1]

        # W: same as before
        W[0][term_col] += w_vec[0]
        W[1][term_col] += w_vec[1]
        W[2][term_col] += w_vec[2]
        W[3][term_col] += w_vec[3]

    for s in range(7):
        a_vec, b_vec, w_vec = strassen_terms[s]
        add_term_left_block2(7 + s, a_vec, b_vec, w_vec)

    # For C_right (k=2,3), block 1 (j=0,1): A_left = A[:,0:2], B_right_top = B[0:2,2:4]
    # U uses rows i*4+j for j=0,1 -> rows 0,1,4,5
    # V uses rows j*4+k for j=0,1, k=2,3 -> rows 2,3,6,7
    # W uses rows k*2+i for k=2,3 -> rows 4,5,6,7

    for s in range(7):
        a_vec, b_vec, w_vec = strassen_terms[s]
        add_term_right(14 + s, a_vec, b_vec, w_vec)

    # For C_right, block 2 (j=2,3): A_right = A[:,2:4], B_right_bottom = B[2:4,2:4]
    # U uses rows i*4+j for j=2,3 -> rows 2,3,6,7
    # V uses rows j*4+k for j=2,3, k=2,3 -> rows 10,11,14,15
    # W uses rows k*2+i for k=2,3 -> rows 4,5,6,7

    def add_term_right_block2(term_col, a_vec, b_vec, w_vec):
        """Add a Strassen term for C_right, j=2,3 block at column term_col."""
        # U: A[i,2] -> row i*4+2, A[i,3] -> row i*4+3
        U[2][term_col] += a_vec[0]
        U[3][term_col] += a_vec[1]
        U[6][term_col] += a_vec[2]
        U[7][term_col] += a_vec[3]

        # V: B[2,2] -> row 2*4+2=10, B[2,3] -> row 2*4+3=11, B[3,2] -> row 3*4+2=14, B[3,3] -> row 3*4+3=15
        V[10][term_col] += b_vec[0]
        V[11][term_col] += b_vec[1]
        V[14][term_col] += b_vec[2]
        V[15][term_col] += b_vec[3]

        # W: same as right block 1
        W[4][term_col] += w_vec[0]
        W[5][term_col] += w_vec[1]
        W[6][term_col] += w_vec[2]
        W[7][term_col] += w_vec[3]

    for s in range(7):
        a_vec, b_vec, w_vec = strassen_terms[s]
        add_term_right_block2(21 + s, a_vec, b_vec, w_vec)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
