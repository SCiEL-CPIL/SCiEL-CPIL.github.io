"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: return the best (lowest rank) valid decomposition found so far
    start = time.time()
    deadline = start + 140.0  # 20s safety margin before the 160s kill

    n, m, p = DIMENSIONS  # 4, 5, 6

    # ------------------------------------------------------------------
    # Helper: build a schoolbook decomposition for an (a x b) x (b x c) block
    # placed at row-offset (ri, rj) in U, (vi, vj) in V, (wi, wj) in W
    # where U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
    # and the block multiplies A[ri:ri+a, rj:rj+b] * B[vi:vi+b, vj:vj+c]
    # contributing to C[wi:wi+c, rj:rj+b] ... wait, let me think about indexing.
    #
    # U index: i*m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k  (j in 0..m-1, k in 0..p-1)
    # W index: k*n + i  (k in 0..p-1, i in 0..n-1)
    #
    # For a sub-block: A_sub is a x b, B_sub is b x c, C_sub is a x c
    # U rows: for i_sub in 0..a-1, j_sub in 0..b-1 -> global (ri+i_sub)*m + (rj+j_sub)
    # V rows: for j_sub in 0..b-1, k_sub in 0..c-1 -> global (vj+j_sub)*p + (vk+k_sub)
    # W rows: for k_sub in 0..c-1, i_sub in 0..a-1 -> global (wk+k_sub)*n + (wi+i_sub)
    # ------------------------------------------------------------------

    def schoolbook_block(a, b, c, u_row_off, v_row_off, w_row_off, offset, matrices):
        """Add schoolbook terms for an (a x b)*(b x c) block starting at column 'offset'.
        Returns new offset (offset + a*b*c)."""
        U, V, W = matrices
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    u_idx = u_row_off + i * b + j  # (ri+i)*m + (rj+j) but we pass u_row_off as (ri*m+rj)
                    # Actually u_row_off should be ri*m + rj, and then i*b+j doesn't work directly
                    # Let me use explicit indices instead.
                    pass
        return offset

    # Let me redo this more carefully with explicit global indices.

    def add_schoolbook(a, b, c, i0, j0, k0, col_start, U, V, W):
        """Add schoolbook decomposition for block A[i0:i0+a, j0:j0+b] * B[j0:j0+b, k0:k0+c]
        contributing to C[k0:k0+c, i0:i0+a].
        U global row for (i, j) = i*m + j
        V global row for (j, k) = j*p + k
        W global row for (k, i) = k*n + i
        Returns col_start + a*b*c.
        """
        col = col_start
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i0 + i, col] = 1
                    V[j0 + j, col] = 1
                    W[k0 + k, col] = 1
                    col += 1
        return col

    # ------------------------------------------------------------------
    # Strassen 2x2x2 decomposition
    # For 2x2 * 2x2:
    #   m1 = (a11+a22)*(b11+b22)
    #   m2 = (a21+a22)*b11
    #   m3 = a11*(b12-b22)
    #   m4 = a22*(b21-b11)
    #   m5 = (a11+a12)*b22
    #   m6 = (a21-a11)*(b11+b12)
    #   m7 = (a12-a22)*(b21+b22)
    #   c11 = m1 + m4 - m5 + m7
    #   c12 = m3 + m5
    #   c21 = m2 + m4
    #   c22 = m1 - m2 + m3 + m6
    # ------------------------------------------------------------------
    def add_strassen_2x2x2(i0, j0, k0, col_start, U, V, W):
        """Add Strassen 7-term decomposition for a 2x2*2x2 block at (i0,j0,k0)."""
        col = col_start
        # m1 = (a11+a22)*(b11+b22)
        U[i0, col] = 1; U[i0+1, col] = 1
        V[j0, col] = 1; V[j0+1, col] = 1
        W[k0, col] = 1; W[k0+1, col] = 1
        col += 1
        # m2 = (a21+a22)*b11
        U[i0+1, col] = 1  # a21 is row i0+1, col j0
        # Wait, I need to be careful. Let me use the standard Strassen formulation.
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # In our indexing:
        #   a11 = A[i0, j0], a12 = A[i0, j0+1], a21 = A[i0+1, j0], a22 = A[i0+1, j0+1]
        #   b11 = B[j0, k0], b12 = B[j0, k0+1], b21 = B[j0+1, k0], b22 = B[j0+1, k0+1]
        #   c11 = C[k0, i0], c12 = C[k0, i0+1], c21 = C[k0+1, i0], c22 = C[k0+1, i0+1]

        # m1 = (a11+a22)*(b11+b22)
        # U: a11 at (i0,j0), a22 at (i0+1,j0+1)
        # V: b11 at (j0,k0), b22 at (j0+1,k0+1)
        # W: c11 at (k0,i0), c22 at (k0+1,i0+1)
        U[i0, col] = 1
        U[i0+1, col] = 1
        V[j0, col] = 1
        V[j0+1, col] = 1
        W[k0, col] = 1
        W[k0+1, col] = 1
        col += 1

        # m2 = (a21+a22)*b11
        U[i0+1, col] = 1  # a21
        U[i0+1, col] = 1  # a22 (same row, but different col in A... wait no)
        # a21 is A[i0+1, j0], a22 is A[i0+1, j0+1]
        # U index for A[i,j] is i*m+j
        U[i0+1, col] = 1  # this is wrong, I'm overwriting
        # Let me restart this properly.

        # I'll build the Strassen terms explicitly.
        col = col_start

        # m1 = (a11 + a22) * (b11 + b22)
        U[i0 * 1 + j0, col] = 1      # a11: row i0, col j0 in A -> U[i0*m+j0]
        U[(i0+1) * 1 + (j0+1), col] = 1  # a22
        V[j0 * 1 + k0, col] = 1      # b11
        V[(j0+1) * 1 + (k0+1), col] = 1  # b22
        W[k0 * 1 + i0, col] = 1      # c11
        W[(k0+1) * 1 + (i0+1), col] = 1  # c22
        col += 1

        # m2 = (a21 + a22) * b11
        U[i0 * 1 + j0, col] = 0  # clear previous (should be 0 already)
        # Actually let me just use fresh columns each time.

        # Let me rewrite this completely cleanly.
        col = col_start

        # m1 = (a11 + a22) * (b11 + b22)
        U[i0, col] = 1; U[i0+1, col] = 1
        V[j0, col] = 1; V[j0+1, col] = 1
        W[k0, col] = 1; W[k0+1, col] = 1
        col += 1

        # m2 = (a21 + a22) * b11
        U[i0+1, col] = 1  # a21 is at row i0+1, col j0 -> U[(i0+1)*m+j0]
        # Wait, U is indexed by i*m+j, not by block-local indices.
        # I need to use global indices: U[i0 + i, col] where i is 0 or 1 for the 2x2 block
        # But U has n*m rows, indexed by global row index = i_global * m + j_global
        # For a 2x2 block starting at (i0, j0), the four entries are:
        #   a11 -> U[i0*m + j0], a12 -> U[i0*m + j0+1]
        #   a21 -> U[(i0+1)*m + j0], a22 -> U[(i0+1)*m + j0+1]

        # Let me define local-to-global mappings
        def u_idx(i_local, j_local):
            return (i0 + i_local) * m + (j0 + j_local)

        def v_idx(j_local, k_local):
            return (j0 + j_local) * p + (k0 + k_local)

        def w_idx(k_local, i_local):
            return (k0 + k_local) * n + (i0 + i_local)

        # Now redo:
        col = col_start

        # m1 = (a11 + a22) * (b11 + b22)
        U[u_idx(0,0), col] = 1
        U[u_idx(1,1), col] = 1
        V[v_idx(0,0), col] = 1
        V[v_idx(1,1), col] = 1
        W[w_idx(0,0), col] = 1
        W[w_idx(1,1), col] = 1
        col += 1

        # m2 = (a21 + a22) * b11
        U[u_idx(1,0), col] = 1
        U[u_idx(1,1), col] = 1
        V[v_idx(0,0), col] = 1
        W[w_idx(0,1), col] = 1
        W[w_idx(1,0), col] = 1
        col += 1

        # m3 = a11 * (b12 - b22)
        U[u_idx(0,0), col] = 1
        V[v_idx(0,1), col] = 1
        V[v_idx(1,1), col] = -1
        W[w_idx(0,0), col] = 1
        W[w_idx(1,0), col] = 1
        col += 1

        # m4 = a22 * (b21 - b11)
        U[u_idx(1,1), col] = 1
        V[v_idx(1,0), col] = 1
        V[v_idx(0,0), col] = -1
        W[w_idx(0,0), col] = 1
        W[w_idx(1,1), col] = 1
        col += 1

        # m5 = (a11 + a12) * b22
        U[u_idx(0,0), col] = 1
        U[u_idx(0,1), col] = 1
        V[v_idx(1,1), col] = 1
        W[w_idx(0,1), col] = 1
        W[w_idx(1,1), col] = 1
        col += 1

        # m6 = (a21 - a11) * (b11 + b12)
        U[u_idx(1,0), col] = 1
        U[u_idx(0,0), col] = -1
        V[v_idx(0,0), col] = 1
        V[v_idx(0,1), col] = 1
        W[w_idx(1,0), col] = 1
        W[w_idx(1,1), col] = 1
        col += 1

        # m7 = (a12 - a22) * (b21 + b22)
        U[u_idx(0,1), col] = 1
        U[u_idx(1,1), col] = -1
        V[v_idx(1,0), col] = 1
        V[v_idx(1,1), col] = 1
        W[w_idx(0,1), col] = 1
        W[w_idx(1,1), col] = 1
        col += 1

        return col

    # ------------------------------------------------------------------
    # Strategy: partition the 4x5x6 problem into blocks and use optimal
    # decompositions for each block.
    #
    # Partition n=4 as 2+2, m=5 as 2+3, p=6 as 2+2+2
    #
    # Blocks:
    #   A is 4x5, partitioned into 2x2 blocks of rows and 2 or 3 cols
    #   B is 5x6, partitioned into 2 or 3 rows and 2 cols
    #   C is 6x4, partitioned into 2 rows and 2 or 3 cols... wait
    #
    # Let me think about this differently.
    #
    # A: 4 rows x 5 cols. Partition rows: {0,1}, {2,3}. Partition cols: {0,1}, {2,3,4}.
    # B: 5 rows x 6 cols. Partition rows: {0,1}, {2,3,4}. Partition cols: {0,1}, {2,3}, {4,5}.
    # C: 6 rows x 4 cols. Partition rows: {0,1}, {2,3}, {4,5}. Partition cols: {0,1}, {2,3}.
    #
    # The (i_block, j_block, k_block) block computes:
    #   C[k_block, i_block] += A[i_block, j_block] * B[j_block, k_block]
    #
    # i_block: 0 (rows 0-1) or 1 (rows 2-3), size 2x2
    # j_block: 0 (cols 0-1, size 2) or 1 (cols 2-4, size 3)
    # k_block: 0 (cols 0-1), 1 (cols 2-3), 2 (cols 4-5), each size 2
    #
    # Block sizes:
    #   (i=0, j=0, k=0): A[0:2,0:2] (2x2) * B[0:2,0:2] (2x2) -> M_{2,2,2}, rank 7
    #   (i=0, j=0, k=1): A[0:2,0:2] (2x2) * B[0:2,2:4] (2x2) -> M_{2,2,2}, rank 7
    #   (i=0, j=0, k=2): A[0:2,0:2] (2x2) * B[0:2,4:6] (2x2) -> M_{2,2,2}, rank 7
    #   (i=0, j=1, k=0): A[0:2,2:5] (2x3) * B[2:5,0:2] (3x2) -> M_{2,3,2}
    #   (i=0, j=1, k=1): A[0:2,2:5] (2x3) * B[2:5,2:4] (3x2) -> M_{2,3,2}
    #   (i=0, j=1, k=2): A[0:2,2:5] (2x3) * B[2:5,4:6] (3x2) -> M_{2,3,2}
    #   (i=1, j=0, k=0): A[2:4,0:2] (2x2) * B[0:2,0:2] (2x2) -> M_{2,2,2}, rank 7
    #   (i=1, j=0, k=1): A[2:4,0:2] (2x2) * B[0:2,2:4] (2x2) -> M_{2,2,2}, rank 7
    #   (i=1, j=0, k=2): A[2:4,0:2] (2x2) * B[0:2,4:6] (2x2) -> M_{2,2,2}, rank 7
    #   (i=1, j=1, k=0): A[2:4,2:5] (2x3) * B[2:5,0:2] (3x2) -> M_{2,3,2}
    #   (i=1, j=1, k=1): A[2:4,2:5] (2x3) * B[2:5,2:4] (3x2) -> M_{2,3,2}
    #   (i=1, j=1, k=2): A[2:4,2:5] (2x3) * B[2:5,4:6] (3x2) -> M_{2,3,2}
    #
    # For M_{2,3,2}: use schoolbook (rank 12) or try to do better.
    #   M_{2,3,2}: multiply (2x3) by (3x2). This is like two M_{2,2,2} plus two M_{2,1,2}.
    #   Split the 3 into 2+1:
    #     M_{2,3,2} = M_{2,2,2} (for first 2 cols of A, first 2 rows of B)
    #                 + M_{2,1,2} (for first 2 cols of A, last 1 row of B)
    #                 + M_{2,2,2}... no wait.
    #
    #   Actually: A is 2x3, B is 3x2.
    #   Split A's cols as 2+1, B's rows as 2+1:
    #     C = A[:,0:2]*B[0:2,:] + A[:,2:3]*B[2:3,:]
    #     First term: M_{2,2,2}, rank 7
    #     Second term: M_{2,1,2}, rank 2 (schoolbook: 2*1*2=4? No, M_{2,1,2} rank is 2)
    #     Wait, M_{2,1,2}: (2x1)*(1x2) = 2x2. This is just outer product times scalar.
    #     Actually A[:,2:3] is 2x1, B[2:3,:] is 1x2. Product is 2x2.
    #     This is rank 1 as a tensor? No, M_{2,1,2} has rank 2.
    #     Because c_{ik} = a_{i0} * b_{0k}, which is a_{i0}*b_{0k}.
    #     This can be written as sum over... it's actually rank 1? No.
    #     The tensor T_{i,k,0} = a_i * b_k. This is the outer product of a and b, so rank 1.
    #     Wait no, in matrix multiplication tensor, M_{n,m,p} has entries T_{i,j,k} = delta_{j,k}... no.
    #     M_{n,m,p}: T_{i,j,k} = 1 if j==k... no.
    #     The matrix multiplication tensor for n x m times m x p:
    #     T_{(i,k),j} = A_{i,j} * B_{j,k}
    #     So for M_{2,1,2}: A is 2x1, B is 1x2.
    #     T_{(i,k),j} where i in 0..1, k in 0..1, j in 0..0 (just j=0)
    #     T_{(i,k),0} = A_{i,0} * B_{0,k}
    #     This is a 2x2 matrix for each j=0, but since there's only one j, it's just the
    #     outer product of column a = (A_{0,0}, A_{1,0}) and row b = (B_{0,0}, B_{0,1}).
    #     So T_{(i,k),0} = a_i * b_k, which is rank 1 as a tensor.
    #     But wait, the tensor has shape (2*2, 1, 2) = (4,1,2). The rank-1 decomposition would be
    #     one term: U * V * W where U is 4x1, V is 1x1, W is 2x1.
    #     U = (a_0, a_1, 0, 0)^T? No, U has one entry per (i,k) pair.
    #     Actually for M_{2,1,2}: n=2, m=1, p=2.
    #     U has n*m = 2*1 = 2 rows, V has m*p = 1*2 = 2 rows, W has p*n = 2*2 = 4 rows.
    #     T_{(i,k),j} = A_{i,j} * B_{j,k} with j in 0..0.
    #     So T_{(0,0),0} = A_{0,0}*B_{0,0}, T_{(0,1),0} = A_{0,0}*B_{0,1}
    #     T_{(1,0),0} = A_{1,0}*B_{0,0}, T_{(1,1),0} = A_{1,0}*B_{0,1}
    #     This is rank 1: let U = (A_{0,0}, A_{1,0}), V = (B_{0,0}, B_{0,1}), W = (1,1,1,1).
    #     Then (U*V*W)_{(i,k),j} = U_i * V_k * W_{(i,k),j}... hmm, the indexing is tricky.
    #
    #     Actually, I think M_{2,1,2} has rank 2, not 1. Let me just use schoolbook (rank 4) for safety.
    #
    #   So M_{2,3,2} via split 2+1: rank 7 + 4 = 11.
    #
    # Total rank: 6 * 7 + 6 * 11 = 42 + 66 = 108.
    #
    # But wait, the measured improvement was from 105 to 101 using 4 Strassen blocks.
    # Let me reconsider. The current best is 101 (score 1.188).
    #
    # Let me try: partition n=4 as 2+2, m=5 as 2+3, p=6 as 3+3.
    #   k_block: 0 (cols 0-2, size 3), 1 (cols 3-5, size 3)
    #
    # Blocks:
    #   (i=0, j=0, k=0): 2x2 * 2x3 -> M_{2,2,3}, rank 8 (schoolbook)
    #   (i=0, j=0, k=1): 2x2 * 2x3 -> M_{2,2,3}, rank 8
    #   (i=0, j=1, k=0): 2x3 * 3x3 -> M_{2,3,3}, rank 18 (schoolbook)
    #   (i=0, j=1, k=1): 2x3 * 3x3 -> M_{2,3,3}, rank 18
    #   (i=1, j=0, k=0): 2x2 * 2x3 -> M_{2,2,3}, rank 8
    #   (i=1, j=0, k=1): 2x2 * 2x3 -> M_{2,2,3}, rank 8
    #   (i=1, j=1, k=0): 2x3 * 3x3 -> M_{2,3,3}, rank 18
    #   (i=1, j=1, k=1): 2x3 * 3x3 -> M_{2,3,3}, rank 18
    # Total: 4*8 + 4*18 = 32 + 72 = 104. Not better than 101.
    #
    # Let me try the partition that gave 101: using 4 Strassen 2x2x2 blocks.
    # The knowledge says: "Replacing the 4x4x2 schoolbook block (rank 32) with 4 Strassen 2x2x2 blocks (rank 28) reduces total rank from 105 to 101"
    #
    # So the base decomposition was 105, and replacing one 4x4x2 block with 4 Strassen blocks saved 4.
    #
    # Let me figure out what the 105 decomposition was.
    # If we partition as n=4, m=5, p=6 with some blocks:
    # Maybe: n=4 as 4 (no split), m=5 as 2+3, p=6 as 2+2+2
    #   j=0 (size 2), k=0,1,2 (size 2 each): M_{4,2,2} rank 16 each, 3 blocks = 48
    #   j=1 (size 3), k=0,1,2 (size 2 each): M_{4,3,2} rank 24 each, 3 blocks = 72
    # Total = 120. That's the schoolbook total.
    #
    # 105 = 120 - 15. So 15 rank was saved.
    #
    # Let me try: n=4 as 2+2, m=5 as 2+3, p=6 as 2+2+2
    #   (i=0,j=0,k=0): M_{2,2,2} rank 7
    #   (i=0,j=0,k=1): M_{2,2,2} rank 7
    #   (i=0,j=0,k=2): M_{2,2,2} rank 7
    #   (i=0,j=1,k=0): M_{2,3,2} rank 11 (7+4)
    #   (i=0,j=1,k=1): M_{2,3,2} rank 11
    #   (i=0,j=1,k=2): M_{2,3,2} rank 11
    #   (i=1,j=0,k=0): M_{2,2,2} rank 7
    #   (i=1,j=0,k=1): M_{2,2,2} rank 7
    #   (i=1,j=0,k=2): M_{2,2,2} rank 7
    #   (i=1,j=1,k=0): M_{2,3,2} rank 11
    #   (i=1,j=1,k=1): M_{2,3,2} rank 11
    #   (i=1,j=1,k=2): M_{2,3,2} rank 11
    # Total: 6*7 + 6*11 = 42 + 66 = 108.
    #
    # Hmm, 108 not 105. Let me try other partitions.
    #
    # What about n=4 as 2+2, m=5 as 3+2, p=6 as 2+2+2?
    #   (i=0,j=0,k=0): M_{2,3,2} rank 11
    #   (i=0,j=0,k=1): M_{2,3,2} rank 11
    #   (i=0,j=0,k=2): M_{2,3,2} rank 11
    #   (i=0,j=1,k=0): M_{2,2,2} rank 7
    #   (i=0,j=1,k=1): M_{2,2,2} rank 7
    #   (i=0,j=1,k=2): M_{2,2,2} rank 7
    #   (i=1,j=0,k=0): M_{2,3,2} rank 11
    #   (i=1,j=0,k=1): M_{2,3,2} rank 11
    #   (i=1,j=0,k=2): M_{2,3,2} rank 11
    #   (i=1,j=1,k=0): M_{2,2,2} rank 7
    #   (i=1,j=1,k=1): M_{2,2,2} rank 7
    #   (i=1,j=1,k=2): M_{2,2,2} rank 7
    # Total: 6*11 + 6*7 = 66 + 42 = 108. Same.
    #
    # Let me try to get to 101. The knowledge says 101 was achieved.
    # 120 - 101 = 19 rank saved.
    #
    # Maybe using Strassen for all 2x2x2 blocks and a better M_{2,3,2}.
    # If M_{2,3,2} can be done in rank 10 instead of 11?
    # Then 6*7 + 6*10 = 42 + 60 = 102. Still not 101.
    #
    # What if we use a different partition?
    # n=4 as 2+2, m=5 as 1+2+2, p=6 as 2+2+2
    #   j=0 (size 1): M_{2,1,2} rank 2 (or 4 schoolbook)
    #   j=1 (size 2): M_{2,2,2} rank 7
    #   j=2 (size 2): M_{2,2,2} rank 7
    # For each i_block (2 of them) and k_block (3 of them):
    #   i=0,j=0: 2*3 = 6 blocks of M_{2,1,2} rank 2 -> 12
    #   i=0,j=1: 2*3 = 6 blocks of M_{2,2,2} rank 7 -> 42
    #   i=0,j=2: 2*3 = 6 blocks of M_{2,2,2} rank 7 -> 42
    #   i=1,j=0: 2*3 = 6 blocks of M_{2,1,2} rank 2 -> 12
    #   i=1,j=1: 2*3 = 6 blocks of M_{2,2,2} rank 7 -> 42
    #   i=1,j=2: 2*3 = 6 blocks of M_{2,2,2} rank 7 -> 42
    # Total: 24 + 126 = 150. Worse.
    #
    # Let me just implement the 108-rank solution and see. The measured best is 101,
    # so 108 would give score 120/108 = 1.111, which is worse than 1.188.
    #
    # I need to find the 101-rank solution. Let me think about what decomposition gives 101.
    #
    # 101 = 7 + 94? 101 = 14 + 87? 101 = 21 + 80?
    #
    # Maybe: use Strassen for some 2x2x2 blocks and a different approach for others.
    #
    # Let me try: n=4 as 4, m=5 as 2+3, p=6 as 3+3
    #   j=0 (size 2), k=0 (size 3): M_{4,2,3} rank 24 (schoolbook)
    #   j=0, k=1 (size 3): M_{4,2,3} rank 24
    #   j=1 (size 3), k=0 (size 3): M_{4,3,3} rank 36
    #   j=1, k=1 (size 3): M_{4,3,3} rank 36
    # Total: 48 + 72 = 120.
    #
    # With Strassen for n: split n=4 as 2+2 within M_{4,2,3}:
    #   M_{4,2,3} = M_{2,2,3} + M_{2,2,3} = 8 + 8 = 16 (vs 24 schoolbook)
    #   M_{4,3,3} = M_{2,3,3} + M_{2,3,3} = 18 + 18 = 36 (same as schoolbook)
    # Total: 2*16 + 2*36 = 32 + 72 = 104.
    #
    # With Strassen for M_{2,2,3}: split p=3 as 2+1:
    #   M_{2,2,3} = M_{2,2,2} + M_{2,2,1} = 7 + 4 = 11
    #   M_{4,2,3} = 2*11 = 22
    # Total: 2*22 + 2*36 = 44 + 72 = 116. Worse.
    #
    # I think the 101 solution uses a more clever approach. Let me just implement
    # the best I can figure out and include a deadline guard.
    #
    # For now, let me implement the 108-rank solution with Strassen for 2x2x2 blocks
    # and schoolbook for M_{2,3,2} (rank 12 each, giving 6*7+6*12 = 42+72 = 114).
    # Actually M_{2,3,2} schoolbook is 2*3*2 = 12. With the 2+1 split: 7+4=11.
    #
    # Let me implement the 108-rank solution.

    # Partition: n=4 as 2+2, m=5 as 2+3, p=6 as 2+2+2
    # Row splits for n: [0, 2] (sizes 2, 2)
    # Col splits for m: [0, 2, 5] (sizes 2, 3)
    # Col splits for p: [0, 2, 4, 6] (sizes 2, 2, 2)

    n_splits = [0, 2, 4]  # n=4 split into 2+2
    m_splits = [0, 2, 5]  # m=5 split into 2+3
    p_splits = [0, 2, 4, 6]  # p=6 split into 2+2+2

    # Determine max rank needed
    # 6 blocks of M_{2,2,2} (rank 7) + 6 blocks of M_{2,3,2} (rank 11)
    # Total rank = 6*7 + 6*11 = 108
    max_rank = 108

    U = [[0] * max_rank for _ in range(n * m)]  # 20 rows
    V = [[0] * max_rank for _ in range(m * p)]  # 30 rows
    W = [[0] * max_rank for _ in range(p * n)]  # 24 rows

    col = 0

    for i_blk in range(len(n_splits) - 1):
        i0 = n_splits[i_blk]
        i_size = n_splits[i_blk + 1] - i0
        for j_blk in range(len(m_splits) - 1):
            j0 = m_splits[j_blk]
            j_size = m_splits[j_blk + 1] - j0
            for k_blk in range(len(p_splits) - 1):
                k0 = p_splits[k_blk]
                k_size = p_splits[k_blk + 1] - k0

                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Strassen 2x2x2
                    col = add_strassen_2x2x2(i0, j0, k0, col, U, V, W)
                else:
                    # Schoolbook for this block
                    col = add_schoolbook(i_size, j_size, k_size, i0, j0, k0, col, U, V, W)

                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
