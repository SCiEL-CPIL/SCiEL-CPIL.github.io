"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    """Return (U, V, W) for 2x2x2 Strassen multiplication."""
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U rows: (i,j) for i,j in {0,1} -> indices 0,1,2,3
    # V rows: (j,k) for j,k in {0,1} -> indices 0,1,2,3
    # W rows: (k,i) for k,i in {0,1} -> indices 0,1,2,3
    #
    # For each product Mc, U[:,c] gives coefficients for A entries,
    # V[:,c] gives coefficients for B entries,
    # W[:,c] gives coefficients for C entries.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    # A00 -> U[0,0]=1, A11 -> U[3,0]=1
    # B00 -> V[0,0]=1, B11 -> V[3,0]=1
    # C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00 (k=0,i=0)
    W[3][0] = 1  # C11 (k=1,i=1)
    
    # M2 = (A10 + A11)B00
    # A10 -> U[2,1]=1, A11 -> U[3,1]=1
    # B00 -> V[0,1]=1
    # C10 += M2, C11 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10 (k=1,i=0)
    W[3][1] = -1 # C11 (k=1,i=1)
    
    # M3 = A00(B01 - B11)
    # A00 -> U[0,2]=1
    # B01 -> V[1,2]=1, B11 -> V[3,2]=-1
    # C01 += M3, C11 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01 (k=0,i=1)
    W[3][2] = 1  # C11 (k=1,i=1)
    
    # M4 = A11(B10 - B00)
    # A11 -> U[3,3]=1
    # B10 -> V[2,3]=1, B00 -> V[0,3]=-1
    # C00 += M4, C10 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5 = (A00 + A01)B11
    # A00 -> U[0,4]=1, A01 -> U[1,4]=1
    # B11 -> V[3,4]=1
    # C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    # A10 -> U[2,5]=1, A00 -> U[0,5]=-1
    # B00 -> V[0,5]=1, B01 -> V[1,5]=1
    # C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    # A01 -> U[1,6]=1, A11 -> U[3,6]=-1
    # B10 -> V[2,6]=1, B11 -> V[3,6]=1
    # C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return U, V, W

def _combine_factors(global_U, global_V, global_W, 
                     local_U, local_V, local_W,
                     a_i0, a_j0, b_j0, b_k0, c_k0, c_i0,
                     a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
    """Place local factors into global factors.
    
    local_U: (a_rows*a_cols, rank_local) - rows indexed by (i_local, j_local)
    local_V: (b_rows*b_cols, rank_local) - rows indexed by (j_local, k_local)
    local_W: (c_rows*c_cols, rank_local) - rows indexed by (k_local, i_local)
    
    Global U rows: (i, j) with i in [0,n), j in [0,m)
    Global V rows: (j, k) with j in [0,m), k in [0,p)
    Global W rows: (k, i) with k in [0,p), i in [0,n)
    """
    rank_local = len(local_U[0]) if local_U else 0
    for c in range(rank_local):
        u_col = global_U_col  # placeholder
        # Actually, we need to track column offset
        pass

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 100  # safety margin
    
    # Block decomposition of 3x5x5:
    # A is 3x5, B is 5x5, C is 3x5
    # 
    # Partition A into rows: A0 (row 0, 1x5), A1 (rows 1-2, 2x5)
    # Partition B into cols: B0 (cols 0-1, 5x2), B1 (cols 2-4, 5x3)
    # Wait, let me think about this more carefully.
    #
    # A: 3 rows, 5 cols. Index (i, j) where i in {0,1,2}, j in {0,1,2,3,4}
    # B: 5 rows, 5 cols. Index (j, k) where j in {0,1,2,3,4}, k in {0,1,2,3,4}
    # C: 3 rows, 5 cols. Index (i, k) where i in {0,1,2}, k in {0,1,2,3,4}
    #
    # W is indexed by (k, i) -> row index = k*n + i = k*3 + i
    #
    # Let's partition:
    # A rows: {0} and {1,2} -> A_top (1x5), A_bot (2x5)
    # B cols: {0,1}, {2,3}, {4} -> B_left (5x2), B_mid (5x2), B_right (5x1)
    # 
    # Actually, let's use a simpler partition that allows Strassen on 2x2 blocks.
    #
    # Partition A into row blocks: R0 = {0} (1 row), R1 = {1,2} (2 rows)
    # Partition B into col blocks: C0 = {0,1} (2 cols), C1 = {2,3} (2 cols), C2 = {4} (1 col)
    #
    # Terms in C = A*B:
    # C[R0, C0] = A[R0,:] * B[:,C0]  -> 1x5 * 5x2 = 1x2
    # C[R0, C1] = A[R0,:] * B[:,C1]  -> 1x5 * 5x2 = 1x2
    # C[R0, C2] = A[R0,:] * B[:,C2]  -> 1x5 * 5x1 = 1x1
    # C[R1, C0] = A[R1,:] * B[:,C0]  -> 2x5 * 5x2 = 2x2
    # C[R1, C1] = A[R1,:] * B[:,C1]  -> 2x5 * 5x2 = 2x2
    # C[R1, C2] = A[R1,:] * B[:,C2]  -> 2x5 * 5x1 = 2x1
    #
    # For 2x2 terms, we can use Strassen if we further partition the 5 cols of A and 5 rows of B.
    #
    # For A[R1,:] (2x5) * B[:,C0] (5x2):
    #   Partition the 5 cols of A[R1,:] into: {0,1}, {2,3}, {4}
    #   Partition the 5 rows of B[:,C0] into: {0,1}, {2,3}, {4}
    #   Sub-products:
    #     A[1,0:2] * B[0:2,0:2] -> 2x2 * 2x2 -> Strassen, rank 7
    #     A[1,2:4] * B[2:4,0:2] -> 2x2 * 2x2 -> Strassen, rank 7
    #     A[1,4:5] * B[4:5,0:2] -> 2x1 * 1x2 -> schoolbook, rank 4
    #     A[1,0:2] * B[2:4,0:2] -> 2x2 * 2x2 -> schoolbook, rank 16? No wait...
    #
    # Actually, I need to be more careful. The 5 columns of A and 5 rows of B are partitioned the same way.
    # Let me use partition: {0,1}, {2,3}, {4} for both.
    #
    # For A[R1,:] * B[:,C0]:
    #   A[R1, {0,1}] * B[{0,1}, C0] -> 2x2 * 2x2, Strassen rank 7
    #   A[R1, {0,1}] * B[{2,3}, C0] -> 2x2 * 2x2, schoolbook rank 16
    #   A[R1, {0,1}] * B[{4}, C0]   -> 2x1 * 1x2, schoolbook rank 4
    #   A[R1, {2,3}] * B[{0,1}, C0] -> 2x2 * 2x2, schoolbook rank 16
    #   A[R1, {2,3}] * B[{2,3}, C0] -> 2x2 * 2x2, Strassen rank 7
    #   A[R1, {2,3}] * B[{4}, C0]   -> 2x1 * 1x2, schoolbook rank 4
    #   A[R1, {4}]   * B[{0,1}, C0] -> 1x2 * 2x2, schoolbook rank 8
    #   A[R1, {4}]   * B[{2,3}, C0] -> 1x2 * 2x2, schoolbook rank 8
    #   A[R1, {4}]   * B[{4}, C0]   -> 1x1 * 1x2, schoolbook rank 2
    #
    # Total for this term: 7 + 16 + 4 + 16 + 7 + 4 + 8 + 8 + 2 = 72
    # That's worse than schoolbook 2*5*2 = 20.
    #
    # Hmm, this approach isn't working well. Let me reconsider.
    #
    # Actually, for the 2x2 * 2x2 sub-problems that are diagonal (same block indices),
    # I can use Strassen. For off-diagonal, schoolbook is fine.
    #
    # But wait, the total schoolbook rank for 3x5x5 is 75. My current best is 75.
    # I need to find a way to reduce below 75.
    #
    # Let me try a different approach. Use Strassen on the entire 3x5x5 by embedding
    # into a larger problem? No, that would increase rank.
    #
    # Alternative: Use the fact that 3 = 2+1 and 5 = 2+2+1.
    #
    # Let me try:
    # A: rows {0}, {1,2}
    # B: cols {0,1}, {2,3}, {4}
    #
    # C[0, {0,1}] = A[0,:] * B[:, {0,1}]
    #   = A[0,{0,1}] * B[{0,1},{0,1}] + A[0,{2,3}] * B[{2,3},{0,1}] + A[0,{4}] * B[{4},{0,1}]
    #   = 1x2 * 2x2 (rank 4) + 1x2 * 2x2 (rank 4) + 1x1 * 1x2 (rank 2)
    #   Total: 10
    #
    # C[0, {2,3}] = A[0,:] * B[:, {2,3}]
    #   = A[0,{0,1}] * B[{0,1},{2,3}] + A[0,{2,3}] * B[{2,3},{2,3}] + A[0,{4}] * B[{4},{2,3}]
    #   = 1x2 * 2x2 (rank 4) + 1x2 * 2x2 (rank 4) + 1x1 * 1x2 (rank 2)
    #   Total: 10
    #
    # C[0, {4}] = A[0,:] * B[:, {4}]
    #   = A[0,{0,1}] * B[{0,1},{4}] + A[0,{2,3}] * B[{2,3},{4}] + A[0,{4}] * B[{4},{4}]
    #   = 1x2 * 2x1 (rank 4) + 1x2 * 2x1 (rank 4) + 1x1 * 1x1 (rank 1)
    #   Total: 9
    #
    # C[{1,2}, {0,1}] = A[{1,2},:] * B[:, {0,1}]
    #   = A[{1,2},{0,1}] * B[{0,1},{0,1}] + A[{1,2},{2,3}] * B[{2,3},{0,1}] + A[{1,2},{4}] * B[{4},{0,1}]
    #   = 2x2 * 2x2 (Strassen rank 7) + 2x2 * 2x2 (schoolbook rank 16) + 2x1 * 1x2 (schoolbook rank 4)
    #   Total: 27
    #
    # C[{1,2}, {2,3}] = A[{1,2},:] * B[:, {2,3}]
    #   = A[{1,2},{0,1}] * B[{0,1},{2,3}] + A[{1,2},{2,3}] * B[{2,3},{2,3}] + A[{1,2},{4}] * B[{4},{2,3}]
    #   = 2x2 * 2x2 (schoolbook rank 16) + 2x2 * 2x2 (Strassen rank 7) + 2x1 * 1x2 (schoolbook rank 4)
    #   Total: 27
    #
    # C[{1,2}, {4}] = A[{1,2},:] * B[:, {4}]
    #   = A[{1,2},{0,1}] * B[{0,1},{4}] + A[{1,2},{2,3}] * B[{2,3},{4}] + A[{1,2},{4}] * B[{4},{4}]
    #   = 2x2 * 2x1 (schoolbook rank 8) + 2x2 * 2x1 (schoolbook rank 8) + 2x1 * 1x1 (schoolbook rank 2)
    #   Total: 18
    #
    # Grand total: 10 + 10 + 9 + 27 + 27 + 18 = 101
    # That's worse than 75.
    #
    # The problem is that off-diagonal blocks use schoolbook which is inefficient.
    #
    # Let me try using Strassen for ALL 2x2 * 2x2 sub-problems, not just diagonal.
    #
    # C[{1,2}, {0,1}]:
    #   = A[{1,2},{0,1}] * B[{0,1},{0,1}] + A[{1,2},{2,3}] * B[{2,3},{0,1}] + A[{1,2},{4}] * B[{4},{0,1}]
    #   = Strassen(2x2*2x2) + Strassen(2x2*2x2) + schoolbook(2x1*1x2)
    #   = 7 + 7 + 4 = 18
    #
    # C[{1,2}, {2,3}]:
    #   = A[{1,2},{0,1}] * B[{0,1},{2,3}] + A[{1,2},{2,3}] * B[{2,3},{2,3}] + A[{1,2},{4}] * B[{4},{2,3}]
    #   = Strassen + Strassen + schoolbook
    #   = 7 + 7 + 4 = 18
    #
    # C[{1,2}, {4}]:
    #   = A[{1,2},{0,1}] * B[{0,1},{4}] + A[{1,2},{2,3}] * B[{2,3},{4}] + A[{1,2},{4}] * B[{4},{4}]
    #   = schoolbook(2x2*2x1) + schoolbook(2x2*2x1) + schoolbook(2x1*1x1)
    #   = 8 + 8 + 2 = 18
    #
    # Grand total: 10 + 10 + 9 + 18 + 18 + 18 = 83
    # Still worse than 75.
    #
    # I think I need a fundamentally different approach. Let me just implement the
    # previous proposal more carefully. The previous proposal had a total of 71.
    #
    # Let me re-examine the previous proposal's partition.
    #
    # Actually, I realize I should just implement a correct 71-rank decomposition.
    # Let me use the partition from the previous proposal.
    #
    # A: 3x5, partition rows into {0}, {1,2}
    # B: 5x5, partition cols into {0,1}, {2,3}, {4}
    #
    # But the previous proposal had a different partition. Let me re-read it.
    #
    # The previous proposal had:
    # Term 1: A_{11} B_{11} (1x2 * 2x2, Rank 7? No, 1x2*2x2 is rank 4 schoolbook)
    #
    # I'm getting confused. Let me just implement a clean 75-rank schoolbook and 
    # then try to improve from there.
    
    # Start with schoolbook rank 75
    U, V, W = _schoolbook_factors(n, m, p)
    
    # Try to reduce rank using Strassen on 2x2 blocks
    # For this, we need to identify 2x2 sub-problems in the schoolbook decomposition
    # and replace them with Strassen.
    #
    # In schoolbook, each term (i,j,k) has rank 1.
    # We can group terms where i in {i1,i2}, j in {j1,j2}, k in {k1,k2} into a 2x2x2 block.
    # If we have all 8 such terms, we can replace them with Strassen rank 7.
    #
    # For 3x5x5, we can partition:
    # i: {0}, {1,2}
    # j: {0,1}, {2,3}, {4}
    # k: {0,1}, {2,3}, {4}
    #
    # The only 2x2x2 blocks are where i={1,2}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    # So we have 2x2 = 4 such blocks, each with 8 terms.
    # Replacing each with Strassen saves 1 rank per block.
    # Total savings: 4
    # New rank: 75 - 4 = 71
    
    # Let's implement this by building the decomposition from scratch
    # with Strassen for the 2x2x2 blocks and schoolbook for the rest.
    
    U = [[0] * 71 for _ in range(n * m)]
    V = [[0] * 71 for _ in range(m * p)]
    W = [[0] * 71 for _ in range(p * n)]
    col = 0
    
    # Define the partitions
    i_blocks = [(0,), (1, 2)]  # i=0, i=1,2
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    # For each combination of blocks, determine if it's a 2x2x2 block
    for i_b in range(len(i_blocks)):
        for j_b in range(len(j_blocks)):
            for k_b in range(len(k_blocks)):
                i_set = i_blocks[i_b]
                j_set = j_blocks[j_b]
                k_set = k_blocks[k_b]
                
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    # Use Strassen
                    su, sv, sw = _strassen_222()
                    # Map local indices to global
                    # Local i: 0->i_set[0], 1->i_set[1]
                    # Local j: 0->j_set[0], 1->j_set[1]
                    # Local k: 0->k_set[0], 1->k_set[1]
                    for c in range(7):
                        for li in range(2):
                            gi = i_set[li]
                            for lj in range(2):
                                gj = j_set[lj]
                                U[gi * m + gj][col + c] = su[li * 2 + lj][c]
                        for lj in range(2):
                            gj = j_set[lj]
                            for lk in range(2):
                                gk = k_set[lk]
                                V[gj * p + gk][col + c] = sv[lj * 2 + lk][c]
                        for lk in range(2):
                            gk = k_set[lk]
                            for li in range(2):
                                gi = i_set[li]
                                W[gk * n + gi][col + c] = sw[lk * 2 + li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_set:
                        for gj in j_set:
                            for gk in k_set:
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    # col should be 71
    assert col == 71, f"Expected rank 71, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
