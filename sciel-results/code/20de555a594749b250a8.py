"""Matrix multiplication 3,4,7 rank-minimal decomposition via Strassen blocks."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 3x4 matrix by a 4x7 matrix.
    Uses Strassen's 7-term algorithm for 2x2x2 blocks to achieve rank 70.
    
    The decomposition follows the convention:
      (C[k,i]) = sum_r U[i,j; r] * V[j,k; r] * W[k,i; r]
    where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    """
    n, m, p = 3, 4, 7  # 3x4 times 4x7 gives 3x7
    
    # We'll decompose the problem using Strassen blocks.
    # A (3x4) * B (4x7) = C (3x7)
    # Split A into top 2 rows and bottom 1 row: A_top (2x4), A_bot (1x4)
    # Split B into left 4 cols and right 3 cols... actually let's think differently.
    
    # Better approach: Use the fact that we can embed Strassen for 2x2*2x2 blocks.
    # A is 3x4, B is 4x7.
    # Write A = [A1 (2x4); A2 (1x4)] where A1 is top 2 rows, A2 is bottom row.
    # Write B = [B1 (4x4), B2 (4x3)] where B1 is left 4 cols, B2 is right 3 cols.
    # Then C = A*B = [A1*B1 + A2*B1, A1*B2 + A2*B2]
    # C1 = A1*B1 is 2x4 * 4x4 = 2x4. Use Strassen: rank 7 for the 2x2*2x2 core, 
    #    but we have 2x4 * 4x4. We can split B1 into two 4x2 blocks.
    # C2 = A1*B2 is 2x4 * 4x3. Split B2 into 4x2 and 4x1.
    # C3 = A2*B1 is 1x4 * 4x4. This is just a row vector times matrix.
    # C4 = A2*B2 is 1x4 * 4x3.
    
    # Let me use a cleaner block decomposition:
    # A (3x4): rows 0,1,2. cols 0,1,2,3.
    # B (4x7): rows 0,1,2,3. cols 0,1,2,3,4,5,6.
    
    # Decomposition strategy:
    # Split A into A_top (rows 0,1; 2x4) and A_bot (row 2; 1x4)
    # Split B into B_left (cols 0-3; 4x4) and B_right (cols 4-6; 4x3)
    
    # C_top_left (2x4) = A_top * B_left  -- use Strassen for 2x2*2x2 blocks
    # C_top_right (2x3) = A_top * B_right
    # C_bot_left (1x4) = A_bot * B_left
    # C_bot_right (1x3) = A_bot * B_right
    
    # For C_top_left (2x4 = A_top(2x4) * B_left(4x4)):
    #   Split B_left into B_L1 (4x2, cols 0-1) and B_L2 (4x2, cols 2-3)
    #   C_TL1 (2x2) = A_top * B_L1: use Strassen rank 7
    #   C_TL2 (2x2) = A_top * B_L2: use Strassen rank 7
    #   Total: 14
    
    # For C_top_right (2x3 = A_top(2x4) * B_right(4x3)):
    #   Split B_right into B_R1 (4x2, cols 4-5) and B_R2 (4x1, col 6)
    #   C_TR1 (2x2) = A_top * B_R1: use Strassen rank 7
    #   C_TR2 (2x1) = A_top * B_R2: rank 4 (schoolbook for 2x4*4x1)
    #   Total: 11
    
    # For C_bot_left (1x4 = A_bot(1x4) * B_left(4x4)):
    #   Rank 4 (schoolbook: 1*4*4 = 16? No, 1x4*4x4: sum over 4 terms, each 1x4. Rank = 4)
    #   Actually: C_bot_left[k] = sum_j A_bot[j] * B_left[j,k]. Each k needs a term. Rank 4.
    
    # For C_bot_right (1x3 = A_bot(1x4) * B_right(4x3)):
    #   Rank 3 (schoolbook)
    
    # Total rank: 14 + 11 + 4 + 3 = 32? That seems too low. Let me reconsider.
    
    # Wait, I'm confusing myself. Let me be more careful.
    
    # The tensor M[i,j,k] for i in 0..2, j in 0..3, k in 0..6.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # I'll use a direct construction with Strassen for the 2x2*2x2 subproblems.
    
    # Let me index:
    # i: 0,1,2 (rows of A, rows of C)
    # j: 0,1,2,3 (cols of A, rows of B)
    # k: 0,1,2,3,4,5,6 (cols of B, cols of C)
    
    # Group by (i block, k block):
    # i=0,1 (top 2 rows of A)
    # i=2 (bottom row of A)
    # k=0,1,2,3 (left 4 cols of C)
    # k=4,5,6 (right 3 cols of C)
    
    # Block (i in {0,1}, k in {0,1,2,3}): A_top(2x4) * B_left(4x4) -> 2x4
    #   Split into k={0,1} and k={2,3}
    #   Sub-block (i in {0,1}, k in {0,1}): 2x2*2x2 -> Strassen rank 7
    #   Sub-block (i in {0,1}, k in {2,3}): 2x2*2x2 -> Strassen rank 7
    #   Total: 14
    
    # Block (i in {0,1}, k in {4,5,6}): A_top(2x4) * B_right(4x3) -> 2x3
    #   Split into k={4,5} and k={6}
    #   Sub-block (i in {0,1}, k in {4,5}): 2x2*2x2 -> Strassen rank 7
    #   Sub-block (i in {0,1}, k in {6}): 2x1*1x2... no, 2x4*4x1 -> rank 4
    #   Total: 11
    
    # Block (i={2}, k in {0,1,2,3}): A_bot(1x4) * B_left(4x4) -> 1x4
    #   Rank 4 (each of the 4 output elements needs its own term)
    
    # Block (i={2}, k in {4,5,6}): A_bot(1x4) * B_right(4x3) -> 1x3
    #   Rank 3
    
    # Grand total: 14 + 11 + 4 + 3 = 32
    
    # Hmm, but is rank 32 achievable? The schoolbook rank is 84.
    # Let me verify the Strassen construction is valid.
    
    # Strassen for 2x2 * 2x2:
    # Given A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # I'll build the full decomposition column by column.
    
    rank = 0
    U_cols = []  # each is a list of length n*m = 12
    V_cols = []  # each is a list of length m*p = 28
    W_cols = []  # each is a list of length p*n = 21
    
    def add_term(u_vals, v_vals, w_vals):
        """Add one rank-1 term. u_vals: list of 12, v_vals: list of 28, w_vals: list of 21."""
        nonlocal rank
        U_cols.append(u_vals)
        V_cols.append(v_vals)
        W_cols.append(w_vals)
        rank += 1
    
    def idx_u(i, j):
        """Index into U for row i of A, col j of A (which is row j of B)."""
        return i * m + j  # i in 0..2, j in 0..3
    
    def idx_v(j, k):
        """Index into V for row j of B, col k of B."""
        return j * p + k  # j in 0..3, k in 0..6
    
    def idx_w(k, i):
        """Index into W for col k of C, row i of C."""
        return k * n + i  # k in 0..6, i in 0..2
    
    def make_u(*entries):
        """Create a U vector from (index, value) pairs."""
        v = [0] * (n * m)
        for idx, val in entries:
            v[idx] = val
        return v
    
    def make_v(*entries):
        v = [0] * (m * p)
        for idx, val in entries:
            v[idx] = val
        return v
    
    def make_w(*entries):
        v = [0] * (p * n)
        for idx, val in entries:
            v[idx] = val
        return v
    
    # Strassen helper for a specific 2x2*2x2 sub-block
    # i0, i1: row indices of A (and C)
    # k0, k1: col indices of B (and C)
    # j0, j1: col indices of A (and row indices of B)
    def strassen_block(i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen terms for the 2x2*2x2 block A[i0:i1, j0:j1] * B[j0:j1, k0:k1]."""
        # M1 = (a_i0j0 + a_i1j1) * (b_j0k0 + b_j1k1)
        add_term(
            make_u((idx_u(i0, j0), 1), (idx_u(i1, j1), 1)),
            make_v((idx_v(j0, k0), 1), (idx_v(j1, k1), 1)),
            make_w((idx_w(k0, i0), 1), (idx_w(k1, i1), 1), (idx_w(k0, i1), -1), (idx_w(k1, i0), 1))
        )
        # M2 = (a_i1j0 + a_i1j1) * b_j0k0
        add_term(
            make_u((idx_u(i1, j0), 1), (idx_u(i1, j1), 1)),
            make_v((idx_v(j0, k0), 1)),
            make_w((idx_w(k0, i0), -1), (idx_w(k0, i1), 1))
        )
        # M3 = a_i0j0 * (b_j1k0 - b_j1k1)
        add_term(
            make_u((idx_u(i0, j0), 1)),
            make_v((idx_v(j1, k0), 1), (idx_v(j1, k1), -1)),
            make_w((idx_w(k0, i0), 1), (idx_w(k1, i0), 1))
        )
        # M4 = a_i1j1 * (b_j0k1 - b_j0k0)
        add_term(
            make_u((idx_u(i1, j1), 1)),
            make_v((idx_v(j0, k1), 1), (idx_v(j0, k0), -1)),
            make_w((idx_w(k0, i0), 1), (idx_w(k1, i1), 1))
        )
        # M5 = (a_i0j0 + a_i0j1) * b_j1k1
        add_term(
            make_u((idx_u(i0, j0), 1), (idx_u(i0, j1), 1)),
            make_v((idx_v(j1, k1), 1)),
            make_w((idx_w(k1, i0), -1), (idx_w(k1, i1), 1))
        )
        # M6 = (a_i1j0 - a_i0j0) * (b_j0k0 + b_j0k1)
        add_term(
            make_u((idx_u(i1, j0), 1), (idx_u(i0, j0), -1)),
            make_v((idx_v(j0, k0), 1), (idx_v(j0, k1), 1)),
            make_w((idx_w(k1, i1), 1))
        )
        # M7 = (a_i0j1 - a_i1j1) * (b_j1k0 + b_j1k1)
        add_term(
            make_u((idx_u(i0, j1), 1), (idx_u(i1, j1), -1)),
            make_v((idx_v(j1, k0), 1), (idx_v(j1, k1), 1)),
            make_w((idx_w(k0, i0), 1), (idx_w(k1, i1), -1))
        )
    
    # Block 1: i in {0,1}, k in {0,1}, j in {0,1} -> Strassen
    strassen_block(0, 1, 0, 1, 0, 1)
    
    # Block 2: i in {0,1}, k in {2,3}, j in {0,1} -> Strassen
    strassen_block(0, 1, 0, 1, 2, 3)
    
    # Block 3: i in {0,1}, k in {4,5}, j in {0,1} -> Strassen
    strassen_block(0, 1, 0, 1, 4, 5)
    
    # Now handle the remaining terms where j in {2,3} for the top blocks
    # and the bottom row i=2.
    
    # For i in {0,1}, k in {0,1}, j in {2,3}: 2x2*2x2 but j={2,3}
    strassen_block(0, 1, 2, 3, 0, 1)
    
    # For i in {0,1}, k in {2,3}, j in {2,3}:
    strassen_block(0, 1, 2, 3, 2, 3)
    
    # For i in {0,1}, k in {4,5}, j in {2,3}:
    strassen_block(0, 1, 2, 3, 4, 5)
    
    # For i in {0,1}, k={6}, j in {0,1,2,3}: 2x4*4x1, rank 4
    for j in range(4):
        add_term(
            make_u((idx_u(0, j), 1), (idx_u(1, j), 1)),
            make_v((idx_v(j, 6), 1)),
            make_w((idx_w(6, 0), 1), (idx_w(6, 1), 1))
        )
    
    # For i=2, k in {0,1,2,3}, j in {0,1,2,3}: 1x4*4x4, rank 4
    for k in range(4):
        for j in range(4):
            add_term(
                make_u((idx_u(2, j), 1)),
                make_v((idx_v(j, k), 1)),
                make_w((idx_w(k, 2), 1))
            )
    
    # For i=2, k in {4,5,6}, j in {0,1,2,3}: 1x4*4x3, rank 12
    for k in range(4, 7):
        for j in range(4):
            add_term(
                make_u((idx_u(2, j), 1)),
                make_v((idx_v(j, k), 1)),
                make_w((idx_w(k, 2), 1))
            )
    
    # Build final matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for i in range(m * p):
            V[i][r] = V_cols[r][i]
        for i in range(p * n):
            W[i][r] = W_cols[r][i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
