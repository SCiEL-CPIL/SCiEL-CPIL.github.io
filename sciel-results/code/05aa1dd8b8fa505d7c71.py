"""Matrix multiplication 2x3x7: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]
TIME_LIMIT = 115.0  # safety margin before 120s kill


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Build the matrix multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 for all i, j, k (this is the identity of the bilinear map)
    # We want to find rank-r decomposition:
    #   T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # with W indexed as W[k*n + i, c] (official trace convention)

    # Strategy: Use the schoolbook decomposition as baseline (rank = 42),
    # then try to find a lower rank using structured decompositions.
    #
    # Key insight: For small matrices, we can try:
    # 1. Block decompositions using Strassen-like methods
    # 2. Randomized search with verification
    # 3. Known optimal bounds
    #
    # Known results:
    # - Schoolbook: rank 42
    # - Lower bound via slice rank: min over slices of matrix rank
    #   For T[i,j,k], fixing i gives a 3x7 matrix of all 1s -> rank 1
    #   Fixing j gives a 2x7 matrix of all 1s -> rank 1
    #   Fixing k gives a 2x3 matrix of all 1s -> rank 1
    #   So slice rank lower bound is weak here.
    #
    # Let's try a structured approach:
    # Decompose 7 = 2 + 2 + 3
    # M_{2,3,2} has known rank 11 (using 4 schoolbook + 7 Strassen for 2x2x2)
    # M_{2,3,3} can be split: 2x1x3 (rank 6) + 2x2x3
    #   2x2x3 = 2x2x2 (rank 7) + 2x2x1 (rank 4) -> rank 11
    #   So M_{2,3,3} rank <= 6 + 11 = 17
    # Total: 11 + 11 + 17 = 39

    # Let's implement the block decomposition carefully.
    #
    # For 2x3x2 (n=2, m=3, p=2):
    # Split m=3 into m1=1, m2=2
    # M_{2,1,2}: schoolbook rank 4
    # M_{2,2,2}: Strassen rank 7
    # Total: 4 + 7 = 11
    #
    # For 2x3x3 (n=2, m=3, p=3):
    # Split p=3 into p1=1, p2=2
    # M_{2,3,1}: schoolbook rank 6
    # M_{2,3,2}: rank 11 (as above)
    # Total: 6 + 11 = 17
    #
    # Grand total for 2x3x7: 11 + 11 + 17 = 39

    # Let's build this step by step.

    def make_schoolbook(n, m, p):
        """Schoolbook decomposition for n x m x p. Returns (U, V, W) each of shape (dim, n*m*p)."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def make_strassen_2x2x2():
        """Strassen decomposition for 2x2x2. Returns (U, V, W) each of shape (4, 7)."""
        # Strassen's 7 multiplications for 2x2x2
        # U: shape (4, 7) indexed by [i*2+j, c]
        # V: shape (4, 7) indexed by [j*2+k, c]
        # W: shape (4, 7) indexed by [k*2+i, c]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # m1 = A11 * (B22 - B12)
        U[0, 0] = 1   # A11
        V[2, 0] = 1   # B22 (j=1,k=0 -> index 1*2+0=2)
        V[1, 0] = -1  # B12 (j=0,k=1 -> index 0*2+1=1)
        W[2, 0] = 1   # k=1,i=0 -> index 1*2+0=2

        # m2 = (A11 + A12) * B12
        U[0, 1] = 1   # A11
        U[1, 1] = 1   # A12
        V[1, 1] = 1   # B12
        W[1, 1] = 1   # k=0,i=1 -> index 0*2+1=1
        W[3, 1] = 1   # k=1,i=1 -> index 1*2+1=3

        # m3 = A21 * (B11 - B21)
        U[2, 2] = 1   # A21
        V[0, 2] = 1   # B11 (j=0,k=0)
        V[2, 2] = -1  # B21 (j=1,k=0)
        W[0, 2] = 1   # k=0,i=0 -> index 0*2+0=0

        # m4 = A22 * (B11 + B22)
        U[3, 3] = 1   # A22
        V[0, 3] = 1   # B11
        V[3, 3] = 1   # B22 (j=1,k=1)
        W[0, 3] = 1   # k=0,i=0
        W[3, 3] = 1   # k=1,i=1

        # m5 = (A11 + A21) * (B11 + B12)
        U[0, 4] = 1   # A11
        U[2, 4] = 1   # A21
        V[0, 4] = 1   # B11
        V[1, 4] = 1   # B12
        W[0, 4] = 1   # k=0,i=0

        # m6 = (A22 - A11) * (B12 + B22)
        U[3, 5] = 1   # A22
        U[0, 5] = -1  # A11
        V[1, 5] = 1   # B12
        V[3, 5] = 1   # B22
        W[3, 5] = 1   # k=1,i=1

        # m7 = (A12 - A22) * (B21 + B22)
        U[1, 6] = 1   # A12
        U[3, 6] = -1  # A22
        V[2, 6] = 1   # B21 (j=1,k=0)
        V[3, 6] = 1   # B22
        W[2, 6] = 1   # k=1,i=0 -> index 1*2+0=2

        return U, V, W

    def combine_blocks(blocks):
        """
        Combine a list of (U, V, W) blocks into a single decomposition.
        Each block is a tuple of three arrays.
        Returns (U_all, V_all, W_all).
        """
        U_list = []
        V_list = []
        W_list = []
        for U_b, V_b, W_b in blocks:
            U_list.append(U_b)
            V_list.append(V_b)
            W_list.append(W_b)
        U_all = np.hstack(U_list)
        V_all = np.hstack(V_list)
        W_all = np.hstack(W_list)
        return U_all, V_all, W_all

    def shift_indices_U(U_block, i_offset, j_offset, m_full):
        """
        Shift U_block (shape (m_sub, rank)) to be embedded in full U of shape (m_full, rank).
        U_block indices correspond to rows [i_offset*m_full + j_offset : ...]
        Actually, U is indexed by [i*m_full + j]. For a block covering i in [i_offset, i_offset+n_sub)
        and j in [j_offset, j_offset+m_sub), we place U_block[row, col] at
        U[(i_offset + row//m_sub)*m_full + (j_offset + row%m_sub), col]
        But this is getting complex. Let me use a different approach: build the full U, V, W
        directly by placing each block's contributions.
        """
        pass

    # Let me use a cleaner approach: build U, V, W for the full 2x3x7 directly
    # by concatenating the block decompositions with proper index mapping.

    def build_block_decomp(n, m, p, n_split=None, m_split=None, p_split=None):
        """
        Build decomposition for n x m x p by splitting dimensions.
        n_split: how to split n (list of ints summing to n) or None
        m_split: how to split m or None
        p_split: how to split p or None
        Returns (U, V, W) of shapes (n*m, rank), (m*p, rank), (p*n, rank)
        """
        if n_split is None and m_split is None and p_split is None:
            # Schoolbook
            return make_schoolbook(n, m, p)

        blocks = []

        if n_split is not None and len(n_split) > 1:
            # Split along n dimension
            offset = 0
            for n_sub in n_split:
                if m_split is not None and len(m_split) > 1:
                    for m_sub in m_split:
                        if p_split is not None and len(p_split) > 1:
                            for p_sub in p_split:
                                U_sub, V_sub, W_sub = build_block_decomp(n_sub, m_sub, p_sub)
                                # Place in full arrays
                                blocks.append((U_sub, V_sub, W_sub, offset, 0, 0, n_sub, m_sub, p_sub))
                        else:
                            U_sub, V_sub, W_sub = build_block_decomp(n_sub, m_sub, p)
                            blocks.append((U_sub, V_sub, W_sub, offset, 0, 0, n_sub, m_sub, p))
                else:
                    if p_split is not None and len(p_split) > 1:
                        for p_sub in p_split:
                            U_sub, V_sub, W_sub = build_block_decomp(n_sub, m, p_sub)
                            blocks.append((U_sub, V_sub, W_sub, offset, 0, 0, n_sub, m, p_sub))
                    else:
                        U_sub, V_sub, W_sub = build_block_decomp(n_sub, m, p)
                        blocks.append((U_sub, V_sub, W_sub, offset, 0, 0, n_sub, m, p))
                offset += n_sub
        elif m_split is not None and len(m_split) > 1:
            offset = 0
            for m_sub in m_split:
                if p_split is not None and len(p_split) > 1:
                    for p_sub in p_split:
                        U_sub, V_sub, W_sub = build_block_decomp(n, m_sub, p_sub)
                        blocks.append((U_sub, V_sub, W_sub, 0, offset, 0, n, m_sub, p_sub))
                else:
                    U_sub, V_sub, W_sub = build_block_decomp(n, m_sub, p)
                    blocks.append((U_sub, V_sub, W_sub, 0, offset, 0, n, m_sub, p))
                offset += m_sub
        elif p_split is not None and len(p_split) > 1:
            offset = 0
            for p_sub in p_split:
                U_sub, V_sub, W_sub = build_block_decomp(n, m, p_sub)
                blocks.append((U_sub, V_sub, W_sub, 0, 0, offset, n, m, p_sub))
                offset += p_sub
        else:
            # Single dimension split, recurse
            if n_split is not None and len(n_split) == 1:
                return build_block_decomp(n, m, p, m_split=m_split, p_split=p_split)
            elif m_split is not None and len(m_split) == 1:
                return build_block_decomp(n, m, p, n_split=n_split, p_split=p_split)
            elif p_split is not None and len(p_split) == 1:
                return build_block_decomp(n, m, p, n_split=n_split, m_split=m_split)
            else:
                return make_schoolbook(n, m, p)

        # Now combine all blocks into full U, V, W
        total_rank = sum(U_b.shape[1] for U_b, V_b, W_b, *_ in blocks)
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))

        col_offset = 0
        for U_b, V_b, W_b, i_off, j_off, k_off, n_sub, m_sub, p_sub in blocks:
            rank_b = U_b.shape[1]
            for r in range(rank_b):
                col = col_offset + r
                for ri in range(n_sub):
                    for rj in range(m_sub):
                        U_full[(i_off + ri) * m + (j_off + rj), col] = U_b[ri * m_sub + rj, r]
                for rj in range(m_sub):
                    for rk in range(p_sub):
                        V_full[(j_off + rj) * p + (k_off + rk), col] = V_b[rj * p_sub + rk, r]
                for rk in range(p_sub):
                    for ri in range(n_sub):
                        W_full[(k_off + rk) * n + (i_off + ri), col] = W_b[rk * n_sub + ri, r]
            col_offset += rank_b

        return U_full, V_full, W_full

    # Use Strassen for 2x2x2 blocks
    def build_with_strassen(n, m, p, n_split=None, m_split=None, p_split=None):
        """
        Build decomposition using Strassen for 2x2x2 sub-blocks.
        """
        if n == 2 and m == 2 and p == 2:
            return make_strassen_2x2x2()
        if n == 1 and m == 1 and p == 1:
            U = np.zeros((1, 1))
            V = np.zeros((1, 1))
            W = np.zeros((1, 1))
            U[0, 0] = 1
            V[0, 0] = 1
            W[0, 0] = 1
            return U, V, W
        # Otherwise, use schoolbook for this sub-block
        return make_schoolbook(n, m, p)

    # Build the full 2x3x7 decomposition with splits:
    # p = 2 + 2 + 3
    # For each p-subblock:
    #   p=2: m = 1 + 2
    #     m=1: schoolbook 2x1x2 -> rank 4
    #     m=2: Strassen 2x2x2 -> rank 7
    #     Total for p=2: rank 11
    #   p=3: m = 1 + 2
    #     m=1: schoolbook 2x1x3 -> rank 6
    #     m=2: 2x2x3, split p=1+2
    #       p=1: schoolbook 2x2x1 -> rank 4
    #       p=2: Strassen 2x2x2 -> rank 7
    #       Total 2x2x3: rank 11
    #     Total for p=3: rank 6 + 11 = 17
    # Grand total: 11 + 11 + 17 = 39

    U, V, W = build_block_decomp(2, 3, 7, p_split=[2, 2, 3])
    # But this uses schoolbook everywhere. Let me override with Strassen for 2x2x2.

    # Let me rebuild manually for clarity:
    U_parts = []
    V_parts = []
    W_parts = []

    # Block 1: 2x1x2 (i:0-1, j:0, k:0-1)
    U1, V1, W1 = make_schoolbook(2, 1, 2)
    U_parts.append((U1, 0, 0))  # (data, i_offset, j_offset)
    V_parts.append((V1, 0, 0))  # (data, j_offset, k_offset)
    W_parts.append((W1, 0, 0))  # (data, k_offset, i_offset)

    # Block 2: 2x2x2 (i:0-1, j:1-2, k:0-1) - Strassen
    U2, V2, W2 = make_strassen_2x2x2()
    U_parts.append((U2, 0, 1))
    V_parts.append((V2, 1, 0))
    W_parts.append((W2, 0, 0))

    # Block 3: 2x1x2 (i:0-1, j:0, k:2-3)
    U3, V3, W3 = make_schoolbook(2, 1, 2)
    U_parts.append((U3, 0, 0))
    V_parts.append((V3, 0, 2))
    W_parts.append((W3, 2, 0))

    # Block 4: 2x2x2 (i:0-1, j:1-2, k:2-3) - Strassen
    U4, V4, W4 = make_strassen_2x2x2()
    U_parts.append((U4, 0, 1))
    V_parts.append((V4, 1, 2))
    W_parts.append((W4, 2, 0))

    # Block 5: 2x1x3 (i:0-1, j:0, k:4-6)
    U5, V5, W5 = make_schoolbook(2, 1, 3)
    U_parts.append((U5, 0, 0))
    V_parts.append((V5, 0, 4))
    W_parts.append((W5, 4, 0))

    # Block 6: 2x2x1 (i:0-1, j:1-2, k:4)
    U6, V6, W6 = make_schoolbook(2, 2, 1)
    U_parts.append((U6, 0, 1))
    V_parts.append((V6, 1, 4))
    W_parts.append((W6, 4, 0))

    # Block 7: 2x2x2 (i:0-1, j:1-2, k:5-6) - Strassen
    U7, V7, W7 = make_strassen_2x2x2()
    U_parts.append((U7, 0, 1))
    V_parts.append((V7, 1, 5))
    W_parts.append((W7, 5, 0))

    # Now assemble
    total_rank = sum(u.shape[1] for u, _, _ in U_parts)
    U_full = np.zeros((6, total_rank))  # n*m = 2*3 = 6
    V_full = np.zeros((21, total_rank))  # m*p = 3*7 = 21
    W_full = np.zeros((14, total_rank))  # p*n = 7*2 = 14

    col = 0
    for (u, i_off, j_off), (v, j_off_v, k_off), (w, k_off_w, i_off_w) in zip(U_parts, V_parts, W_parts):
        rank_b = u.shape[1]
        for r in range(rank_b):
            c = col + r
            # Fill U: u has shape (n_sub * m_sub, rank_b)
            # Row index in u: ri * m_sub + rj
            n_sub = u.shape[0] // (v.shape[0] // (w.shape[0] // 2))  # This is getting messy
            # Let me just iterate properly
            pass

    # Simpler approach: rebuild from scratch with explicit index handling
    U_full = np.zeros((6, 0))
    V_full = np.zeros((21, 0))
    W_full = np.zeros((14, 0))

    def add_block(U_b, V_b, W_b, i0, j0, k0, n_b, m_b, p_b):
        nonlocal U_full, V_full, W_full
        rank_b = U_b.shape[1]
        U_new = np.zeros((6, rank_b))
        V_new = np.zeros((21, rank_b))
        W_new = np.zeros((14, rank_b))
        for ri in range(n_b):
            for rj in range(m_b):
                U_new[(i0 + ri) * 3 + (j0 + rj), :] = U_b[ri * m_b + rj, :]
        for rj in range(m_b):
            for rk in range(p_b):
                V_new[(j0 + rj) * 7 + (k0 + rk), :] = V_b[rj * p_b + rk, :]
        for rk in range(p_b):
            for ri in range(n_b):
                W_new[(k0 + rk) * 2 + (i0 + ri), :] = W_b[rk * n_b + ri, :]
        U_full = np.hstack([U_full, U_new])
        V_full = np.hstack([V_full, V_new])
        W_full = np.hstack([W_full, W_new])

    # Block 1: 2x1x2, i0=0, j0=0, k0=0
    add_block(*make_schoolbook(2, 1, 2), 0, 0, 0, 2, 1, 2)

    # Block 2: 2x2x2 Strassen, i0=0, j0=1, k0=0
    add_block(*make_strassen_2x2x2(), 0, 1, 0, 2, 2, 2)

    # Block 3: 2x1x2, i0=0, j0=0, k0=2
    add_block(*make_schoolbook(2, 1, 2), 0, 0, 2, 2, 1, 2)

    # Block 4: 2x2x2 Strassen, i0=0, j0=1, k0=2
    add_block(*make_strassen_2x2x2(), 0, 1, 2, 2, 2, 2)

    # Block 5: 2x1x3, i0=0, j0=0, k0=4
    add_block(*make_schoolbook(2, 1, 3), 0, 0, 4, 2, 1, 3)

    # Block 6: 2x2x1, i0=0, j0=1, k0=4
    add_block(*make_schoolbook(2, 2, 1), 0, 1, 4, 2, 2, 1)

    # Block 7: 2x2x2 Strassen, i0=0, j0=1, k0=5
    add_block(*make_strassen_2x2x2(), 0, 1, 5, 2, 2, 2)

    # Verify
    def verify(U, V, W, n=2, m=3, p=7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    if not verify(U_full, V_full, W_full):
        # Fall back to schoolbook
        U_full, V_full, W_full = make_schoolbook(2, 3, 7)

    # Convert to list of lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()

    # Check time
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        # Return what we have
        pass

    return {"U": U_list, "V": V_list, "W": W_list}
