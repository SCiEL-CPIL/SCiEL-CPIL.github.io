"""Matrix multiplication 3x5x6: rank-minimal exact tensor decomposition.

Strategy: reproduce the known rank-84 recursive block-embedding solution.
The 2x2x2 Strassen block (rank 7) is embedded into the 3x5x6 tensor, saving
one rank per embedded block relative to the schoolbook rank-90 baseline.

We verify the decomposition against a random ensemble before returning, and
fall back to the safe schoolbook rank-90 if verification fails.
"""
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill

def _build_schoolbook():
    """Trivial rank n*m*p decomposition (always valid)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_strassen_block():
    """
    Rank-7 decomposition of the 2x2x2 matrix multiplication tensor.

    Convention: for a 2x2 block, indices (i,j,k) with i in {0,1} (row of A),
    j in {0,1} (col of A / row of B), k in {0,1} (col of B).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].

    Strassen's 7 products:
      m0 = (A00+A11)(B00+B11)
      m1 = (A10+A11) B00
      m2 = A00 (B01-B11)
      m3 = A11 (B10-B00)
      m4 = (A00+A01) B11
      m5 = (A10-A00)(B00+B10)
      m6 = (A01-A11)(B01+B11)

    Output:
      C00 = m0 + m3 - m4 - m6
      C01 = m2 + m4
      C10 = m1 + m5
      C11 = m0 - m1 + m2 + m5
    """
    # 7 factors. U has 4 rows (i*2+j), V has 4 rows (j*2+k), W has 4 rows (k*2+i).
    # We express each Strassen product as a rank-1 term:
    #   m = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # For each m_t, we need to factor the linear forms.

    # U[i*2+j, c] : coefficient of A[i,j]
    # V[j*2+k, c] : coefficient of B[j,k]
    # W[k*2+i, c] : coefficient of C[k,i]  -> output index k*2+i

    U = [[0.0] * 7 for _ in range(4)]
    V = [[0.0] * 7 for _ in range(4)]
    W = [[0.0] * 7 for _ in range(4)]

    # m0 = (A00+A11)(B00+B11)
    # A part: A00 + A11 -> U[0*2+0,0]=1, U[1*2+1,0]=1
    # B part: B00 + B11 -> V[0*2+0,0]=1, V[1*2+1,0]=1
    # C part: contributes to C00 (+1), C11 (+1)
    #   C00 = W[0*2+0, 0] = W[0,0] = +1
    #   C11 = W[1*2+1, 0] = W[3,0] = +1
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0

    # m1 = (A10+A11) B00
    # A: A10 + A11 -> U[1*2+0,1]=1, U[1*2+1,1]=1
    # B: B00 -> V[0*2+0,1]=1
    # C: C10 (+1), C11 (-1)
    #   C10 = W[1*2+0, 1] = W[2,1] = +1
    #   C11 = W[1*2+1, 1] = W[3,1] = -1
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0; W[3][1] = -1.0

    # m2 = A00 (B01-B11)
    # A: A00 -> U[0*2+0,2]=1
    # B: B01 - B11 -> V[0*2+1,2]=1, V[1*2+1,2]=-1
    # C: C01 (+1), C11 (+1)
    #   C01 = W[0*2+1, 2] = W[1,2] = +1
    #   C11 = W[1*2+1, 2] = W[3,2] = +1
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0; W[3][2] = 1.0

    # m3 = A11 (B10-B00)
    # A: A11 -> U[1*2+1,3]=1
    # B: B10 - B00 -> V[1*2+0,3]=1, V[0*2+0,3]=-1
    # C: C00 (+1)
    #   C00 = W[0*2+0, 3] = W[0,3] = +1
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0

    # m4 = (A00+A01) B11
    # A: A00 + A01 -> U[0*2+0,4]=1, U[0*2+1,4]=1
    # B: B11 -> V[1*2+1,4]=1
    # C: C00 (-1), C01 (+1)
    #   C00 = W[0*2+0, 4] = W[0,4] = -1
    #   C01 = W[0*2+1, 4] = W[1,4] = +1
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[1][4] = 1.0

    # m5 = (A10-A00)(B00+B10)
    # A: A10 - A00 -> U[1*2+0,5]=1, U[0*2+0,5]=-1
    # B: B00 + B10 -> V[0*2+0,5]=1, V[1*2+0,5]=1
    # C: C10 (+1), C11 (+1)
    #   C10 = W[1*2+0, 5] = W[2,5] = +1
    #   C11 = W[1*2+1, 5] = W[3,5] = +1
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[2][5] = 1.0
    W[2][5] = 1.0; W[3][5] = 1.0

    # m6 = (A01-A11)(B01+B11)
    # A: A01 - A11 -> U[0*2+1,6]=1, U[1*2+1,6]=-1
    # B: B01 + B11 -> V[0*2+1,6]=1, V[1*2+1,6]=1
    # C: C00 (-1)
    #   C00 = W[0*2+0, 6] = W[0,6] = -1
    U[1][6] = 1.0; U[3][6] = -1.0
    V[1][6] = 1.0; V[3][6] = 1.0
    W[0][6] = -1.0

    return U, V, W


def _embed_2x2_block(U_full, V_full, W_full, block_U, block_V, block_W,
                     i_off, j_off, k_off, c_off):
    """Embed a 2x2x2 block at offsets (i_off, j_off, k_off) with column offset c_off.

    U_full has rows indexed by i*m + j (m=5). Block U row r = bi*2+bj maps to (i_off+bi, j_off+bj).
    V_full has rows indexed by j*p + k (p=6). Block V row r = bj*2+bk maps to (j_off+bj, k_off+bk).
    W_full has rows indexed by k*n + i (n=3). Block W row r = bk*2+bi maps to (k_off+bk, i_off+bi).
    """
    n, m, p = DIMENSIONS
    for bi in range(2):
        for bj in range(2):
            Urow = (i_off + bi) * m + (j_off + bj)
            for c in range(7):
                val = block_U[bi * 2 + bj][c]
                if val != 0:
                    U_full[Urow][c_off + c] += val
    for bj in range(2):
        for bk in range(2):
            Vrow = (j_off + bj) * p + (k_off + bk)
            for c in range(7):
                val = block_V[bj * 2 + bk][c]
                if val != 0:
                    V_full[Vrow][c_off + c] += val
    for bk in range(2):
        for bi in range(2):
            Wrow = (k_off + bk) * n + (i_off + bi)
            for c in range(7):
                val = block_W[bk * 2 + bi][c]
                if val != 0:
                    W_full[Wrow][c_off + c] += val


def _build_rank84():
    """
    Construct the rank-84 decomposition by embedding 2x2x2 Strassen blocks
    into the 3x5x6 tensor.

    The 3x5x6 tensor is partitioned into overlapping 2x2x2 blocks. We use a
    tiling that covers all (i,j,k) triples with 2x2 blocks, saving 1 rank per
    block vs the 8 schoolbook cost.

    Layout: n=3 rows, m=5 cols, p=6 cols.
    We tile with 2x2x2 blocks. For a 3x5x6 grid, a natural tiling uses
    blocks at even offsets. The known construction yields rank 84.

    We embed blocks at offsets (i_off, j_off, k_off) for all valid 2x2x2
    positions, then fill remaining single elements with rank-1 terms.
    """
    n, m, p = DIMENSIONS
    block_U, block_V, block_W = _build_strassen_block()

    # Determine which (i,j,k) cells are covered by 2x2 blocks and which need
    # individual rank-1 terms. We tile greedily: place 2x2x2 blocks at
    # (0,0,0), (0,0,2), (0,0,4), (0,2,0), (0,2,2), (0,2,4), (2,0,0), ...
    # but n=3 so i_off can be 0 or 1. Let's use a systematic tiling.

    # For simplicity and correctness, we use a known valid tiling:
    # i in {0,1} (2 values, covers i=0,1,2 with overlap at 1)
    # Actually, let's use non-overlapping where possible:
    # i: blocks at i_off=0 (covers i=0,1) and i_off=1 (covers i=1,2) -- overlap at i=1
    # This is complex. Instead, use the measured known-good construction:
    # The rank-84 result comes from a specific recursive embedding.

    # Let me use a clean approach: tile the 3x5x6 box with 2x2x2 blocks
    # placed at (i_off, j_off, k_off) where i_off in {0,1}, j_off in {0,2,3},
    # k_off in {0,2,4}. But we need to avoid double-counting.

    # Simpler correct approach: use 2x2x2 blocks on a grid with stride 1 in
    # a way that each cell belongs to exactly one block, except boundaries.
    # This is the standard "block Strassen" embedding.

    # Let me just use a verified construction: place 2x2x2 blocks at all
    # (i_off, j_off, k_off) with i_off in {0,1}, j_off in {0,2,3}, k_off in {0,2,4}
    # and handle overlaps by only using disjoint blocks.

    # Disjoint tiling of 3x5x6:
    # i: block at 0 (covers 0,1), leftover i=2
    # j: blocks at 0 (covers 0,1), 2 (covers 2,3), leftover j=4
    # k: blocks at 0 (covers 0,1), 2 (covers 2,3), 4 (covers 4,5)
    #
    # 2x2x2 blocks: i_off in {0}, j_off in {0,2}, k_off in {0,2,4} -> 1*2*3 = 6 blocks
    # Plus i_off=1: i=1,2 with j_off in {0,2}, k_off in {0,2,4} -> 6 more blocks
    # Total 12 blocks of rank 7 = 84. But this double-counts cells where i=1.

    # The correct non-overlapping tiling:
    # i: 0-1 (block), 2 (single)
    # j: 0-1 (block), 2-3 (block), 4 (single)
    # k: 0-1 (block), 2-3 (block), 4-5 (block)
    #
    # 2x2x2 blocks: i_off=0, j_off in {0,2}, k_off in {0,2,4} -> 6 blocks (rank 7 each)
    # i=2 rows: need 1x2x2 blocks and 1x1x2 blocks etc.
    # This gets complicated. The rank-84 construction is specific.

    # Given the measured success of the recursive approach, let me implement
    # the most straightforward valid construction: use 2x2x2 Strassen blocks
    # wherever a full 2x2x2 sub-box fits, and rank-1 terms elsewhere, ensuring
    # each (i,j,k) is covered exactly once.

    U_full = [[0.0] * 90 for _ in range(n * m)]
    V_full = [[0.0] * 90 for _ in range(m * p)]
    W_full = [[0.0] * 90 for _ in range(p * n)]

    # covered[i][j][k] = True if handled by a block
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]

    c = 0
    # Place 2x2x2 blocks greedily
    for i_off in range(0, n - 1):
        for j_off in range(0, m - 1):
            for k_off in range(0, p - 1):
                # Check if this 2x2x2 block is entirely uncovered
                ok = True
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if covered[i_off+di][j_off+dj][k_off+dk]:
                                ok = False
                if ok:
                    _embed_2x2_block(U_full, V_full, W_full, block_U, block_V, block_W,
                                     i_off, j_off, k_off, c)
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                covered[i_off+di][j_off+dj][k_off+dk] = True
                    c += 7

    # Fill remaining cells with rank-1 terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U_full[i * m + j][c] = 1.0
                    V_full[j * p + k][c] = 1.0
                    W_full[k * n + i][c] = 1.0
                    c += 1

    rank = c
    # Trim to actual rank
    U_out = [row[:rank] for row in U_full]
    V_out = [row[:rank] for row in V_full]
    W_out = [row[:rank] for row in W_full]
    return {"U": U_out, "V": V_out, "W": W_out}


def _verify(decomp, n_trials=20):
    """Verify the decomposition on random matrices."""
    n, m, p = DIMENSIONS
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]

    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B

        # Reconstruct: C[i,k] = sum_j A[i,j] B[j,k]
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_k' V[j*p+k',c] B[j,k'])
        # More precisely: C[i,k] = sum_c W[k*n+i,c] * (U_c . A_i) * (V_c . B_k)
        # where (U_c . A_i) = sum_j U[i*m+j, c] * A[i,j]
        # and (V_c . B_k) = sum_j V[j*p+k, c] * B[j,k]

        C_recon = np.zeros((n, p))
        for c in range(rank):
            u_col = U[:, c]  # shape (n*m,)
            v_col = V[:, c]  # shape (m*p,)
            w_col = W[:, c]  # shape (p*n,)
            # u_col reshaped to (n, m): row i is U[i*m:(i+1)*m, c]
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)  # w_mat[k, i] = W[k*n+i, c]
            # C[i,k] += w_mat[k,i] * (u_mat[i,:] . A[i,:]) * (v_mat[:,k] . B[:,k])
            # u_mat[i,:] . A[i,:] is a scalar per i
            a_part = u_mat * A  # element-wise, shape (n,m) -> sum over j
            a_sum = a_part.sum(axis=1)  # shape (n,)
            b_part = v_mat * B  # shape (m,p)
            b_sum = b_part.sum(axis=0)  # shape (p,)
            # C[i,k] += w_mat[k,i] * a_sum[i] * b_sum[k]
            C_recon += np.outer(a_sum, b_sum) * w_mat.T

        if not np.allclose(C_recon, C_expected, atol=1e-9):
            return False
    return True


def construct():
    start = time.time()
    n, m, p = DIMENSIONS

    # Try the rank-84 construction
    try:
        decomp84 = _build_rank84()
        rank84 = len(decomp84["U"][0])
        if _verify(decomp84):
            return decomp84
    except Exception:
        pass

    # Fallback: schoolbook (always valid, rank 90)
    return _build_schoolbook()
# EVOLVE-BLOCK-END
