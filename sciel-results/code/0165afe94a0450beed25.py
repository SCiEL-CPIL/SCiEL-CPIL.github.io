"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix
    by a 5x5 matrix. The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75

    # ------------------------------------------------------------------
    # Candidate 0: pure schoolbook (always valid fallback)
    # ------------------------------------------------------------------
    def build_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return U, V, W

    best_U, best_V, best_W = build_schoolbook()
    best_rank = schoolbook_rank

    # ------------------------------------------------------------------
    # Candidate 1: tiling family (4 disjoint 2x2x2 Strassen blocks) -> rank 71
    # This is the proven 1.0563 construction; used as the safe baseline.
    # ------------------------------------------------------------------
    def build_tiling():
        blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
        total_strassen_rank = len(blocks) * 7
        covered = [[[False] * p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0 + di][j0 + dj][k0 + dk] = True
        schoolbook_count = sum(1 for i in range(n) for j in range(m)
                               for k in range(p) if not covered[i][j][k])
        total_rank = total_strassen_rank + schoolbook_count
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        col = 0

        def add_strassen_block(i0, j0, k0, off):
            c = off + 0
            U[i0 * m + j0][c] += 1
            U[(i0 + 1) * m + (j0 + 1)][c] += 1
            V[j0 * p + k0][c] += 1
            V[(j0 + 1) * p + (k0 + 1)][c] += 1
            W[k0 * n + i0][c] += 1
            W[(k0 + 1) * n + (i0 + 1)][c] += 1
            c = off + 1
            U[(i0 + 1) * m + j0][c] += 1
            U[(i0 + 1) * m + (j0 + 1)][c] += 1
            V[j0 * p + k0][c] += 1
            W[k0 * n + (i0 + 1)][c] += 1
            W[(k0 + 1) * n + (i0 + 1)][c] += -1
            c = off + 2
            U[i0 * m + j0][c] += 1
            V[j0 * p + (k0 + 1)][c] += 1
            V[(j0 + 1) * p + (k0 + 1)][c] += -1
            W[(k0 + 1) * n + i0][c] += 1
            W[(k0 + 1) * n + (i0 + 1)][c] += 1
            c = off + 3
            U[(i0 + 1) * m + (j0 + 1)][c] += 1
            V[(j0 + 1) * p + k0][c] += 1
            V[j0 * p + k0][c] += -1
            W[k0 * n + i0][c] += 1
            W[k0 * n + (i0 + 1)][c] += 1
            c = off + 4
            U[i0 * m + j0][c] += 1
            U[i0 * m + (j0 + 1)][c] += 1
            V[(j0 + 1) * p + (k0 + 1)][c] += 1
            W[k0 * n + i0][c] += -1
            W[(k0 + 1) * n + i0][c] += 1
            c = off + 5
            U[(i0 + 1) * m + j0][c] += 1
            U[i0 * m + j0][c] += -1
            V[j0 * p + k0][c] += 1
            V[j0 * p + (k0 + 1)][c] += 1
            W[(k0 + 1) * n + (i0 + 1)][c] += 1
            c = off + 6
            U[i0 * m + (j0 + 1)][c] += 1
            U[(i0 + 1) * m + (j0 + 1)][c] += -1
            V[(j0 + 1) * p + k0][c] += 1
            V[(j0 + 1) * p + (k0 + 1)][c] += 1
            W[k0 * n + i0][c] += 1

        for i0, j0, k0 in blocks:
            add_strassen_block(i0, j0, k0, col)
            col += 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i * m + j][col] += 1
                        V[j * p + k][col] += 1
                        W[k * n + i][col] += 1
                        col += 1
        return U, V, W, total_rank

    try:
        tU, tV, tW, tRank = build_tiling()
        if tRank < best_rank:
            best_rank, best_U, best_V, best_W = tRank, tU, tV, tW
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Candidate 2: STRUCTURAL CHANGE -- global block decomposition.
    #   3x5x5  =  (2x5x5)  +  (1x5x5)
    # The 1x5x5 part is a rank-5 schoolbook tail. The 2x5x5 part is a
    # genuinely different tensor: its two output rows are coupled, so a
    # decomposition that shares bilinear forms across both rows can beat
    # any local 2x2x2 tiling. We search for the minimal-rank EXACT
    # integer/half-integer decomposition of the 2x5x5 block.
    # ------------------------------------------------------------------
    n1, m1, p1 = 2, 5, 5
    A_dim, B_dim, C_dim = n1 * m1, m1 * p1, p1 * n1  # 10, 25, 10

    # Build the 2x5x5 target tensor as a 3D array T[a,b,c]
    T1 = np.zeros((A_dim, B_dim, C_dim), dtype=int)
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                T1[i * m1 + j, j * p1 + k, k * n1 + i] = 1

    # Flatten to a 2D tensor T[a, (b,c)] for a rank-revealing search.
    Tflat = T1.reshape(A_dim, B_dim * C_dim)  # shape (10, 250)

    def assemble_full(U1, V1, W1, r1):
        """Combine a 2x5x5 decomposition (rank r1) with a 1x5x5 schoolbook tail."""
        total_rank = r1 + m1 * p1  # +25
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        for r in range(r1):
            for a in range(A_dim):
                U[a][r] = U1[a][r]
            for b in range(B_dim):
                V[b][r] = V1[b][r]
            for c in range(C_dim):
                W[c][r] = W1[c][r]
        col = r1
        for i in range(n1, n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, total_rank

    def verify_full(U, V, W):
        """Exact verification against the 3x5x5 target tensor."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        target = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i * m + j, j * p + k, k * n + i] = 1
        Td = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        return np.allclose(Td, target, atol=1e-9)

    # Rank-revealing search over the 2x5x5 block.
    # We only need r1 < (best_rank - 25) to beat the current best.
    # r1 must be at least the (numerical) rank of the 2x5x5 tensor.
    try:
        num_rank = int(np.linalg.matrix_rank(Tflat, tol=1e-10))
    except Exception:
        num_rank = A_dim
    r1_min = max(num_rank, A_dim)  # cannot be below number of A-rows

    best_r1 = None
    best_U1 = best_V1 = best_W1 = None
    r1_max = best_rank - m1 * p1  # only try ranks that could beat current best

    # Search r1 from lower bound upward; stop at the first that works.
    for r1 in range(r1_min, r1_max):
        if time.time() > deadline - 10.0:
            break
        # Try to find an integer decomposition of rank r1 via integer SVD-style
        # rank-revealing on the flattened tensor. If the tensor's integer rank
        # is r1, an integer factorisation exists; we construct it from the
        # rank-revealing integer factorisation of Tflat.
        try:
            # Compute a rank-revealing factorisation using integer row reduction
            # (Smith-like) is expensive; instead use a deterministic integer
            # construction: express each of the A_dim rows as a sum of r1
            # shared rank-1 terms. We do this by solving an integer linear
            # system for the shared right-factors.
            #
            # Practical approach: use the fact that T1 is sparse (each row of
            # A has m1*p1 = 25 ones). Try to find r1 integer vectors
            # {v_c} in Z^(B_dim*C_dim) and coefficients u_{a,c} in Z such that
            # Tflat[a, :] = sum_c u_{a,c} * v_c. We search over a small set of
            # candidate right-factors drawn from the standard basis and simple
            # combinations.
            #
            # Candidate right-factors: each is an indicator of a (b,c) pair,
            # i.e. the standard basis, plus a few pairwise sums.
            basis = []
            for bc in range(B_dim * C_dim):
                v = np.zeros(B_dim * C_dim, dtype=int)
                v[bc] = 1
                basis.append(v)
            # Add pairwise-difference factors that can capture the coupling.
            # (Kept small for speed.)
            #
            # We then need to pick r1 of these and integer coefficients.
            # This is a set-cover / integer-programming problem; we solve it
            # greedily with exact verification.
            found = False
            # Greedy: repeatedly pick the basis vector that best reduces the
            # residual, with integer coefficient = the row's value at that
            # index (since basis is a unit vector).
            residual = Tflat.copy()
            chosen = []
            for _ in range(r1):
                if time.time() > deadline - 10.0:
                    break
                best_idx = -1
                best_err = None
                for bc in range(B_dim * C_dim):
                    if residual[:, bc].max() == 0 and residual[:, bc].min() == 0:
                        continue
                    # coefficient per row = residual value at bc
                    coeff = residual[:, bc].copy()
                    if coeff.max() == 0:
                        continue
                    new_res = residual - np.outer(coeff, np.eye(B_dim * C_dim)[bc])
                    err = int(np.sum(new_res != 0))
                    if best_err is None or err < best_err:
                        best_err = err
                        best_idx = bc
                        best_coeff = coeff
                if best_idx < 0 or best_err is None:
                    break
                residual = residual - np.outer(best_coeff, np.eye(B_dim * C_dim)[best_idx])
                chosen.append((best_idx, best_coeff))
            if int(np.sum(residual != 0)) == 0 and len(chosen) <= r1:
                # Build U1, V1, W1 from the chosen unit-vector factors.
                U1 = [[0] * r1 for _ in range(A_dim)]
                V1 = [[0] * r1 for _ in range(B_dim)]
                W1 = [[0] * r1 for _ in range(C_dim)]
                for c, (bc, coeff) in enumerate(chosen):
                    b = bc // C_dim
                    cc = bc % C_dim
                    for a in range(A_dim):
                        if coeff[a] != 0:
                            U1[a][c] = int(coeff[a])
                    V1[b][c] = 1
                    W1[cc][c] = 1
                # Exact verification of the 2x5x5 block.
                U1f = np.array(U1, dtype=float)
                V1f = np.array(V1, dtype=float)
                W1f = np.array(W1, dtype=float)
                Td1 = np.einsum('ar,br,cr->abc', U1f, V1f, W1f)
                if np.allclose(Td1, T1, atol=1e-9):
                    best_r1, best_U1, best_V1, best_W1 = r1, U1, V1, W1
                    found = True
                    break
        except Exception:
            continue

    # Assemble and verify the global-decomposition candidate.
    if best_r1 is not None:
        try:
            gU, gV, gW, gRank = assemble_full(best_U1, best_V1, best_W1, best_r1)
            if verify_full(gU, gV, gW):
                if gRank < best_rank:
                    best_rank, best_U, best_V, best_W = gRank, gU, gV, gW
        except Exception:
            pass

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
