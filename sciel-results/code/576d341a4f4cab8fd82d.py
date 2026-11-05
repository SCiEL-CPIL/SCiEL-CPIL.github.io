"""Search for a lower-rank exact tensor decomposition of 3x5 * 5x7 matrix multiplication."""
import time
import itertools
import numpy as np

N, M, P = 3, 5, 7
RANK_SCHOOLBOOK = N * M * P  # 105


def _build_tensor():
    """Return dict {(a,b,c): 1} for the matmul tensor support."""
    T = {}
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[(i * M + j, j * P + k, k * N + i)] = 1
    return T


def _validate_decomposition(U, V, W):
    """Check U,V,W exactly decompose the tensor. U:(N*M,rank) V:(M*P,rank) W:(P*N,rank)."""
    T = _build_tensor()
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # non-zero entries must equal 1
    for (a, b, c) in T:
        s = float(np.dot(Ua[a], Va[b] * Wa[c]))
        if abs(s - 1.0) > 1e-9:
            return False
    # zero entries must equal 0
    for a in range(N * M):
        for b in range(M * P):
            if (a, b, 0) in T:
                pass
    # full check of zeros
    UaT = Ua.T  # (rank, N*M)
    VaT = Va.T
    WaT = Wa.T
    # For each a,b compute vector over c: sum_r U[a,r] V[b,r] W[c,r] = (Ua[a] * Va[b]) @ WaT
    for a in range(N * M):
        for b in range(M * P):
            vec = (Ua[a] * Va[b]) @ WaT  # length P*N
            for c in range(P * N):
                if (a, b, c) in T:
                    if abs(float(vec[c]) - 1.0) > 1e-9:
                        return False
                else:
                    if abs(float(vec[c])) > 1e-9:
                        return False
    return True


def _schoolbook_decomposition():
    rank = RANK_SCHOOLBOOK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_terms():
    """7 Strassen terms for 2x2x2: each term is (u_mask, v_mask, w_mask) where mask is 2x2 of +/-1 or 0.
    Returns list of (U2x2, V2x2, W2x2) integer matrices."""
    terms = []
    # Standard Strassen:
    # M1 = (A11)(B12-B22)
    # M2 = (A11+A12)(B22)
    # M3 = (A21-A11)(B11)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A22)(B11+B22)
    # M6 = (A12-A22)(B21+B22)
    # M7 = (A21-A12)(B11+B12)
    A11 = np.array([[1, 0], [0, 0]])
    A12 = np.array([[0, 1], [0, 0]])
    A21 = np.array([[0, 0], [1, 0]])
    A22 = np.array([[0, 0], [0, 1]])
    B11 = np.array([[1, 0], [0, 0]])
    B12 = np.array([[0, 1], [0, 0]])
    B21 = np.array([[0, 0], [1, 0]])
    B22 = np.array([[0, 0], [0, 1]])
    terms.append((A11, B12 - B22, np.eye(2, dtype=int)))
    terms.append((A11 + A12, B22, np.eye(2, dtype=int)))
    terms.append((A21 - A11, B11, np.eye(2, dtype=int)))
    terms.append((A22, B21 - B11, np.eye(2, dtype=int)))
    terms.append((A11 + A22, B11 + B22, np.eye(2, dtype=int)))
    terms.append((A12 - A22, B21 + B22, np.eye(2, dtype=int)))
    terms.append((A21 - A12, B11 + B12, np.eye(2, dtype=int)))
    return terms


def _all_partitions(n, part_len):
    """All ways to split n consecutive indices into blocks each of length <= part_len (allowing smaller blocks).
    Returns list of tuples of (start, length) for each block, covering 0..n-1 in order."""
    results = []
    def rec(pos, blocks):
        if pos == n:
            results.append(tuple(blocks))
            return
        if pos > n:
            return
        for ln in range(1, min(part_len, n - pos) + 1):
            rec(pos + ln, blocks + [(pos, ln)])
    rec(0, [])
    return results


def _compute_rank(part_i, part_j, part_k, terms):
    """Compute the rank of the decomposition given partitions of i (N), j (M), k (P) into 2-blocks,
    using the given strassen terms. Returns rank and the decomposition matrices."""
    # part_i: list of (start, length) covering 0..N-1, each length in {1,2}
    # For each block pair (bi in part_i, bj in part_j, bk in part_k) we place one strassen term
    # But a block of length 1 is just a scalar (1x1). We need to map terms appropriately.
    # Simpler: only allow all blocks of length exactly 2 except we pad. Instead, let's require
    # partitions where each block is length 1 or 2, and for a block of length 1 we use the 1x1 identity term.
    # Actually cleaner: build U,V,W by iterating over block triples and assigning one rank slot per term.
    # For a block of size s (1 or 2), the term matrices are s x s.
    rank = 0
    U = [[0] * 0 for _ in range(N * M)]  # placeholder
    # We'll build column by column.
    cols_U = []
    cols_V = []
    cols_W = []
    for bi, (si, li) in enumerate(part_i):
        for bj, (sj, lj) in enumerate(part_j):
            for bk, (sk, lk) in enumerate(part_k):
                # Determine term based on sizes
                if li == 2 and lj == 2 and lk == 2:
                    # Use the 7 strassen terms
                    for (U2, V2, W2) in terms:
                        colU = [0] * (N * M)
                        colV = [0] * (M * P)
                        colW = [0] * (P * N)
                        for ii in range(2):
                            for jj in range(2):
                                i = si + ii
                                j = sj + jj
                                colU[i * M + j] = int(U2[ii, jj])
                        for jj in range(2):
                            for kk in range(2):
                                j = sj + jj
                                k = sk + kk
                                colV[j * P + k] = int(V2[jj, kk])
                        for kk in range(2):
                            for ii in range(2):
                                k = sk + kk
                                i = si + ii
                                colW[k * N + i] = int(W2[kk, ii])
                        cols_U.append(colU)
                        cols_V.append(colV)
                        cols_W.append(colW)
                else:
                    # Fallback: scalar schoolbook for this triple (li*lj*lk terms)
                    for ii in range(li):
                        for jj in range(lj):
                            for kk in range(lk):
                                i = si + ii
                                j = sj + jj
                                k = sk + kk
                                colU = [0] * (N * M)
                                colV = [0] * (M * P)
                                colW = [0] * (P * N)
                                colU[i * M + j] = 1
                                colV[j * P + k] = 1
                                colW[k * N + i] = 1
                                cols_U.append(colU)
                                cols_V.append(colV)
                                cols_W.append(colW)
    rank = len(cols_U)
    U = [[cols_U[r][a] for r in range(rank)] for a in range(N * M)]
    V = [[cols_V[r][b] for r in range(rank)] for b in range(M * P)]
    W = [[cols_W[r][c] for r in range(rank)] for c in range(P * N)]
    return rank, {"U": U, "V": V, "W": W}


def _build_decomposition(part_i, part_j, part_k, terms):
    rank, decomp = _compute_rank(part_i, part_j, part_k, terms)
    return rank, decomp


def _try_3x3x3_block():
    """Try partitions of N=3, M=5, P=7 into blocks of size 1 or 2, evaluate rank, validate."""
    terms = _strassen_terms()
    parts_i = _all_partitions(N, 2)
    parts_j = _all_partitions(M, 2)
    parts_k = _all_partitions(P, 2)
    best_rank = RANK_SCHOOLBOOK
    best_decomp = None
    for pi in parts_i:
        for pj in parts_j:
            for pk in parts_k:
                rank, decomp = _build_decomposition(pi, pj, pk, terms)
                if rank < best_rank:
                    if _validate_decomposition(decomp["U"], decomp["V"], decomp["W"]):
                        best_rank = rank
                        best_decomp = decomp
    return best_rank, best_decomp


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)

    # Start with schoolbook
    best = _schoolbook_decomposition()
    best_rank = RANK_SCHOOLBOOK

    # Try block-partition Strassen constructions
    rank, decomp = _try_3x3x3_block()
    if decomp is not None and rank < best_rank:
        best = decomp
        best_rank = rank

    return best
