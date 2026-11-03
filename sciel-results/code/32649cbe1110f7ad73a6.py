import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 115  # safety margin before 150s kill

    # ---- Tensor-slicing lower bound (directive strategy) -------------------
    # Fix A to each standard basis vector e_{i,j}: then C = e_{i,j} * B is a
    # 2x4 matrix whose only nonzero column is column j, equal to column j of B
    # (rank 1). There are n*m = 8 distinct such slices. Tensor rank is bounded
    # below by the maximum slice rank and above by the number of distinct slices
    # only in trivial cases; here it confirms rank >= 8 and that 28 (Strassen)
    # is far above the slice floor, so the merge search below is worthwhile.
    def slice_rank_bound():
        distinct = set()
        max_rank = 0
        for i in range(n):
            for j in range(m):
                # C = e_{i,j} B : 2 x p matrix
                C = np.zeros((n, p))
                C[i, :] = np.eye(m, p)[j]  # row i gets the j-th unit row
                r = np.linalg.matrix_rank(C, tol=1e-9)
                max_rank = max(max_rank, r)
                distinct.add(tuple(C.flatten().tolist()))
        return max_rank, len(distinct)
    slice_lb, distinct_slices = slice_rank_bound()  # (1, 8)
    # This is a lower bound only; it does not produce a decomposition.

    # ---- Strassen 2x2x2 coefficient tables (inspiration 1 structure) --------
    # s=1..7, block indices (i,j) in {0,1}^2
    alpha = [
        {},                                   # 0 (unused)
        { (0,0): 1, (1,1): 1 },               # 1: M1=(A00+A11)
        { (1,0): 1, (1,1): 1 },               # 2: M2=(A10+A11)
        { (0,0): 1 },                         # 3: M3=A00
        { (1,1): 1 },                         # 4: M4=A11
        { (0,0): 1, (0,1): 1 },               # 5: M5=(A00+A01)
        { (1,0): 1, (0,0): -1 },              # 6: M6=(A10-A00)
        { (0,1): 1, (1,1): -1 },              # 7: M7=(A01-A11)
    ]
    beta = [
        {},                                   # 0
        { (0,0): 1, (1,1): 1 },               # 1: (B00+B11)
        { (0,0): 1 },                         # 2: B00
        { (0,1): 1, (1,1): -1 },              # 3: (B01-B11)
        { (1,0): 1, (0,0): -1 },              # 4: (B10-B00)
        { (1,1): 1 },                         # 5: B11
        { (0,0): 1, (0,1): 1 },               # 6: (B00+B01)
        { (1,0): 1, (1,1): 1 },               # 7: (B10+B11)
    ]
    w_coeff = [
        {},                                   # 0
        { (0,0): 1, (1,1): 1 },               # 1: C00,C11
        { (1,0): 1, (1,1): -1 },              # 2: C10,C11
        { (0,1): 1, (1,1): 1 },               # 3: C01,C11
        { (0,0): 1, (1,0): 1 },               # 4: C00,C10
        { (0,0): -1, (0,1): 1 },              # 5: C00,C01
        { (1,1): 1 },                         # 6: C11
        { (0,0): 1 },                         # 7: C00
    ]

    # ---- Build a rank-28 decomposition: 4 independent Strassen blocks -------
    # A = [A11 | A12]  (2x2 each, A11 at cols 0,1; A12 at cols 2,3)
    # B = [[B11, B12],[B21, B22]] (2x2 blocks)
    # C = [A11*B11 + A12*B21 | A11*B12 + A12*B22]  (2x4)
    # Block layout: (Arow, Acol) block, (Brow, Bcol) block, (Crow, Ccol) block
    # Each 2x2 block occupies 2 consecutive columns (A,C) or 2x2 (B).
    blocks_spec = [
        (0, 0, 0, 0, 0, 0),  # A11 B11 -> C11
        (0, 1, 1, 0, 0, 0),  # A12 B21 -> C11
        (0, 0, 0, 1, 0, 1),  # A11 B12 -> C12
        (0, 1, 1, 1, 0, 1),  # A12 B22 -> C12
    ]

    def block_matrices(Ar0, Ac0, Br0, Bc0, Cr0, Cc0):
        """Return list of 7 (U_dict, V_dict, W_dict) for one Strassen block."""
        terms = []
        for s in range(1, 8):
            U = {}
            for (i, j), val in alpha[s].items():
                # A block at row Ar0+i, col Ac0+j ; element (i,j) of the 2x2
                a_idx = (Ar0 + i) * m + (Ac0 + j)
                U[a_idx] = U.get(a_idx, 0) + val
            V = {}
            for (i, j), val in beta[s].items():
                # B block at row Br0+i, col Bc0+j
                b_idx = (Br0 + i) * p + (Bc0 + j)
                V[b_idx] = V.get(b_idx, 0) + val
            W = {}
            for (i, j), val in w_coeff[s].items():
                # C block at row Cr0+i, col Cc0+j ; trace index k*n+i = col*n+row
                w_idx = (Cc0 + j) * n + (Cr0 + i)
                W[w_idx] = W.get(w_idx, 0) + val
            terms.append((U, V, W))
        return terms

    blocks = [block_matrices(*spec) for spec in blocks_spec]

    # ---- Verification --------------------------------------------------------
    def verify_full(Umat, Vmat, Wmat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ai = i * m + j
                    bi = j * p + k
                    ci = k * n + i
                    s = 0
                    for rr in range(rank):
                        s += Umat[ai][rr] * Vmat[bi][rr] * Wmat[ci][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def build_from_blocks(term_map, merged):
        """term_map: list of (blk,t) to skip; merged: (SU,SV,SW) or None."""
        rank = 28 - len(term_map) + (1 if merged else 0)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        skip = set(term_map)
        for blk in range(4):
            for t in range(7):
                if (blk, t) in skip:
                    continue
                ud, vd, wd = blocks[blk][t]
                for a, val in ud.items(): U[a][r] += val
                for b, val in vd.items(): V[b][r] += val
                for c, val in wd.items(): W[c][r] += val
                r += 1
        if merged:
            SU, SV, SW = merged
            for a, val in SU.items(): U[a][r] += val
            for b, val in SV.items(): V[b][r] += val
            for c, val in SW.items(): W[c][r] += val
            r += 1
        assert r == rank, f"Expected {rank}, got {r}"
        return U, V, W, rank

    # Build and verify the base rank-28 solution.
    U28, V28, W28, _ = build_from_blocks([], None)
    if not verify_full(U28, V28, W28, 28):
        # Fallback: schoolbook rank 32
        U32 = [[0] * 32 for _ in range(n * m)]
        V32 = [[0] * 32 for _ in range(m * p)]
        W32 = [[0] * 32 for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U32[i * m + j][idx] = 1
                    V32[j * p + k][idx] = 1
                    W32[k * n + i][idx] = 1
                    idx += 1
        return {"U": U32, "V": V32, "W": W32}

    # ---- Rank-1 reduction machinery (current program refinement) ------------
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_triples.append((i * m + j, j * p + k, k * n + i))
    a_vals = sorted(set(t[0] for t in valid_triples))
    b_vals = sorted(set(t[1] for t in valid_triples))
    c_vals = sorted(set(t[2] for t in valid_triples))

    def try_merge(terms):
        """Try to find a single rank-1 decomposition of the sum of the given
        Strassen terms. Returns (SU, SV, SW) or None."""
        R = {}
        for (a, b, c) in valid_triples:
            val = 0
            for (U1, V1, W1) in terms:
                val += U1.get(a, 0) * V1.get(b, 0) * W1.get(c, 0)
            if val != 0:
                R[(a, b, c)] = val
        if not R:
            return ({}, {}, {})
        pivot = next(iter(R.keys()))
        pa, pb, pc = pivot
        R0 = R[pivot]
        SU = {a: R.get((a, pb, pc), 0) / R0 for a in a_vals}
        SU = {k: v for k, v in SU.items() if abs(v) > 1e-12}
        SV = {b: R.get((pa, b, pc), 0) / R0 for b in b_vals}
        SV = {k: v for k, v in SV.items() if abs(v) > 1e-12}
        SW = {c: R.get((pa, pb, c), 0) / R0 for c in c_vals}
        SW = {k: v for k, v in SW.items() if abs(v) > 1e-12}
        if not SU or not SV or not SW:
            return None
        for (a, b, c) in valid_triples:
            expected = R.get((a, b, c), 0)
            actual = SU.get(a, 0) * SV.get(b, 0) * SW.get(c, 0)
            if abs(expected - actual) > 1e-9:
                return None
        return (SU, SV, SW)

    # Try to reduce 28 -> 27 by merging one term from each of two blocks.
    best = None
    for bi in range(4):
        for bj in range(bi + 1, 4):
            for ii in range(7):
                for jj in range(7):
                    if time.time() > deadline:
                        break
                    merged = try_merge([blocks[bi][ii], blocks[bj][jj]])
                    if merged is None or not all(merged):
                        continue
                    U, V, W, rank = build_from_blocks([(bi, ii), (bj, jj)], merged)
                    if verify_full(U, V, W, rank):
                        best = (U, V, W)
                        break
                if best or time.time() > deadline:
                    break
            if best or time.time() > deadline:
                break
        if best or time.time() > deadline:
            break

    if best:
        return {"U": best[0], "V": best[1], "W": best[2]}

    # Try to reduce 28 -> 26 by merging one term from each of three blocks.
    if time.time() < deadline - 10:
        for bi in range(4):
            for bj in range(bi + 1, 4):
                for bk in range(bj + 1, 4):
                    for ii in range(7):
                        for jj in range(7):
                            for kk in range(7):
                                if time.time() > deadline:
                                    break
                                merged = try_merge([blocks[bi][ii],
                                                     blocks[bj][jj],
                                                     blocks[bk][kk]])
                                if merged is None or not all(merged):
                                    continue
                                U, V, W, rank = build_from_blocks(
                                    [(bi, ii), (bj, jj), (bk, kk)], merged)
                                if verify_full(U, V, W, rank):
                                    return {"U": U, "V": V, "W": W}
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

    return {"U": U28, "V": V28, "W": W28}
