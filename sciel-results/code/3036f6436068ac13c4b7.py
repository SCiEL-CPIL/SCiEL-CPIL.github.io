"""Matrix multiplication 2x4x4: low-rank exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0
    n, m, p = 2, 4, 4

    def _build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
        return U, V, W, rank

    def _verify(U, V, W, rank):
        """Check T[i,j,k]=sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Baseline
    U0, V0, W0, rank0 = _build_schoolbook()
    best_res = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = rank0

    # Try merging pairs of basis tensors e_{i,j,k} + e_{i',j',k'} into one rank-1 term.
    # e_a + e_b is rank-1 multiplicative iff there exist A(2x4),B(4x4),C(4x2) with
    # A[i,j]*B[j,k]*C[k,i] = 1 at the two support points, 0 elsewhere.
    def _merge_pair(ia, ja, ka, ib, jb, kb):
        """Return (A, B, C) if e_{ia,ja,ka}+e_{ib,jb,kb} is rank-1 multiplicative, else None."""
        S_A = {(ia, ja), (ib, jb)}
        S_B = {(ja, ka), (jb, kb)}
        S_C = {(ka, ia), (kb, ib)}
        # Join must equal exactly the two support points
        join = set()
        for (i, j) in S_A:
            for (jj, k) in S_B:
                if jj != j:
                    continue
                for (kk, ii) in S_C:
                    if kk == k and ii == i:
                        join.add((i, j, k))
        target = {(ia, ja, ka), (ib, jb, kb)}
        if join != target:
            return None
        # Assign signs
        for sA1, sB1, sC1 in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]:
            A = np.zeros((n, m)); B = np.zeros((m, p)); C = np.zeros((p, n))
            A[ia, ja] = sA1; B[ja, ka] = sB1; C[ka, ia] = sC1
            A2_fix = (ib, jb) == (ia, ja)
            B2_fix = (jb, kb) == (ja, ka)
            C2_fix = (kb, ib) == (ka, ia)
            A2 = sA1 if A2_fix else None
            B2 = sB1 if B2_fix else None
            C2 = sC1 if C2_fix else None
            if A2_fix and B2_fix and C2_fix:
                if A2 * B2 * C2 != 1:
                    continue
                A[ib, jb] = A2; B[jb, kb] = B2; C[kb, ib] = C2
            elif not A2_fix:
                A2v = B2 * C2
                A[ib, jb] = A2v; B[jb, kb] = B2; C[kb, ib] = C2
            elif not B2_fix:
                B2v = A2 * C2
                A[ib, jb] = A2; B[jb, kb] = B2v; C[kb, ib] = C2
            elif not C2_fix:
                C2v = A2 * B2
                A[ib, jb] = A2; B[jb, kb] = B2; C[kb, ib] = C2v
            else:
                # All fixed and consistent
                if A2 * B2 * C2 != 1:
                    continue
                A[ib, jb] = A2; B[jb, kb] = B2; C[kb, ib] = C2
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        exp = 1.0 if ((i, j, k) == (ia, ja, ka) or (i, j, k) == (ib, jb, kb)) else 0.0
                        got = A[i, j] * B[j, k] * C[k, i]
                        if abs(got - exp) > 1e-12:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return (A, B, C)
        return None

    # Find all valid merges
    merges = []  # list of (a, b, A, B, C)
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(m):
            for k in range(p):
                a = i * m * p + j * p + k
                for i2 in range(n):
                    if time.time() > deadline:
                        break
                    for j2 in range(m):
                        for k2 in range(p):
                            b = i2 * m * p + j2 * p + k2
                            if b <= a:
                                continue
                            res = _merge_pair(i, j, k, i2, j2, k2)
                            if res is not None:
                                merges.append((a, b) + res)
    # Greedy non-overlapping merges
    if merges:
        used = set()
        chosen = []
        for (a, b, A, B, C) in merges:
            if a in used or b in used:
                continue
            used.add(a); used.add(b)
            chosen.append((a, b, A, B, C))
        if chosen:
            new_rank = nT - len(chosen) if (nT := n*m*p) else 0
            new_rank = n * m * p - len(chosen)
            U = np.zeros((n * m, new_rank))
            V = np.zeros((m * p, new_rank))
            W = np.zeros((p * n, new_rank))
            r = 0
            merged_set = set()
            for (a, b, A, B, C) in chosen:
                merged_set.add(a); merged_set.add(b)
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, r] = A[i, j]
                        for k in range(p):
                            V[j * p + k, r] = B[j, k]
                            W[k * n + i, r] = C[k, i]
                r += 1
            for idx in range(n * m * p):
                if idx in merged_set:
                    continue
                i = idx // (m * p); j = (idx // p) % m; k = idx % p
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
            if _verify(U, V, W, new_rank):
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_res = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return best_res
# EVOLVE-BLOCK-END
