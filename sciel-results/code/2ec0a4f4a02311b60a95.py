"""Model-and-optimize: parameterize rank-r factorization and fit/optimize parameters.
Target: exact rank decomposition of 3x4x6 matrix multiplication tensor, minimize rank.

Combined design:
  * Start from a partition-based baseline (rank 66) built from Strassen blocks.
  * Structural sparsity pruning (hypergraph-cover reduction): detect columns whose
    contribution is a scalar multiple of another's and merge them.
  * Randomized merge-and-drop local search to reduce rank while preserving exactness.
  * NumPy-vectorized verification and rank-1 checks for speed.
"""
import random
import time

import numpy as np

n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p

# Index helpers
def Aidx(i, j): return i * m + j          # U row index (n*m)
def Bidx(j, k): return j * p + k          # V row index (m*p)
def Cidx(i, k): return k * n + i          # W row index (p*n), trace convention W[k*n+i]

# Precompute the flattened triple index for the 72 tensor entries
_TRIPLES = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
# For each of the 72 entries, the (U_row, V_row, W_row) indices
_UROW = [Aidx(i, j) for (i, j, k) in _TRIPLES]
_VROW = [Bidx(j, k) for (i, j, k) in _TRIPLES]
_WROW = [Cidx(i, k) for (i, j, k) in _TRIPLES]

def verify(U, V, W):
    """Verify exact decomposition using vectorized numpy. U,V,W are (rows, r) arrays."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    r = Ua.shape[1]
    # contribution[t, l] = U[urow[t], l] * V[vrow[t], l] * W[wrow[t], l]
    contrib = Ua[_UROW, :] * Va[_VROW, :] * Wa[_WROW, :]
    total = contrib.sum(axis=1)
    return bool(np.all(np.abs(total - 1.0) <= 1e-9))

def baseline_66():
    U = np.zeros((n*m, 66), dtype=float)
    V = np.zeros((m*p, 66), dtype=float)
    W = np.zeros((p*n, 66), dtype=float)
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
        A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
        C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
        # M1
        U[A00, col]+=1; U[A11, col]+=1; V[B00, col]+=1; V[B11, col]+=1; W[C00, col]+=1; W[C11, col]+=1; col+=1
        # M2
        U[A10, col]+=1; U[A11, col]+=1; V[B00, col]+=1; W[C10, col]+=1; W[C11, col]-=1; col+=1
        # M3
        U[A00, col]+=1; V[B01, col]+=1; V[B11, col]-=1; W[C01, col]+=1; W[C11, col]+=1; col+=1
        # M4
        U[A11, col]+=1; V[B10, col]+=1; V[B00, col]-=1; W[C00, col]+=1; W[C10, col]+=1; col+=1
        # M5
        U[A00, col]+=1; U[A01, col]+=1; V[B11, col]+=1; W[C00, col]-=1; W[C01, col]+=1; col+=1
        # M6
        U[A10, col]+=1; U[A00, col]-=1; V[B00, col]+=1; V[B01, col]+=1; W[C11, col]+=1; col+=1
        # M7
        U[A01, col]+=1; U[A11, col]-=1; V[B10, col]+=1; V[B11, col]+=1; W[C00, col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        U[A0, col]+=1; V[B00, col]+=1; W[C0, col]+=1; col+=1
        U[A1, col]+=1; V[B10, col]+=1; W[C0, col]+=1; col+=1
        U[A0, col]+=1; V[B01, col]+=1; W[C1, col]+=1; col+=1
        U[A1, col]+=1; V[B11, col]+=1; W[C1, col]+=1; col+=1

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66, col
    return U[:, :col], V[:, :col], W[:, :col]

def term_contribution(U, V, W, l):
    """Return the 72-vector of tensor entries contributed by rank term l."""
    return (U[_UROW, l] * V[_VROW, l] * W[_WROW, l])

def structural_prune(U, V, W, deadline):
    """Merge columns that are scalar multiples of each other (support subset + proportional).

    If term l2's contribution vector is exactly c * term l1's contribution vector, then
    we can fold l2 into l1 (l1 += c*l2) and drop l2. This exploits global structural
    sparsity / proportional hyperedges that the local search may miss.
    """
    U = U.copy(); V = V.copy(); W = W.copy()
    r = U.shape[1]
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        r = U.shape[1]
        contribs = np.stack([term_contribution(U, V, W, l) for l in range(r)], axis=1)  # (72, r)
        for l1 in range(r):
            for l2 in range(l1 + 1, r):
                v1 = contribs[:, l1]; v2 = contribs[:, l2]
                # find a coordinate where v1 != 0
                nz = np.where(np.abs(v1) > 1e-12)[0]
                if nz.size == 0:
                    continue
                c = v2[nz[0]] / v1[nz[0]]
                # check proportionality over all coordinates
                if np.all(np.abs(v2 - c * v1) <= 1e-9 * (1.0 + np.abs(v1))):
                    # fold l2 into l1
                    U[:, l1] += c * U[:, l2]
                    V[:, l1] += c * V[:, l2]
                    W[:, l1] += c * W[:, l2]
                    keep = [x for x in range(r) if x != l2]
                    U = U[:, keep]; V = V[:, keep]; W = W[:, keep]
                    contribs = contribs[:, keep]
                    r -= 1
                    changed = True
                    break
            if changed:
                break
    return U, V, W

def extract_rank1_factors(T):
    """Given a rank-1 tensor T of shape (n,m,p), return (u,v,w) with
    u[i,j]*v[j,k]*w[i,k] == T[i,j,k]. Deterministic via fixing one anchor."""
    u = np.zeros(n*m)
    v = np.zeros(m*p)
    w = np.zeros(p*n)
    # find an anchor (i0,j0,k0) with T != 0
    anchor = None
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, j, k]) > 1e-12:
                    anchor = (i, j, k)
                    break
            if anchor: break
        if anchor: break
    if anchor is None:
        return u, v, w  # zero tensor
    i0, j0, k0 = anchor
    # Set u[i0,j0] = 1, then v[j0,k0] * w[i0,k0] = T[i0,j0,k0].
    # Choose v[j0,k0] = 1, w[i0,k0] = T[i0,j0,k0].
    u[Aidx(i0, j0)] = 1.0
    v[Bidx(j0, k0)] = 1.0
    w[Cidx(i0, k0)] = T[i0, j0, k0]
    # Now propagate. For any (i,j,k):
    #   T[i,j,k] = u[i,j]*v[j,k]*w[i,k]
    # Use the anchor row/col relations.
    # For fixed i, j: pick k0' where w[i,k0'] known and v[j,k0'] known? 
    # Better: use iterative proportional fitting, but since it's rank-1 we can solve directly:
    #   u[i,j] = T[i,j,k0] / (v[j,k0]*w[i,k0])  for any k0 with nonzero denom
    #   v[j,k] = T[i0,j,k] / (u[i0,j]*w[i0,k])  for any i0 with nonzero denom
    #   w[i,k] = T[i,j0,k] / (u[i,j0]*v[j0,k])  for any j0 with nonzero denom
    for _ in range(8):
        # update u
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    denom = v[Bidx(j, k)] * w[Cidx(i, k)]
                    if abs(denom) > 1e-12:
                        u[Aidx(i, j)] = T[i, j, k] / denom
                        break
        # update v
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    denom = u[Aidx(i, j)] * w[Cidx(i, k)]
                    if abs(denom) > 1e-12:
                        v[Bidx(j, k)] = T[i, j, k] / denom
                        break
        # update w
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    denom = u[Aidx(i, j)] * v[Bidx(j, k)]
                    if abs(denom) > 1e-12:
                        w[Cidx(i, k)] = T[i, j, k] / denom
                        break
    return u, v, w

def is_rank1(T):
    """Check if tensor T (n,m,p) is rank-1 via unfolding SVD."""
    M = T.reshape(n*m, p)
    if M.size == 0:
        return True
    s = np.linalg.svd(M, compute_uv=False)
    tol = 1e-9 * max(M.shape) * (s[0] if s.size else 1.0)
    return int(np.sum(s > tol)) <= 1

def try_merge(U, V, W, l1, l2):
    """If T_l1 + T_l2 is rank-1, return (u,v,w) factors of the sum, else None."""
    T = np.zeros((n, m, p))
    for t, (i, j, k) in enumerate(_TRIPLES):
        T[i, j, k] = (U[Aidx(i, j), l1] * V[Bidx(j, k), l1] * W[Cidx(i, k), l1] +
                      U[Aidx(i, j), l2] * V[Bidx(j, k), l2] * W[Cidx(i, k), l2])
    if not is_rank1(T):
        return None
    return extract_rank1_factors(T)

def try_merge3(U, V, W, l1, l2, l3):
    """If T_l1 + T_l2 + T_l3 is rank-1, return (u,v,w) factors of the sum, else None."""
    T = np.zeros((n, m, p))
    for t, (i, j, k) in enumerate(_TRIPLES):
        T[i, j, k] = (U[Aidx(i, j), l1] * V[Bidx(j, k), l1] * W[Cidx(i, k), l1] +
                      U[Aidx(i, j), l2] * V[Bidx(j, k), l2] * W[Cidx(i, k), l2] +
                      U[Aidx(i, j), l3] * V[Bidx(j, k), l3] * W[Cidx(i, k), l3])
    if not is_rank1(T):
        return None
    return extract_rank1_factors(T)

def try_drop(U, V, W, l):
    """Return True if term l contributes nothing (removing it preserves exactness)."""
    c = term_contribution(U, V, W, l)
    return bool(np.all(np.abs(c) <= 1e-12))

def local_search(U, V, W, deadline, rng):
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_r = best_U.shape[1]

    while time.time() < deadline - 3:
        improved = False
        r = best_U.shape[1]

        # Phase 1: drop zero-contribution terms
        for l in range(r):
            if try_drop(best_U, best_V, best_W, l):
                keep = [x for x in range(r) if x != l]
                best_U = best_U[:, keep]; best_V = best_V[:, keep]; best_W = best_W[:, keep]
                best_r = len(keep)
                improved = True
                break
        if improved:
            continue

        # Phase 2: structural proportional merge (cheap, deterministic)
        U2, V2, W2 = structural_prune(best_U, best_V, best_W, deadline)
        if U2.shape[1] < best_r:
            best_U, best_V, best_W = U2, V2, W2
            best_r = best_U.shape[1]
            improved = True
            continue

        # Phase 3: randomized merge pairs (2 -> 1)
        merged = False
        order = list(range(best_r))
        rng.shuffle(order)
        for a in order:
            if time.time() > deadline - 3:
                break
            for b in order:
                if time.time() > deadline - 3:
                    break
                if b <= a:
                    continue
                res = try_merge(best_U, best_V, best_W, a, b)
                if res is not None:
                    u_new, v_new, w_new = res
                    keep = [x for x in range(best_r) if x != a and x != b]
                    newU = np.column_stack([best_U[:, keep], u_new])
                    newV = np.column_stack([best_V[:, keep], v_new])
                    newW = np.column_stack([best_W[:, keep], w_new])
                    # verify exactness before accepting
                    if verify(newU, newV, newW):
                        best_U, best_V, best_W = newU, newV, newW
                        best_r = newU.shape[1]
                        improved = True
                        merged = True
                        break
            if merged:
                break

        # Phase 4: randomized merge triples (3 -> 1)
        if not merged and best_r >= 3:
            for a in order:
                if time.time() > deadline - 3:
                    break
                for b in order:
                    if time.time() > deadline - 3:
                        break
                    if b <= a:
                        continue
                    for c in order:
                        if time.time() > deadline - 3:
                            break
                        if c <= b:
                            continue
                        res = try_merge3(best_U, best_V, best_W, a, b, c)
                        if res is not None:
                            u_new, v_new, w_new = res
                            keep = [x for x in range(best_r) if x != a and x != b and x != c]
                            newU = np.column_stack([best_U[:, keep], u_new])
                            newV = np.column_stack([best_V[:, keep], v_new])
                            newW = np.column_stack([best_W[:, keep], w_new])
                            if verify(newU, newV, newW):
                                best_U, best_V, best_W = newU, newV, newW
                                best_r = newU.shape[1]
                                improved = True
                                merged = True
                                break
                    if merged:
                        break
                if merged:
                    break

        if not improved:
            break

    return best_r, best_U, best_V, best_W

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(42)

    U, V, W = baseline_66()
    r = U.shape[1]

    # Structural sparsity pruning first (deterministic, exploits global structure)
    U, V, W = structural_prune(U, V, W, deadline)
    r = U.shape[1]

    # Local search to drop/merge redundant terms
    r, U, V, W = local_search(U, V, W, deadline, rng)

    # Final verification
    if not verify(U, V, W):
        U, V, W = baseline_66()
        r = 66

    # Convert to list-of-lists interface (integer/half-integer entries)
    def to_list(A):
        return [[float(x) for x in row] for row in A]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
