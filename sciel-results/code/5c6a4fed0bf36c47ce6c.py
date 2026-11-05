"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110.0   # hard wall-clock guard (evaluator kills at 150s)


def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}))
    terms.append(({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}))
    terms.append(({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}))
    terms.append(({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}))
    terms.append(({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}))
    terms.append(({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}))
    terms.append(({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}))
    return terms


def _winograd_3x3x3_terms():
    """Return the 23 Winograd terms for 3x3x3 matrix multiplication.
    Uses the standard Winograd F(3,3,3) algorithm with rank 23."""
    # Winograd's algorithm for 3x3 matrix multiplication
    # Based on the factorization: F = A^T * M * B where A, B are 3x23, M is 23x23 diagonal
    # We need terms (u_pat, v_pat, w_pat) for each of the 23 multiplications
    
    # The 23 terms come from the Winograd factorization.
    # Let me construct them from the known Winograd F(3,3,3) decomposition.
    
    # Standard Winograd 3x3x3 uses these patterns:
    # I'll use the well-known 23-term decomposition
    
    terms = []
    
    # Term 0: (a00+a01+a02)(b00+b10+b20)
    u = {(0,0):1, (0,1):1, (0,2):1}
    v = {(0,0):1, (1,0):1, (2,0):1}
    w = {(0,0):1}  # contributes to c[0,0]
    terms.append((u, v, w))
    
    # Actually, let me use a cleaner approach. The Winograd 3x3x3 algorithm
    # can be described as follows. Let me use the standard formulation.
    
    # Reset and use a known correct 23-term Winograd decomposition
    terms = []
    
    # I'll use the formulation from Winograd's paper / standard references.
    # The key insight: we precompute linear combinations of A and B, 
    # do 23 scalar multiplications, then combine to get C.
    
    # Let me define the 23 terms explicitly using the known Winograd F(3,3,3):
    # Each term t has:
    #   u_pat: dict from (i,j) in A to coefficient
    #   v_pat: dict from (j,k) in B to coefficient  
    #   w_pat: dict from (k,i) in C to coefficient (output index is (k,i) per convention W[k*n+i])
    
    # Actually the convention is W[k*n+i, col] where the output C[i,k] gets contribution
    # Let me re-read: W index is (lk, li) meaning output position.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # In our notation: U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    # So for the 3x3x3 block with local indices (li, lj, lk):
    # U[li, lj] * V[lj, lk] * W[lk, li] should sum to delta(li,lj,lk) = 1 if this is the (li,lj,lk) entry
    
    # Wait, I need to be more careful. The tensor T[li, lj, lk] = 1 if we're computing C[li,lk] from A[li,lj]*B[lj,lk]
    # So the decomposition must satisfy: sum_col U[li,lj,col] * V[lj,lk,col] * W[lk,li,col] = 1 for all (li,lj,lk)
    
    # This is the standard matrix multiplication tensor of size 3x3x3.
    
    # Let me use a verified 23-term Winograd decomposition.
    # Based on the standard Winograd F(3,3,3) algorithm:
    
    # The 23 multiplications in Winograd's 3x3x3 are:
    # I'll construct them from the factorization matrices.
    
    # From Winograd's algorithm, the 23 terms can be written as:
    # Let me use a concrete set from literature.
    
    # Actually, let me just implement the standard Winograd F(3,3,3) directly.
    # The algorithm computes C = A*B for 3x3 matrices using 23 multiplications.
    
    # Standard Winograd F(3,3,3) intermediate values:
    # d0 = a00 + a01 + a02
    # d1 = a00 + a10 + a20  
    # d2 = a01 + a11 + a21
    # d3 = a02 + a12 + a22
    # d4 = a00 - a01
    # d5 = a10 - a11
    # d6 = a20 - a21
    # d7 = a01 - a02
    # d8 = a11 - a12
    # d9 = a21 - a22
    # d10 = a00 - a10
    # d11 = a01 - a11
    # d12 = a02 - a12
    # d13 = a10 - a20
    # d14 = a11 - a21
    # d15 = a12 - a22
    # d16 = a00 + a11 + a22
    # d17 = a01 + a12 + a20
    # d18 = a02 + a10 + a21
    # d19 = a00 + a10 + a20 + a01 + a11 + a21 + a02 + a12 + a22  (this is sum of all, but that's not right for 23 terms)
    
    # This is getting complicated. Let me use a different approach.
    # I'll use the known fact that the minimum rank of 3x3x3 matrix multiplication 
    # is at most 23 (Winograd) and construct the terms programmatically.
    
    # For now, let me use a simpler but still effective approach:
    # Use Strassen where possible (2x2x2 blocks) and schoolbook elsewhere.
    # The 3x3x3 case is tricky to implement correctly, so let me focus on
    # optimizing the partition strategy with 2x2x2 Strassen blocks.
    
    # Return empty for now - we'll handle 3x3x3 with schoolbook or a simpler method
    return []


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_recursive_strassen(n, m, p, i_groups, j_groups, k_groups):
    """Build a decomposition by splitting each dimension into groups.
    Any (2,2,2) block uses the 7-term Strassen trick; others use schoolbook."""
    nm, mp, pn = n * m, m * p, p * n
    strassen_terms = _strassen_2x2x2_terms()

    total_rank = 0
    block_info = []
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                r = 7 if (si == 2 and sj == 2 and sk == 2) else si * sj * sk
                total_rank += r
                block_info.append((ig, jg, kg, r))

    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]

    col = 0
    for ig, jg, kg, r in block_info:
        si, sj, sk = len(ig), len(jg), len(kg)
        if si == 2 and sj == 2 and sk == 2:
            for (u_pat, v_pat, w_pat) in strassen_terms:
                for (li, lj), val_u in u_pat.items():
                    gi, gj = ig[li], jg[lj]
                    U[gi * m + gj][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    gj, gk = jg[lj], kg[lk]
                    V[gj * p + gk][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    gk, gi = kg[lk], ig[li]
                    W[gk * n + gi][col] = val_w
                col += 1
        else:
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = ig[li], jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy matrix multiplication."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n

    for c in range(pn):
        k = c // n
        i = c % n
        UW = U * W[c, :]
        M = UW @ V.T

        for j in range(m):
            a = i * m + j
            b = j * p + k
            if abs(M[a, b] - 1.0) > 1e-9:
                return False
        for a in range(nm):
            a_i = a // m
            a_j = a % m
            for b in range(mp):
                b_j = b // p
                b_k = b % p
                if a_i == i and a_j == b_j and b_k == k:
                    continue
                if abs(M[a, b]) > 1e-9:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=100):
    """Try to reduce rank by removing columns that are linearly dependent
    across all three factor matrices simultaneously."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]

    for _ in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        reduced = False
        for target in range(r):
            others = [i for i in range(r) if i != target]
            if len(others) == 0:
                continue

            ok = True
            for M in (U, V, W):
                M_others = M[:, others]
                M_target = M[:, target]
                if np.linalg.matrix_rank(np.column_stack([M_others, M_target])) > \
                        np.linalg.matrix_rank(M_others):
                    ok = False
                    break
            if not ok:
                continue

            U = np.delete(U, target, axis=1)
            V = np.delete(V, target, axis=1)
            W = np.delete(W, target, axis=1)
            r -= 1
            reduced = True
            break

        if not reduced:
            break

    return U.tolist(), V.tolist(), W.tolist()


def _generate_partitions(n, m, p):
    """Generate candidate groupings for each dimension. Each dimension is split
    into contiguous groups of size 1 or 2."""
    def _partitions(dim):
        results = []
        def rec(start, cur):
            if start == dim:
                results.append(list(cur))
                return
            if start + 1 <= dim:
                cur.append([start])
                rec(start + 1, cur)
                cur.pop()
            if start + 2 <= dim:
                cur.append([start, start + 1])
                rec(start + 2, cur)
                cur.pop()
        rec(0, [])
        return results

    ip = _partitions(n)
    jp = _partitions(m)
    kp = _partitions(p)
    return ip, jp, kp


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full = n * m * p

    best = _schoolbook(n, m, p)
    best_rank = full

    def timed():
        return time.time() > _DEADLINE

    # Strategy 1: recursive Strassen over many partitions, keep the best valid one.
    ip, jp, kp = _generate_partitions(n, m, p)
    candidates = []
    for ig in ip:
        for jg in jp:
            for kg in kp:
                candidates.append((ig, jg, kg))

    def expected_rank(ig, jg, kg):
        tot = 0
        for a in ig:
            for b in jg:
                for c in kg:
                    tot += 7 if (len(a) == 2 and len(b) == 2 and len(c) == 2) \
                        else len(a) * len(b) * len(c)
        return tot
    candidates.sort(key=lambda t: expected_rank(t[0], t[1], t[2]))

    for (ig, jg, kg) in candidates:
        if timed():
            break
        cand = _build_recursive_strassen(n, m, p, ig, jg, kg)
        if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r
            U, V, W = _try_reduce_rank(cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=50)
            if _verify_fast(U, V, W, n, m, p):
                new_rank = len(U[0])
                if new_rank < best_rank:
                    best = {"U": U, "V": V, "W": W}
                    best_rank = new_rank

    # Strategy 2: Try non-contiguous partitions to find better Strassen block alignments
    if not timed():
        # For 3x5x6, try specific hand-crafted partitions that maximize 2x2x2 blocks
        # i=3: can't make all 2s, so one 2-block and one 1-block
        # j=5: two 2-blocks and one 1-block
        # k=6: three 2-blocks
        # Best alignment: pair up as many 2x2x2 blocks as possible
        
        # Try: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5]
        # This gives 2x2x2 blocks for (i0,j0,k0), (i0,j1,k1), (i0,j1,k2), (i0,j0,k1), (i0,j0,k2)
        # Wait, let me be more systematic.
        
        special_partitions = [
            # i groups, j groups, k groups
            ([[0,1],[2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
            ([[0,1],[2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
            ([[0],[1,2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
            ([[0,1],[2]], [[0],[1,2],[3,4]], [[0,1],[2,3],[4,5]]),
            ([[0,1],[2]], [[0,1],[2],[3,4]], [[0,1],[2,3],[4,5]]),
            ([[0],[1,2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
            ([[0,1],[2]], [[0,1],[2,3],[4]], [[0],[1,2],[3,4],[5]]),
            ([[0,1],[2]], [[0],[1,2],[3,4]], [[0],[1,2],[3,4],[5]]),
        ]
        
        for ig, jg, kg in special_partitions:
            if timed():
                break
            cand = _build_recursive_strassen(n, m, p, ig, jg, kg)
            if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
                r = len(cand["U"][0])
                if r < best_rank:
                    best = cand
                    best_rank = r
                U, V, W = _try_reduce_rank(cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=50)
                if _verify_fast(U, V, W, n, m, p):
                    new_rank = len(U[0])
                    if new_rank < best_rank:
                        best = {"U": U, "V": V, "W": W}
                        best_rank = new_rank

    # Strategy 3: random perturbation + reduction on the best found so far.
    if not timed():
        rng = random.Random(42)
        for target_rank in range(best_rank - 1, max(20, best_rank - 15), -1):
            if timed():
                break
            for attempt in range(20):
                if timed():
                    break
                U = np.array(best["U"], dtype=float)[:, :target_rank].copy()
                V = np.array(best["V"], dtype=float)[:, :target_rank].copy()
                W = np.array(best["W"], dtype=float)[:, :target_rank].copy()
                for _ in range(10):
                    a = rng.randint(0, nm - 1)
                    b = rng.randint(0, mp - 1)
                    c = rng.randint(0, pn - 1)
                    r1 = rng.randint(0, target_rank - 1)
                    r2 = rng.randint(0, target_rank - 1)
                    U[a, r1] += rng.choice([-1, 1]) * 0.5
                    V[b, r2] += rng.choice([-1, 1]) * 0.5
                    W[c, r1] += rng.choice([-1, 1]) * 0.5
                U_red, V_red, W_red = _try_reduce_rank(U.tolist(), V.tolist(), W.tolist(),
                                                       n, m, p, max_iterations=20)
                if _verify_fast(U_red, V_red, W_red, n, m, p):
                    new_rank = len(U_red[0])
                    if new_rank < best_rank:
                        best = {"U": U_red, "V": V_red, "W": W_red}
                        best_rank = new_rank
                    break

    # Final verification; fall back to schoolbook if anything went wrong.
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)

    return best
# EVOLVE-BLOCK-END
