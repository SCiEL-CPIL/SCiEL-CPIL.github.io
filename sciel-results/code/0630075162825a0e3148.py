"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _make_rank17():
    """Return the known rank-17 decomposition (Strassen core + direct blocks)."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))

    terms = [
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]

    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _is_rank1_222(s, tol=1e-9):
    """
    Test whether the 2x2x2 tensor s (indices alpha,beta,gamma) has tensor rank 1.
    A 2x2x2 tensor is rank 1 iff all 2x2 minors of every matricization vanish.
    """
    # matricization 1: (alpha,beta) x gamma
    for g in range(2):
        M = s[:, :, g]
        d = M[0,0]*M[1,1] - M[0,1]*M[1,0]
        if abs(d) > tol:
            return False
    # matricization 2: (alpha,gamma) x beta
    for b in range(2):
        M = s[:, b, :]
        d = M[0,0]*M[1,1] - M[0,1]*M[1,0]
        if abs(d) > tol:
            return False
    # matricization 3: (beta,gamma) x alpha
    for a in range(2):
        M = s[a, :, :]
        d = M[0,0]*M[1,1] - M[0,1]*M[1,0]
        if abs(d) > tol:
            return False
    return True

def _try_merge_two_terms(U, V, W, T, r1, r2, tol=1e-9):
    """
    Try to merge two rank-1 terms into one.
    Returns (True, merged_U_col, merged_V_col, merged_W_col) if mergeable, else (False, None, None, None).
    """
    u1, u2 = U[:, r1], U[:, r2]
    v1, v2 = V[:, r1], V[:, r2]
    w1, w2 = W[:, r1], W[:, r2]

    # Support sets
    nz_u = np.nonzero(np.abs(u1) > tol)[0]
    nz_u = np.setdiff1d(np.union1d(nz_u, np.nonzero(np.abs(u2) > tol)[0]),
                        np.nonzero(np.abs(u1) + np.abs(u2) < tol)[0])
    # Simpler: union of nonzero supports
    sup_u = np.nonzero((np.abs(u1) > tol) | (np.abs(u2) > tol))[0]
    sup_v = np.nonzero((np.abs(v1) > tol) | (np.abs(v2) > tol))[0]
    sup_w = np.nonzero((np.abs(w1) > tol) | (np.abs(w2) > tol))[0]

    if len(sup_u) > 2 or len(sup_v) > 2 or len(sup_w) > 2:
        return False, None, None, None

    # Build the 2x2x2 sub-tensor on the support
    au = list(sup_u) + [0] * (2 - len(sup_u))
    bv = list(sup_v) + [0] * (2 - len(sup_v))
    cw = list(sup_w) + [0] * (2 - len(sup_w))

    s = np.zeros((2, 2, 2))
    for alpha in range(2):
        for beta in range(2):
            for gamma in range(2):
                s[alpha, beta, gamma] = (u1[au[alpha]] * v1[bv[beta]] * w1[cw[gamma]] +
                                         u2[au[alpha]] * v2[bv[beta]] * w2[cw[gamma]])

    if not _is_rank1_222(s, tol):
        return False, None, None, None

    # Merged term is the sum (since sum is rank-1, sum of vectors is a valid rank-1 representation)
    merged_u = u1 + u2
    merged_v = v1 + v2
    merged_w = w1 + w2

    left = np.einsum('a,b,c->abc', u1, v1, w1) + np.einsum('a,b,c->abc', u2, v2, w2)
    right = np.einsum('a,b,c->abc', merged_u, merged_v, merged_w)
    if not np.allclose(left, right, atol=tol):
        return False, None, None, None

    return True, merged_u, merged_v, merged_w

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6

    T = _build_tensor(n, m, p)

    # Start from the known rank-17 solution
    U, V, W = _make_rank17()
    rank = 17

    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook (should never happen)
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # --- Phase 1: exact pair-merge reduction (general rank-1 test) ---
    if time.time() < deadline - 10:
        current_rank = rank
        U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
        merged = True
        while merged and time.time() < deadline - 10:
            merged = False
            for r1 in range(current_rank - 1, -1, -1):
                if time.time() >= deadline - 10:
                    break
                for r2 in range(r1 - 1, -1, -1):
                    if time.time() >= deadline - 10:
                        break
                    success, mu, mv, mw = _try_merge_two_terms(U_cur, V_cur, W_cur, T, r1, r2)
                    if success:
                        U_cur[:, r1] = mu
                        V_cur[:, r1] = mv
                        W_cur[:, r1] = mw
                        if r2 < current_rank - 1:
                            U_cur[:, r2:current_rank-1] = U_cur[:, r2+1:current_rank]
                            V_cur[:, r2:current_rank-1] = V_cur[:, r2+1:current_rank]
                            W_cur[:, r2:current_rank-1] = W_cur[:, r2+1:current_rank]
                        current_rank -= 1
                        merged = True
                        if not _verify_decomposition(U_cur[:, :current_rank], V_cur[:, :current_rank], W_cur[:, :current_rank], T):
                            U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
                            current_rank = rank
                            merged = False
                            break
                        break

        if current_rank < rank:
            U, V, W = U_cur[:, :current_rank], V_cur[:, :current_rank], W_cur[:, :current_rank]
            rank = current_rank

    # --- Phase 2: local search to find a lower-rank {-1,0,1} decomposition ---
    # Objective: minimize squared error of the tensor reconstruction.
    # We fix the number of terms R and search over entries in {-1,0,1}.
    # Start R at rank-1 and go down; stop at the first R for which we find an exact solution.
    if time.time() < deadline - 5:
        for R in range(rank - 1, 10, -1):
            if time.time() >= deadline - 5:
                break
            found = _local_search_rank(R, T, deadline - 5)
            if found is not None:
                U, V, W = found
                rank = R
                break

    # Final exact verification
    if not _verify_decomposition(U, V, W, T):
        # Revert to the verified rank-17
        U, V, W = _make_rank17()
        rank = 17

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _local_search_rank(R, T, deadline):
    """
    Local search for a rank-R decomposition with entries in {-1,0,1}.
    Returns (U,V,W) if an exact solution is found, else None.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n

    def error(U, V, W):
        result = np.zeros(T.shape)
        for r in range(U.shape[1]):
            result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.sum((result - T) ** 2)

    best_err = None
    best = None

    n_restarts = 40
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        rng = random.Random(1000 + restart)

        # Initialize from rank-17, drop (17-R) columns, then perturb
        U0, V0, W0 = _make_rank17()
        # Remove 17-R random columns
        drop = sorted(rng.sample(range(17), 17 - R))
        keep = [i for i in range(17) if i not in drop]
        U = U0[:, keep].copy()
        V = V0[:, keep].copy()
        W = W0[:, keep].copy()

        # Random perturb a few entries to {-1,0,1}
        for _ in range(3):
            for mat in (U, V, W):
                for _ in range(5):
                    a = rng.randrange(mat.shape[0])
                    b = rng.randrange(mat.shape[1])
                    mat[a, b] = rng.choice([-1, 0, 1])

        cur_err = error(U, V, W)
        if cur_err < 1e-9:
            return U, V, W

        # Local search: flip one entry at a time if it reduces error
        improved = True
        iters = 0
        max_iters = 400
        while improved and time.time() < deadline and iters < max_iters:
            improved = False
            iters += 1
            # Try flipping entries in U, V, W
            for mat in (U, V, W):
                if time.time() >= deadline:
                    break
                # Pick a random subset of entries to try (to keep it fast)
                n_try = 30
                for _ in range(n_try):
                    a = rng.randrange(mat.shape[0])
                    b = rng.randrange(mat.shape[1])
                    old = mat[a, b]
                    # Try the other two values
                    for new in (-1, 0, 1):
                        if new == old:
                            continue
                        mat[a, b] = new
                        e = error(U, V, W)
                        if e < cur_err - 1e-12:
                            cur_err = e
                            old = new
                            improved = True
                            if e < 1e-9:
                                break
                        else:
                            mat[a, b] = old
                    if cur_err < 1e-9:
                        break
                if cur_err < 1e-9:
                    break
            if cur_err < 1e-9:
                break

        if cur_err < 1e-9:
            return U, V, W

        if best_err is None or cur_err < best_err:
            best_err = cur_err
            best = (U.copy(), V.copy(), W.copy())

    return None
# EVOLVE-BLOCK-END
