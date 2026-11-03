"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, atol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=atol)

def _schoolbook(n_sub, m_sub, p_sub):
    """Return schoolbook decomposition for n_sub x m_sub x p_sub."""
    rank = n_sub * m_sub * p_sub
    U = np.zeros((n_sub * m_sub, rank), dtype=np.float64)
    V = np.zeros((m_sub * p_sub, rank), dtype=np.float64)
    W = np.zeros((p_sub * n_sub, rank), dtype=np.float64)
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U[i * m_sub + j, col] = 1.0
                V[j * p_sub + k, col] = 1.0
                W[k * n_sub + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7. Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    W convention: W[k*n+i] where n=2, so W[k*2+i] for C[i,k]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _embed_sub_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_off, j_off, k_off, n, m, p):
    """Embed a sub-decomposition into the full 3x5x6 problem."""
    rank = U_sub.shape[1]
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)

    for idx in range(n_sub * m_sub):
        i_s = idx // m_sub
        j_s = idx % m_sub
        i_f = i_off + i_s
        j_f = j_off + j_s
        U_full[i_f * m + j_f, :] = U_sub[idx, :]

    for idx in range(m_sub * p_sub):
        j_s = idx // p_sub
        k_s = idx % p_sub
        j_f = j_off + j_s
        k_f = k_off + k_s
        V_full[j_f * p + k_f, :] = V_sub[idx, :]

    for idx in range(p_sub * n_sub):
        k_s = idx // n_sub
        i_s = idx % n_sub
        k_f = k_off + k_s
        i_f = i_off + i_s
        W_full[k_f * n + i_f, :] = W_sub[idx, :]

    return U_full, V_full, W_full

def _build_block_decomp(n, m, p, n_splits, m_splits, p_splits):
    """Build a block decomposition for arbitrary splits."""
    U_total = np.zeros((n * m, 0), dtype=np.float64)
    V_total = np.zeros((m * p, 0), dtype=np.float64)
    W_total = np.zeros((p * n, 0), dtype=np.float64)

    i_off = 0
    for i_s, n_sub in enumerate(n_splits):
        j_off = 0
        for j_s, m_sub in enumerate(m_splits):
            k_off = 0
            for k_s, p_sub in enumerate(p_splits):
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    U_sub, V_sub, W_sub = _strassen_222()
                else:
                    U_sub, V_sub, W_sub = _schoolbook(n_sub, m_sub, p_sub)

                U_full, V_full, W_full = _embed_sub_decomp(
                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                    i_off, j_off, k_off, n, m, p
                )

                if U_total.shape[1] == 0:
                    U_total = U_full
                    V_total = V_full
                    W_total = W_full
                else:
                    U_total = np.hstack([U_total, U_full])
                    V_total = np.hstack([V_total, V_full])
                    W_total = np.hstack([W_total, W_full])

                k_off += p_sub
            j_off += m_sub
        i_off += n_sub

    return U_total, V_total, W_total

def _block_decomposition_356():
    """Construct a block decomposition for 3x5x6 with rank 84."""
    n, m, p = 3, 5, 6
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    return _build_block_decomp(n, m, p, n_splits, m_splits, p_splits)

def _als_step(T, U, V, W, iters=6, lam=1e-3):
    """
    Alternating least squares to reduce rank while keeping the
    reconstruction error small. T is (a, b, c). U (a,r), V (b,r), W (c,r).
    Returns (U, V, W, rank).
    """
    a, b, c = T.shape
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    for _ in range(iters):
        # Fix V, W; solve for U: T = sum_k U[:,k] outer V[:,k] outer W[:,k]
        # Reshape: T_mat = T.transpose(0,1,2).reshape(a, b*c)
        # U_new = T_mat @ (V * W).T  where (V*W) is (b*c, r)
        VW = (V * W.T).reshape(b * c, r)
        T_mat = T.reshape(a, b * c)
        # Regularized least squares
        A = VW
        # Solve U @ A.T = T_mat  => U = T_mat @ (A.T @ A + lam I)^-1 @ A.T
        AtA = A.T @ A + lam * np.eye(r)
        U = T_mat @ np.linalg.solve(AtA, A.T)

        # Fix U, W; solve for V
        UW = (U * W.T).reshape(a * c, r)
        T_mat2 = T.transpose(0, 2, 1).reshape(a * c, b)
        AtA = UW.T @ UW + lam * np.eye(r)
        V = T_mat2 @ np.linalg.solve(AtA, UW.T)

        # Fix U, V; solve for W
        UV = (U * V.T).reshape(a * b, r)
        T_mat3 = T.transpose(0, 1, 2).reshape(a * b, c)
        AtA = UV.T @ UV + lam * np.eye(r)
        W = T_mat3 @ np.linalg.solve(AtA, UV.T)

    # Try to drop near-zero rank components
    # Compute per-component contribution
    norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
    order = np.argsort(norms)
    for new_r in range(r - 1, 0, -1):
        keep = order[:new_r]
        U_r = U[:, keep]
        V_r = V[:, keep]
        W_r = W[:, keep]
        T_comp = np.einsum('ak,bk,ck->abc', U_r, V_r, W_r, optimize=True)
        if np.allclose(T_comp, T, atol=1e-8):
            return U_r, V_r, W_r, new_r
    return U, V, W, r

def _try_alternative_splits(n, m, p, deadline):
    """Try different block splits to find a lower-rank decomposition."""
    best = None
    best_rank = 90

    splits_to_try = [
        ([2, 1], [2, 2, 1], [2, 2, 2]),
        ([2, 1], [1, 2, 2], [2, 2, 2]),
        ([2, 1], [2, 2, 1], [1, 2, 3]),
        ([1, 2], [2, 2, 1], [2, 2, 2]),
        ([2, 1], [2, 1, 2], [2, 2, 2]),
        ([2, 1], [2, 2, 1], [3, 3]),
        ([2, 1], [3, 2], [2, 2, 2]),
        ([1, 2], [2, 3], [2, 2, 2]),
        ([2, 1], [1, 1, 3], [2, 2, 2]),
        ([2, 1], [2, 2, 1], [2, 4]),
    ]

    for n_splits, m_splits, p_splits in splits_to_try:
        if time.time() > deadline - 10:
            break
        if sum(n_splits) != n or sum(m_splits) != m or sum(p_splits) != p:
            continue

        U_total, V_total, W_total = _build_block_decomp(n, m, p, n_splits, m_splits, p_splits)
        rank = U_total.shape[1]
        if rank < best_rank:
            if _verify_decomp(n, m, p, U_total, V_total, W_total):
                best_rank = rank
                best = (U_total.copy(), V_total.copy(), W_total.copy())

    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # 140s budget, 10s safety margin

    # Primary: block decomposition with rank 84
    U, V, W = _block_decomposition_356()

    primary_valid = _verify_decomp(n, m, p, U, V, W)
    primary_rank = U.shape[1] if primary_valid else 999

    # Try alternative splits
    if time.time() < deadline - 40:
        better = _try_alternative_splits(n, m, p, deadline)
        if better is not None:
            U_b, V_b, W_b = better
            if U_b.shape[1] < primary_rank:
                U, V, W = U_b, V_b, W_b
                primary_rank = U.shape[1]

    # ALS rank reduction polish
    if time.time() < deadline - 15:
        T = _build_target_tensor(n, m, p)
        try:
            U_als, V_als, W_als, new_rank = _als_step(T, U, V, W, iters=8, lam=1e-3)
            if new_rank < primary_rank:
                if _verify_decomp(n, m, p, U_als, V_als, W_als, atol=1e-8):
                    U, V, W = U_als, V_als, W_als
                    primary_rank = new_rank
        except Exception:
            pass

    # Final verification
    if _verify_decomp(n, m, p, U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback to schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    return {
        "U": U_s.tolist(),
        "V": V_s.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
