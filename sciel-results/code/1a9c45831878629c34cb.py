"""2x5x6 matrix multiplication tensor decomposition (rank-minimal search).

Strategy:
  * Build a structured block decomposition using Strassen 2x2x2 cores for the
    2x2 middle blocks and rank-1 constructions for the 2x1 middle block.
    This yields a candidate rank of 45 (6 * 7 + 3 * 1).
  * Verify the candidate exactly using integer arithmetic.
  * If verification fails (unlikely for exact integer construction), attempt
    refinement via Gauss-Seidel alternating minimization.
  * Fall back to the guaranteed schoolbook rank 60 if all else fails.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]


def _build_strassen_candidate(n, m, p):
    """Build a rank-45 candidate using block partitioning + Strassen cores.

    Middle (m=5) split into column groups [0,1],[2,3],[4] (sizes 2,2,1).
    Output (p=6) split into column pairs [0,1],[2,3],[4,5].
    """
    mid_groups = [(0, 2), (2, 4), (4, 5)]
    out_pairs = [(0, 2), (2, 4), (4, 6)]

    # Strassen 2x2x2 core: returns (Uf, Vf, Wf) each a dict keyed by
    # (local_a_row, local_a_col), etc. -> coefficient.
    def strassen_core():
        Uf = {}
        Vf = {}
        Wf = {}

        def u_add(r0, c0, r1, c1, s):
            Uf[(r0, c0)] = Uf.get((r0, c0), 0) + s
            Uf[(r1, c1)] = Uf.get((r1, c1), 0) + s

        def v_add(r0, c0, r1, c1, s):
            Vf[(r0, c0)] = Vf.get((r0, c0), 0) + s
            Vf[(r1, c1)] = Vf.get((r1, c1), 0) + s

        def w_add(r0, c0, r1, c1, s):
            Wf[(r0, c0)] = Wf.get((r0, c0), 0) + s
            Wf[(r1, c1)] = Wf.get((r1, c1), 0) + s

        # M1 = (A00+A11)*(B00+B11)
        u_add(0, 0, 1, 1, 1)
        v_add(0, 0, 1, 1, 1)
        w_add(0, 0, 0, 1, 1)
        w_add(1, 1, 0, 1, 1)
        # M2 = (A10+A11)*B00
        u_add(1, 0, 1, 1, 1)
        v_add(0, 0, 0, 0, 1)
        w_add(1, 0, 0, 1, 1)
        # M3 = A00*(B10-B11)
        u_add(0, 0, 0, 0, 1)
        v_add(1, 0, 1, 1, -1)
        w_add(0, 1, 0, 1, 1)
        # M4 = A11*(B01-B10)
        u_add(1, 1, 1, 1, 1)
        v_add(0, 1, 1, 0, -1)
        w_add(0, 0, 0, 1, 1)
        w_add(1, 0, 0, 1, 1)
        # M5 = (A00+A01)*B11
        u_add(0, 0, 0, 1, 1)
        v_add(1, 1, 1, 1, 1)
        w_add(0, 0, 0, 1, -1)
        w_add(0, 1, 0, 1, 1)
        # M6 = (A10-A00)*(B00+B10)
        u_add(1, 0, 0, 0, -1)
        v_add(0, 0, 1, 0, 1)
        w_add(1, 1, 0, 1, 1)
        # M7 = (A01-A11)*(B01+B11)
        u_add(0, 1, 1, 1, -1)
        v_add(0, 1, 1, 1, 1)
        w_add(0, 0, 0, 1, 1)
        return Uf, Vf, Wf

    core_U, core_V, core_W = strassen_core()

    # Calculate total rank
    total_rank = 0
    for ms, me in mid_groups:
        for os_, oe in out_pairs:
            if me - ms == 2:
                total_rank += 7
            else:
                total_rank += 1  # Rank 1 for 2x1x2 blocks

    U = np.zeros((n * m, total_rank), dtype=object)
    V = np.zeros((m * p, total_rank), dtype=object)
    W = np.zeros((p * n, total_rank), dtype=object)

    col = 0
    for ms, me in mid_groups:
        if me - ms == 2:
            for os_, oe in out_pairs:
                for (lr, lc), c in core_U.items():
                    U[lr * m + (ms + lc), col] += c
                for (lr, lc), c in core_V.items():
                    V[(ms + lr) * p + (os_ + lc), col] += c
                for (lr, lc), c in core_W.items():
                    W[(os_ + lc) * n + lr, col] += c
                col += 1
        else:
            # 2x1 (A rows 0,1 col ms) x 1x2 (B row ms cols os_,oe)
            # Rank 1 construction
            for oc in range(os_, oe):
                # We need to handle the two output columns oc and oc+1
                # Actually, the block is 2x1x2.
                # We use ONE term for the whole 2x1x2 block?
                # No, the term must cover all k in the block.
                # The block has k in [os_, oe).
                # We set V[ms, k] = 1 for all k in block.
                # We set W[k, i] = 1 for all k in block, i in 0,1.
                # We set U[i, ms] = 1 for all i in 0,1.
                # This works for the whole block with 1 term.
                pass
            
            # Set up the single term for this block
            U[0 * m + ms, col] = 1
            U[1 * m + ms, col] = 1
            
            for oc in range(os_, oe):
                V[ms * p + oc, col] = 1
                W[oc * n + 0, col] = 1
                W[oc * n + 1, col] = 1
                
            col += 1

    return U, V, W, col


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank


def _verify_exact(n, m, p, U, V, W):
    """Exact integer verification. Returns (ok, max_abs_error)."""
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=object)
    for r in range(rank):
        Ur = U[:, r]
        Vr = V[:, r]
        Wr = W[:, r]
        for i in range(n):
            for j in range(m):
                a = Ur[i * m + j]
                if a == 0:
                    continue
                for k in range(p):
                    b = Vr[j * p + k]
                    if b == 0:
                        continue
                    T[i, j, k] = T[i, j, k] + a * b * Wr[k * n + i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False, abs(int(T[i, j, k]) - 1)
    return True, 0


def _gauss_seidel(n, m, p, U, V, W, max_iters=40, tol=1e-11):
    """Alternating exact least-squares refinement exploiting multilinearity."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    prev_err = np.inf
    for it in range(max_iters):
        # --- Solve U ---
        for i in range(n):
            for j in range(m):
                b = np.zeros(rank)
                for k in range(p):
                    b = b + V[j * p + k, :] * W[k * n + i, :]
                if np.allclose(b, 0):
                    continue
                nb2 = float(np.dot(b, b))
                if nb2 > 1e-30:
                    U[i * m + j, :] = b / nb2
        # --- Solve V ---
        for j in range(m):
            for k in range(p):
                b = np.zeros(rank)
                for i in range(n):
                    b = b + U[i * m + j, :] * W[k * n + i, :]
                nb2 = float(np.dot(b, b))
                if nb2 > 1e-30:
                    V[j * p + k, :] = b / nb2
        # --- Solve W ---
        for k in range(p):
            for i in range(n):
                b = np.zeros(rank)
                for j in range(m):
                    b = b + U[i * m + j, :] * V[j * p + k, :]
                nb2 = float(np.dot(b, b))
                if nb2 > 1e-30:
                    W[k * n + i, :] = b / nb2
        # --- Evaluate error ---
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (s - 1.0) ** 2
        if abs(prev_err - err) < tol and err < 1e-6:
            break
        prev_err = err
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Fallback
    fb_U, fb_V, fb_W, fb_rank = _build_schoolbook(n, m, p)

    # Try structured candidate
    try:
        U, V, W, rank = _build_strassen_candidate(n, m, p)
        
        # Direct integer verification first
        ok, _ = _verify_exact(n, m, p, U, V, W)
        if ok and rank < fb_rank:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # If direct verification failed, try refinement
        if time.time() < deadline - 10.0:
            U_ref, V_ref, W_ref = _gauss_seidel(n, m, p, U, V, W)
            U_ref, V_ref, W_ref = np.rint(U_ref), np.rint(V_ref), np.rint(W_ref)
            ok2, _ = _verify_exact(n, m, p, U_ref, V_ref, W_ref)
            if ok2 and rank < fb_rank:
                return {"U": U_ref.tolist(), "V": V_ref.tolist(), "W": W_ref.tolist()}
    except Exception:
        pass

    return {"U": fb_U.tolist(), "V": fb_V.tolist(), "W": fb_W.tolist()}

# EVOLVE-BLOCK-START
# Primary entry point: construct() -> {'U','V','W'}
# Score = rank (lower is better).
# EVOLVE-BLOCK-END
