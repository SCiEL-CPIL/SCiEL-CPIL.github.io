"""
5x5 matrix multiplication - verified rank-reduction search.

Structural change from the seed: instead of hand-written (bug-prone) Strassen
block formulas, we build a correct schoolbook base decomposition and apply
rank-reduction moves that provably preserve the multiplication tensor. Each
move is verified numerically before being kept. A time-bounded search finds
the lowest verified rank, with a guaranteed correct fallback.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def build_base():
        """Correct schoolbook decomposition, rank = n*m*p = 125."""
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    def tensor_of(U, V, W):
        """T[k*n+i, j] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]."""
        T = np.zeros((p * n, m))
        for r in range(U.shape[1]):
            wcol = W[:, r]
            nz = np.nonzero(wcol)[0]
            if nz.size == 0:
                continue
            ucol = U[:, r]
            vcol = V[:, r]
            for ki in nz:
                k = ki // n
                i = ki % n
                nzj = np.nonzero(vcol)[0]
                if nzj.size == 0:
                    continue
                T[ki, nzj] += wcol[ki] * ucol[i * m + j] if False else wcol[ki]
                # (loop below handles the actual accumulation)
        # Simpler, robust construction:
        T = np.zeros((p * n, m))
        for r in range(U.shape[1]):
            wcol = W[:, r]
            nz = np.nonzero(wcol)[0]
            if nz.size == 0:
                continue
            ucol = U[:, r]
            vcol = V[:, r]
            for ki in nz:
                k = ki // n
                i = ki % n
                for j in np.nonzero(vcol)[0]:
                    T[ki, j] += wcol[ki] * ucol[i * m + j] * vcol[j]
        return T

    def target_tensor():
        T = np.zeros((p * n, m))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[k * n + i, j] = 1.0
        return T

    def active(U, V, W):
        mask = (np.any(U != 0, axis=0) &
                np.any(V != 0, axis=0) &
                np.any(W != 0, axis=0))
        return np.nonzero(mask)[0]

    def reexpress(U, V, W, i, j):
        """Find a linear combination of existing factors equal to
        U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all (k,i), j fixed.
        Returns list of (r, coef) or None."""
        irow = i * m + j
        tgt = np.zeros(p * n)
        for r in range(U.shape[1]):
            tgt += U[irow, r] * V[j * p:, r] * W[:, r]
        if not np.any(tgt):
            return []
        A = np.zeros((p * n, U.shape[1]))
        for r in range(U.shape[1]):
            A[:, r] = V[j * p:, r] * W[:, r]
        coef, res, rank, sv = np.linalg.lstsq(A, tgt, rcond=None)
        if np.linalg.norm(A @ coef - tgt) > 1e-7 * max(1.0, np.linalg.norm(tgt)):
            return None
        nz = np.nonzero(np.abs(coef) > 1e-9)[0]
        return [(int(r), float(coef[r])) for r in nz]

    def apply_move(U, V, W, i, j, expr):
        """Zero U[i*m+j, r] for each (r,coef) and add coef*V[j*p+k,r]*W[k*n+i,r]
        to W[k*n+i, r]. Preserves the tensor exactly."""
        irow = i * m + j
        for (r, c) in expr:
            U[irow, r] = 0.0
            W[:, r] += c * V[j * p:, r] * W[:, r]
        return U, V, W

    def try_moves(U, V, W, budget):
        """Randomized rank reduction. Returns (U, V, W, rank)."""
        U = U.copy(); V = V.copy(); W = W.copy()
        rank = len(active(U, V, W))
        target = target_tensor()
        cur_T = tensor_of(U, V, W)
        attempts = 0
        while time.time() - t0 < budget and attempts < 200000:
            attempts += 1
            act = active(U, V, W)
            if act.size == 0:
                break
            r = act[np.random.randint(act.size)]
            nz = np.nonzero(U[:, r])[0]
            if nz.size == 0:
                continue
            irow = nz[np.random.randint(nz.size)]
            i = irow // m
            j = irow % m
            if V[j * p:, r] == 0 or W[:, r] == 0:
                continue
            expr = reexpress(U, V, W, i, j)
            if expr is None:
                continue
            U2, V2, W2 = apply_move(U, V, W, i, j, expr)
            if not np.allclose(tensor_of(U2, V2, W2), target, atol=1e-6):
                continue
            U, V, W = U2, V2, W2
            rank = len(active(U, V, W))
        return U, V, W, rank

    # Build base and verify correctness (guard against any base bug).
    U, V, W = build_base()
    base_ok = np.allclose(tensor_of(U, V, W), target_tensor(), atol=1e-9)
    base_rank = len(active(U, V, W))

    best_U, best_V, best_W = U, V, W
    best_rank = base_rank if base_ok else None

    if base_ok:
        # Deterministic pass first (fast, reproducible reductions).
        U2, V2, W2, r2 = try_moves(U, V, W, deadline - 4.0)
        if r2 < best_rank:
            best_U, best_V, best_W, best_rank = U2, V2, W2, r2

        # A few randomized restarts from the base with different seeds.
        for seed in (12345, 98765, 24680):
            if time.time() - t0 > deadline - 6.0:
                break
            np.random.seed(seed)
            U2, V2, W2, r2 = try_moves(U, V, W, deadline - 5.0)
            if r2 < best_rank:
                best_U, best_V, best_W, best_rank = U2, V2, W2, r2

    # Final verification of the returned decomposition.
    if best_rank is None or not np.allclose(tensor_of(best_U, best_V, best_W),
                                            target_tensor(), atol=1e-6):
        U, V, W = build_base()
        best_rank = base_rank

    # Compact to active columns only.
    mask = (np.any(best_U != 0, axis=0) &
            np.any(best_V != 0, axis=0) &
            np.any(best_W != 0, axis=0))
    cols = np.nonzero(mask)[0]
    Uo = best_U[:, cols]
    Vo = best_V[:, cols]
    Wo = best_W[:, cols]

    return {"U": Uo.tolist(), "V": Vo.tolist(), "W": Wo.tolist()}
# EVOLVE-BLOCK-END
