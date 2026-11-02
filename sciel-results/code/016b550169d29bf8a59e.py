"""Matrix multiplication 2x4 x 4x4 tensor decomposition.

Guaranteed rank-28 (Strassen on 2x2 blocks) fallback, plus a *different* rank-27
search family: direct random rank-27 optimisation with an analytical gradient,
plus a near-rank-28 (one term zeroed) seed. If rank 27 is verified it is returned
(score improves to 27); otherwise the verified rank-28 solution is returned.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """Build the rank-28 Strassen decomposition deterministically (verified)."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for b in range(2):  # B column block
        for a_blk in range(2):  # A column block / B row block
            for s in range(7):
                c = col

                U_contrib = {
                    0: [(0, 2*a_blk, 1), (1, 2*a_blk+1, 1)],
                    1: [(1, 2*a_blk, 1), (1, 2*a_blk+1, 1)],
                    2: [(0, 2*a_blk, 1)],
                    3: [(1, 2*a_blk+1, 1)],
                    4: [(0, 2*a_blk, 1), (0, 2*a_blk+1, 1)],
                    5: [(1, 2*a_blk, 1), (0, 2*a_blk, -1)],
                    6: [(0, 2*a_blk+1, 1), (1, 2*a_blk+1, -1)],
                }
                for (i, j, coeff) in U_contrib[s]:
                    U[i * 4 + j, c] += coeff

                V_contrib = {
                    0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    1: [(2*a_blk, 2*b, 1)],
                    2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                    3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                    4: [(2*a_blk+1, 2*b+1, 1)],
                    5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                    6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                }
                for (j, k, coeff) in V_contrib[s]:
                    V[j * 4 + k, c] += coeff

                W_contrib = {
                    0: {(0, 0): 1, (1, 1): 1},
                    1: {(1, 0): 1, (1, 1): -1},
                    2: {(0, 1): 1, (1, 1): 1},
                    3: {(0, 0): 1, (1, 0): 1},
                    4: {(0, 0): -1, (0, 1): 1},
                    5: {(1, 1): 1},
                    6: {(0, 0): 1},
                }
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    W[(2 * b + k_loc) * 2 + i_loc, c] += coeff

                col += 1

    return U, V, W


def _build_target_tensor(n, m, p):
    """Target matmul tensor T[i*m+k, k*p+j, j*n+i] = 1."""
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                target[i*m+k, k*p+j, j*n+i] = 1.0
    return target


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if U, V, W exactly decompose the matmul tensor."""
    rank = U.shape[1]
    target = _build_target_tensor(n, m, p)
    actual = np.zeros_like(target)
    for r in range(rank):
        actual += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(actual, target, atol=tol)


def _grad_step(U, V, W, target, lr):
    """One gradient-descent step minimising sum_r (U_r⊗V_r⊗W_r - T)^2.
    Uses per-term dot products so cost is O(rank * dim), not O(rank * tensor).
    Returns (new_U, new_V, new_W, loss)."""
    R = U.shape[1]
    # per-term dot products with target
    du = np.zeros((U.shape[0], R))
    dv = np.zeros((V.shape[0], R))
    dw = np.zeros((W.shape[0], R))
    loss = 0.0
    for r in range(R):
        ur = U[:, r]
        vr = V[:, r]
        wr = W[:, r]
        tu = np.tensordot(target, (vr, wr), axes=([1, 2], [0, 0]))
        tv = np.tensordot(target, (ur, wr), axes=([0, 2], [0, 0]))
        tw = np.tensordot(target, (ur, vr), axes=([0, 1], [0, 0]))
        du[:, r] = 2.0 * (vr @ wr - tu)
        dv[:, r] = 2.0 * (ur @ wr - tv)
        dw[:, r] = 2.0 * (ur @ vr - tw)
        loss += float(np.dot(ur, tu)) + float(np.dot(vr, tv)) + float(np.dot(wr, tw))
    loss = loss - float(np.sum(target * target))
    U -= lr * du
    V -= lr * dv
    W -= lr * dw
    return U, V, W, loss


def _try_rank27_random(n, m, p, target, max_time, seed=0):
    """Direct random rank-27 search: random init + gradient descent.
    Returns (U,V,W) if a verified rank-27 is found, else None."""
    rng = np.random.RandomState(seed)
    rank = 27
    nU, nV, nW = n*m, m*p, p*n
    start = time.time()
    n_iter = 2500

    # several random restarts; also a near-rank-28 (one term zeroed) seed
    for restart in range(12):
        if time.time() - start > max_time:
            break
        if restart == 0:
            # near-rank-28 seed: take Strassen, zero one term, keep 27 terms
            U, V, W = _build_strassen_rank28()
            drop = rng.randint(0, 28)
            U = U.copy()[:, drop != np.arange(28)]
            V = V.copy()[:, drop != np.arange(28)]
            W = W.copy()[:, drop != np.arange(28)]
            lr = 1e-3
        else:
            U = rng.randn(nU, rank)
            V = rng.randn(nV, rank)
            W = rng.randn(nW, rank)
            lr = 1e-2

        for it in range(n_iter):
            if time.time() - start > max_time:
                break
            U, V, W, loss = _grad_step(U, V, W, target, lr)
            # adapt learning rate
            if loss > 1e6:
                lr *= 0.5
            elif it % 200 == 199 and loss > 1e-6:
                lr *= 0.7
            if loss < 1e-6:
                # verify strictly before accepting
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                    return U, V, W
        # final check for this restart
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return U, V, W
    return None


def construct():
    """Return a rank-minimal exact decomposition of the 2x4 * 4x4 matmul tensor.
    Guaranteed rank-28 fallback; searches for a verified rank-27 via a direct
    random + gradient-descent search (a different family than local perturbation).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # hard limit; keep safety margin below

    # Guaranteed baseline
    U, V, W = _build_strassen_rank28()

    # Search for a verified rank-27 with a safety margin (leave ~15s for final work)
    search_time = min(95, deadline - time.time())
    if search_time > 10:
        try:
            target = _build_target_tensor(n, m, p)
            best = _try_rank27_random(n, m, p, target, max_time=search_time, seed=12345)
            if best is not None:
                U, V, W = best
        except Exception:
            # any failure: fall back to the guaranteed rank-28
            U, V, W = _build_strassen_rank28()

    # Final verification; if anything is off, reset to known-good rank-28
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        U, V, W = _build_strassen_rank28()

    # Safety: ensure we never exceed the deadline (return whatever is verified)
    if time.time() > deadline - 5:
        if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            U, V, W = _build_strassen_rank28()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
