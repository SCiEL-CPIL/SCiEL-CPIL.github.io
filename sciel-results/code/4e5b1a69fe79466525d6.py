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


def _precompute_target_slices(target):
    """Precompute row-slices of target for faster gradient computation.
    target shape: (n*m, m*p, p*n)
    Returns dict with precomputed 2D slices for efficient contraction."""
    nU, nV, nW = target.shape
    # target_u[k, j] = sum_i target[i, k, j] -- not directly useful
    # Instead, for gradient of U[:,r]: tu[i] = sum_{k,j} target[i,k,j]*V[k]*W[j]
    # For gradient of V[:,r]: tv[k] = sum_{i,j} target[i,k,j]*U[i]*W[j]
    # For gradient of W[:,r]: tw[j] = sum_{i,k} target[i,k,j]*U[i]*V[k]
    # We'll compute these via tensordot per term but precompute target as contiguous
    return target


def _grad_step(U, V, W, target, lr):
    """One gradient-descent step minimising sum_r (U_r⊗V_r⊗W_r - T)^2.
    Uses per-term dot products so cost is O(rank * dim), not O(rank * tensor).
    Returns (new_U, new_V, new_W, loss)."""
    R = U.shape[1]
    du = np.zeros((U.shape[0], R))
    dv = np.zeros((V.shape[0], R))
    dw = np.zeros((W.shape[0], R))
    loss = 0.0
    target_sq = float(np.sum(target * target))
    for r in range(R):
        ur = U[:, r]
        vr = V[:, r]
        wr = W[:, r]
        # tu[i] = sum_{k,j} target[i,k,j] * vr[k] * wr[j]
        # = (target * (vr[:,None,None] * wr[None,:,None])) summed over axes 1,2
        # More efficient: contract target with wr over axis 2, then with vr over axis 1
        tmp = np.tensordot(target, wr, axes=([2], [0]))  # shape (nU, nV)
        tu = np.tensordot(tmp, vr, axes=([1], [0]))       # shape (nU,)
        # tv[k] = sum_{i,j} target[i,k,j] * ur[i] * wr[j]
        tmp2 = np.tensordot(target, ur, axes=([0], [0]))  # shape (nV, nW)
        tv = np.tensordot(tmp2, wr, axes=([1], [0]))      # shape (nV,)
        # tw[j] = sum_{i,k} target[i,k,j] * ur[i] * vr[k]
        tmp3 = np.tensordot(target, ur, axes=([0], [0]))  # shape (nV, nW)
        tw = np.tensordot(tmp3, vr, axes=([0], [0]))      # shape (nW,)
        
        du[:, r] = 2.0 * (vr @ wr - tu)
        dv[:, r] = 2.0 * (ur @ wr - tv)
        dw[:, r] = 2.0 * (ur @ vr - tw)
        loss += float(np.dot(ur, tu)) + float(np.dot(vr, tv)) + float(np.dot(wr, tw))
    loss = loss - target_sq
    U -= lr * du
    V -= lr * dv
    W -= lr * dw
    return U, V, W, loss


def _polish(U, V, W, target, n, m, p, max_time, start_time, lr_init=1e-4, tol=1e-10):
    """High-precision polish: small LR, tight tolerance. Returns (U,V,W) if verified."""
    rank = U.shape[1]
    lr = lr_init
    for it in range(500):
        if time.time() - start_time > max_time:
            break
        U, V, W, loss = _grad_step(U, V, W, target, lr)
        if loss < 1e-12:
            break
        if it % 50 == 49:
            lr *= 0.8
        if loss < 1e-8:
            if _verify_decomposition(U, V, W, n, m, p, tol=tol):
                return U, V, W
    if _verify_decomposition(U, V, W, n, m, p, tol=tol):
        return U, V, W
    return None


def _try_rank27_random(n, m, p, target, max_time, seed=0):
    """Direct random rank-27 search: random init + gradient descent + polish.
    Returns (U,V,W) if a verified rank-27 is found, else None."""
    rng = np.random.RandomState(seed)
    rank = 27
    nU, nV, nW = n*m, m*p, p*n
    start = time.time()
    n_iter = 2000
    best_loss = float('inf')
    best_UVW = None

    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() - start > max_time - 5:
            break
        if restart == 0:
            # near-rank-28 seed: take Strassen, zero one term, keep 27 terms
            U, V, W = _build_strassen_rank28()
            drop = rng.randint(0, 28)
            mask = np.arange(28) != drop
            U = U.copy()[:, mask]
            V = V.copy()[:, mask]
            W = W.copy()[:, mask]
            lr = 1e-3
        elif restart == 1:
            # Perturbed near-rank-28: zero one term, add small noise
            U, V, W = _build_strassen_rank28()
            drop = rng.randint(0, 28)
            mask = np.arange(28) != drop
            U = U.copy()[:, mask] + rng.randn(nU, rank) * 0.01
            V = V.copy()[:, mask] + rng.randn(nV, rank) * 0.01
            W = W.copy()[:, mask] + rng.randn(nW, rank) * 0.01
            lr = 1e-3
        else:
            U = rng.randn(nU, rank) * 0.5
            V = rng.randn(nV, rank) * 0.5
            W = rng.randn(nW, rank) * 0.5
            lr = 5e-3

        prev_loss = float('inf')
        for it in range(n_iter):
            if time.time() - start > max_time - 5:
                break
            U, V, W, loss = _grad_step(U, V, W, target, lr)
            if loss > 1e8:
                lr *= 0.1
                continue
            # Adaptive LR
            if it > 0 and loss > prev_loss * 1.1:
                lr *= 0.5
            elif it % 100 == 99 and loss > 1e-6:
                lr *= 0.8
            prev_loss = loss

            if loss < 1e-5:
                # Try polish
                polished = _polish(U.copy(), V.copy(), W.copy(), target, n, m, p,
                                   max_time=max_time, start_time=start, lr_init=1e-4, tol=1e-8)
                if polished is not None:
                    return polished
                if loss < best_loss:
                    best_loss = loss
                    best_UVW = (U.copy(), V.copy(), W.copy())
                break

        # Track best
        if loss < best_loss:
            best_loss = loss
            best_UVW = (U.copy(), V.copy(), W.copy())

        # Final check
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return U, V, W

    # If we have a good candidate, try one more polish
    if best_UVW is not None and best_loss < 1e-3:
        polished = _polish(best_UVW[0].copy(), best_UVW[1].copy(), best_UVW[2].copy(),
                           target, n, m, p, max_time=max_time, start_time=start,
                           lr_init=1e-5, tol=1e-8)
        if polished is not None:
            return polished

    return None


def construct():
    """Return a rank-minimal exact decomposition of the 2x4 * 4x4 matmul tensor.
    Guaranteed rank-28 fallback; searches for a verified rank-27 via a direct
    random + gradient-descent search with polish pass.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # hard limit with safety margin (evaluator kills at 150s)

    # Guaranteed baseline
    U, V, W = _build_strassen_rank28()

    # Search for a verified rank-27
    search_time = min(90, deadline - time.time())
    if search_time > 10:
        try:
            target = _build_target_tensor(n, m, p)
            best = _try_rank27_random(n, m, p, target, max_time=search_time, seed=12345)
            if best is not None:
                U, V, W = best
        except Exception:
            U, V, W = _build_strassen_rank28()

    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        U, V, W = _build_strassen_rank28()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
