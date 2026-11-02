"""Matrix multiplication 3,3,8 tensor rank decomposition.

Searches for a rank-minimal exact decomposition (with integer/half-integer
entries) of the 3x3 x 3x8 matrix multiplication tensor.  Uses a randomised
ALS (alternating least squares) scheme on the 3-way tensor, followed by
rounding to the half-integer lattice and a local-search polish.
"""
import time
import math
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [3, 3, 8]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 72

# ---------------------------------------------------------------------------
# Precompute the multiplication tensor T[i, j, k]
# Convention: (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
# So T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute
# to (i,k).  In the tensor decomposition:
#   T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
# but the interface uses W with index k*n+i (output index) and U with i*m+j,
# V with j*p+k.  So:
#   U[i*m + j, c] = a_{i,j}^{(c)}
#   V[j*p + k, c] = b_{j,k}^{(c)}
#   W[k*n + i, c] = w_{k,i}^{(c)}
# and we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} for each (i,j,k)
# Actually the tensor is: T[i, j, k] = 1 always? No.
#
# The matrix multiplication tensor is defined by:
#   (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
# The trilinear form is: sum_{i,j,k} (AB)_{i,k} A_{i,j} B_{j,k} C_{i,k}
# where C is the output.  The tensor T has indices (i,j,k) and:
#   T[i, j, k] = 1   (for all valid i,j,k)
# Wait, that's not right either.  Let me think again.
#
# The multiplication tensor as a 3-way tensor:
#   T[i, j, k] = 1  if the term A_{i,j} * B_{j,k} contributes to (AB)_{i,k}
# Since each term A_{i,j}*B_{j,k} contributes exactly to (AB)_{i,k}, we have
# T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
#
# But the decomposition is:
#   T[i, j, k] = sum_c  U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
#
# So we need: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
#
# This is a rank-r decomposition of the order-3 tensor T of shape (n*m, m*p, p*n)
# where T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
#
# Actually T is not all-ones in the (n*m, m*p, p*n) space.  It's a specific
# structured tensor.  Let me build it properly.
# ---------------------------------------------------------------------------

def _build_tensor():
    """Build the matrix multiplication tensor as a 3D array T[a,b,c].
    T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

_T = _build_tensor()
_U_SHAPE = (n * m, RANK_SCHOOLBOOK)
_V_SHAPE = (m * p, RANK_SCHOOLBOOK)
_W_SHAPE = (p * n, RANK_SCHOOLBOOK)


def _round_half(x):
    """Round a numpy array to the nearest half-integer."""
    return np.round(2.0 * x) / 2.0


def _validate(U, V, W):
    """Check that U,V,W with integer/half-integer entries exactly decompose T.
    Returns True if valid.
    """
    r = U.shape[1]
    # Reshape and multiply
    # T_recon[a,b,c] = sum_c U[a,c] * V[b,c] * W[c_out,c]
    # We need to be careful: W is indexed by k*n+i, which is the output index.
    # Let's just do the full contraction.
    # T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
    # But W has shape (p*n, r), and the third index of T is p*n.
    T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(T_recon, _T, atol=1e-9)


def _schoolbook():
    """Return the trivial rank-72 schoolbook decomposition."""
    rank = RANK_SCHOOLBOOK
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _als_iterate(T, rank, max_iter=200, tol=1e-10, seed=0):
    """Run ALS on the 3-way tensor T to find a rank-r decomposition.
    T has shape (I, J, K).  We seek U (I,r), V (J,r), W (K,r) such that
    T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c].
    """
    rng = np.random.RandomState(seed)
    I, J, K = T.shape
    # Initialize
    U = rng.randn(I, rank) * 0.5
    V = rng.randn(J, rank) * 0.5
    W = rng.randn(K, rank) * 0.5

    for it in range(max_iter):
        # Fix V, W, solve for U
        # T[i,j,k] = sum_c U[i,c] * (V[j,c]*W[k,c])
        # For each i: T[i,:,:] = sum_c U[i,c] * (V[:,c]*W[:,c])
        # Reshape: T_i has shape (J,K), and we need to solve
        #   T[i] (reshaped to J*K,) = U[i] @ (V*W reshaped)
        # Actually: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # Let M_c = V[:,c] * W[:,c]  (outer product, shape J x K)
        # Then T[i] = sum_c U[i,c] * M_c
        # Stack: T_i (J*K,) = U[i] @ M_stack  where M_stack is (rank, J*K)
        M_stack = np.einsum('jc,kc->cj k', V, W).reshape(rank, J * K)  # (rank, J*K)
        # T[i].reshape(J*K) = U[i] @ M_stack
        # => U[i] = T[i].reshape(J*K) @ pinv(M_stack)
        # Use NNLS or lstsq
        T_flat = T.reshape(I, J * K)  # (I, J*K)
        # U = T_flat @ M_stack^+
        try:
            U_new = T_flat @ np.linalg.pinv(M_stack)
        except Exception:
            U_new = np.linalg.lstsq(M_stack.T, T_flat.T, rcond=None)[0].T

        # Fix U, W, solve for V
        M_stack2 = np.einsum('ic,kc->ci k', U, W).reshape(rank, I * K)
        T_flat2 = T.transpose(1, 0, 2).reshape(J, I * K)
        try:
            V_new = T_flat2 @ np.linalg.pinv(M_stack2)
        except Exception:
            V_new = np.linalg.lstsq(M_stack2.T, T_flat2.T, rcond=None)[0].T

        # Fix U, V, solve for W
        M_stack3 = np.einsum('ic,jc->ci j', U, V).reshape(rank, I * J)
        T_flat3 = T.transpose(2, 0, 1).reshape(K, I * J)
        try:
            W_new = T_flat3 @ np.linalg.pinv(M_stack3)
        except Exception:
            W_new = np.linalg.lstsq(M_stack3.T, T_flat3.T, rcond=None)[0].T

        # Check convergence
        diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
        U, V, W = U_new, V_new, W_new
        if diff < tol:
            break

    return U, V, W


def _residual(U, V, W):
    T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.linalg.norm(T_recon - _T, ord='fro')


def _try_round_and_local_search(U, V, W, max_local_steps=50):
    """Round to half-integers and do local search to fix errors."""
    U_h = _round_half(U)
    V_h = _round_half(V)
    W_h = _round_half(W)
    if _validate(U_h, V_h, W_h):
        return U_h, V_h, W_h
    return None


def _local_search(U, V, W, max_steps=100):
    """Local search: flip individual half-integer entries to reduce residual."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    I, J, K = _T.shape
    best_res = _residual(U, V, W)
    for step in range(max_steps):
        improved = False
        # Try flipping entries in U
        for i in range(I):
            for c in range(r):
                old = U[i, c]
                # Try old +/- 0.5
                for delta in [-0.5, 0.5]:
                    new_val = old + delta
                    U[i, c] = new_val
                    res = _residual(U, V, W)
                    if res < best_res - 1e-12:
                        best_res = res
                        improved = True
                    else:
                        U[i, c] = old
        # Try flipping entries in V
        for j in range(J):
            for c in range(r):
                old = V[j, c]
                for delta in [-0.5, 0.5]:
                    new_val = old + delta
                    V[j, c] = new_val
                    res = _residual(U, V, W)
                    if res < best_res - 1e-12:
                        best_res = res
                        improved = True
                    else:
                        V[j, c] = old
        # Try flipping entries in W
        for k in range(K):
            for c in range(r):
                old = W[k, c]
                for delta in [-0.5, 0.5]:
                    new_val = old + delta
                    W[k, c] = new_val
                    res = _residual(U, V, W)
                    if res < best_res - 1e-12:
                        best_res = res
                        improved = True
                    else:
                        W[k, c] = old
        if not improved:
            break
    if _validate(U, V, W):
        return U, V, W
    return None


def construct():
    t_start = time.time()
    deadline = t_start + 130  # 130s guard (evaluator kills at 150s)

    # Start with schoolbook
    U0, V0, W0 = _schoolbook()
    best = (U0, V0, W0)
    best_rank = RANK_SCHOOLBOOK

    # Try to find lower-rank decompositions
    # Known lower bound for matrix multiplication rank:
    # For M_{n,m,p}, rank >= n+m+p-2 = 3+3+8-2 = 12
    # But realistic ranks for small cases are much higher.
    # Try ranks from 30 up to 71 (below schoolbook)
    # Use ALS with multiple random restarts

    ranks_to_try = []
    # We can't try all ranks from 12 to 71 in 120s.
    # Focus on a range where success is plausible.
    # For 3x3x3, known rank is 23 (vs schoolbook 27).
    # For 3x3x8, the rank is likely in the range 30-50.
    # Let's try a strategic set of ranks.

    # Actually, let's be smarter.  Try a few ranks with multiple restarts.
    candidate_ranks = [40, 45, 50, 55, 60, 65, 70]
    # Sort by likelihood (middle ranks first)
    candidate_ranks.sort(key=lambda r: abs(r - 50))

    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue

        n_restarts = 15
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            seed = hash((rank, restart)) % (2**31)
            try:
                U, V, W = _als_iterate(_T, rank, max_iter=100, tol=1e-12, seed=seed)
            except Exception:
                continue

            # Try rounding
            result = _try_round_and_local_search(U, V, W, max_local_steps=30)
            if result is not None:
                U_r, V_r, W_r = result
                if _validate(U_r, V_r, W_r):
                    best_rank = rank
                    best = (U_r, V_r, W_r)
                    break  # Found valid for this rank, try next (lower) rank
            else:
                # Try local search on the rounded version
                U_h = _round_half(U)
                V_h = _round_half(V)
                W_h = _round_half(W)
                result2 = _local_search(U_h, V_h, W_h, max_steps=40)
                if result2 is not None:
                    U_r, V_r, W_r = result2
                    if _validate(U_r, V_r, W_r):
                        best_rank = rank
                        best = (U_r, V_r, W_r)
                        break

    # Also try a few lower ranks with more restarts if time permits
    if time.time() < deadline - 30:
        for rank in [25, 30, 35]:
            if rank >= best_rank:
                continue
            for restart in range(10):
                if time.time() > deadline - 15:
                    break
                seed = hash((rank, restart + 1000)) % (2**31)
                try:
                    U, V, W = _als_iterate(_T, rank, max_iter=80, tol=1e-12, seed=seed)
                except Exception:
                    continue
                result = _try_round_and_local_search(U, V, W, max_local_steps=25)
                if result is not None:
                    U_r, V_r, W_r = result
                    if _validate(U_r, V_r, W_r):
                        best_rank = rank
                        best = (U_r, V_r, W_r)
                        break

    U_best, V_best, W_best = best
    # Convert to lists of lists
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
