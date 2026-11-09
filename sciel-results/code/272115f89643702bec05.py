"""Tensor rank minimization for 2x5 * 5x5 matrix multiplication.

Uses an alternating least-squares / gradient-descent hybrid over rank-r
decompositions, initialized from the schoolbook rank-50 factorization and
then attempting to zero out columns iteratively (pruning) while keeping
the reconstruction error near zero.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]
_START = time.time()
_DEADLINE = _START + 115.0  # leave 5 s margin for the 120 s cap


def _tensors():
    """Build the 3-mode tensor T[i, j, k] = 1 for the M_{2,5,5} tensor.
    Index conventions:
      A is 2x5, indexed (i, j)  -> flat i*5 + j   (n*m = 10)
      B is 5x5, indexed (j, k)  -> flat j*5 + k   (m*p = 25)
      C is 5x2 (trace convention W[k*n + i, r]), indexed (k, i) -> flat k*2 + i (p*n = 10)
    The tensor T[a, b, c] = 1 iff a, b, c correspond to the same (i, j, k) triple.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _residual(U, V, W, T):
    """Frobenius norm of T - sum_c U[:,c] outer V[:,c] outer W[:,c]."""
    # T approx = einsum('ac,bc,dc->abd', U, V, W)
    approx = np.einsum('ac,bc,dc->abd', U, V, W, optimize=True)
    return np.linalg.norm(approx - T)


def _verify_exact(U, V, W, T, tol=1e-7):
    approx = np.einsum('ac,bc,dc->abd', U, V, W, optimize=True)
    return np.max(np.abs(approx - T)) < tol


def _round_to_half(U, V, W):
    """Round entries to nearest multiple of 0.5 (half-integers)."""
    U = np.round(2.0 * U) / 2.0
    V = np.round(2.0 * V) / 2.0
    W = np.round(2.0 * W) / 2.0
    return U, V, W


def _try_prune(U, V, W, T, r_current, r_target, max_iters=500, tol=1e-6):
    """Attempt to reduce rank from r_current to r_target by zeroing columns
    and re-optimizing the remaining ones. Returns (U, V, W, new_r) or None."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = r_current
    cols_active = list(range(r))
    
    for _ in range(r_current - r_target):
        if time.time() > _DEADLINE:
            return None
        # Try removing each column; pick the one whose removal causes least increase in error
        best_col = None
        best_err = np.inf
        for ci in cols_active:
            # Temporarily zero out column ci
            U[:, ci] = 0
            V[:, ci] = 0
            W[:, ci] = 0
            err = _residual(U, V, W, T)
            if err < best_err:
                best_err = err
                best_col = ci
            # Restore
            # We'll re-zero at the end if this is the chosen column
        if best_col is None:
            return None
        # Zero out the best column permanently
        U[:, best_col] = 0
        V[:, best_col] = 0
        W[:, best_col] = 0
        cols_active.remove(best_col)
        r -= 1
        
        # Re-optimize the remaining active columns via a few ALS steps
        for it in range(30):
            if time.time() > _DEADLINE:
                return None
            # Update U: for each active column c, U[:,c] = (T * (V[:,c] outer W[:,c])) / |V[:,c] outer W[:,c]|^2
            for c in cols_active:
                # Contract T with V[:,c] and W[:,c] to get target for U[:,c]
                # target_a = sum_{b,c'} T[a,b,c'] * V[b,c] * W[c',c]
                target = np.einsum('abc,bc,dc->a', T, V[:, c], W[:, c])
                denom = np.dot(V[:, c], V[:, c]) * np.dot(W[:, c], W[:, c])
                if denom > 1e-12:
                    U[:, c] = target / denom
            # Update V
            for c in cols_active:
                target = np.einsum('abc,bc,dc->b', T, U[:, c], W[:, c])
                denom = np.dot(U[:, c], U[:, c]) * np.dot(W[:, c], W[:, c])
                if denom > 1e-12:
                    V[:, c] = target / denom
            # Update W
            for c in cols_active:
                target = np.einsum('abc,bc,dc->c', T, U[:, c], V[:, c])
                denom = np.dot(U[:, c], U[:, c]) * np.dot(V[:, c], V[:, c])
                if denom > 1e-12:
                    W[:, c] = target / denom
        
        if _residual(U, V, W, T) > 1.0 * tol:
            # Failed; this path won't work, but we could backtrack. For simplicity, continue.
            pass
    
    # Final check
    if _residual(U, V, W, T) < 1.0 * tol:
        # Compact to r_target columns
        U_compact = U[:, cols_active]
        V_compact = V[:, cols_active]
        W_compact = W[:, cols_active]
        return U_compact, V_compact, W_compact, r_target
    return None


def _optimize_fixed_rank(U, V, W, T, r, max_iters=3000, tol=1e-9):
    """Optimize U, V, W to minimize residual for fixed rank r using ALS + GD."""
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    
    lr = 0.1
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        # ALS step: update U
        for c in range(r):
            target = np.einsum('abc,bc,dc->a', T, V[:, c], W[:, c])
            denom = np.dot(V[:, c], V[:, c]) * np.dot(W[:, c], W[:, c])
            if denom > 1e-14:
                U[:, c] = target / denom
        # Update V
        for c in range(r):
            target = np.einsum('abc,bc,dc->b', T, U[:, c], W[:, c])
            denom = np.dot(U[:, c], U[:, c]) * np.dot(W[:, c], W[:, c])
            if denom > 1e-14:
                V[:, c] = target / denom
        # Update W
        for c in range(r):
            target = np.einsum('abc,bc,dc->c', T, U[:, c], V[:, c])
            denom = np.dot(U[:, c], U[:, c]) * np.dot(V[:, c], V[:, c])
            if denom > 1e-14:
                W[:, c] = target / denom
        
        # Gradient descent step to escape ALS plateaus
        # Compute gradient of 0.5 * ||T - sum U_c outer V_c outer W_c||^2
        approx = np.einsum('ac,bc,dc->abd', U, V, W, optimize=True)
        R = T - approx  # residual
        # dU[a,c] = -sum_{b,d} R[a,b,d] * V[b,c] * W[d,c]
        dU = -np.einsum('abd,bc,dc->ac', R, V, W, optimize=True)
        dV = -np.einsum('abd,ac,dc->bc', R, U, W, optimize=True)
        dW = -np.einsum('abd,ac,bc->dc', R, U, V, optimize=True)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        if it % 200 == 199:
            lr *= 0.95
    
    return U, V, W


def construct():
    T = _tensors()
    n, m, p = DIMENSIONS
    rank0 = n * m * p  # 50
    
    # Initialize schoolbook
    U = np.zeros((n * m, rank0))
    V = np.zeros((m * p, rank0))
    W = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Optimize the full rank-50 decomposition (should already be exact)
    U, V, W = _optimize_fixed_rank(U, V, W, T, rank0, max_iters=50)
    
    # Try to prune to lower ranks
    for target_rank in range(rank0 - 1, 29, -1):
        if time.time() > _DEADLINE:
            break
        if target_rank >= best_rank:
            continue
        result = _try_prune(U, V, W, T, best_rank, target_rank)
        if result is not None:
            U, V, W, best_rank = result
            # Round to half-integers
            U, V, W = _round_to_half(U, V, W)
            # Verify
            if _verify_exact(U, V, W, T, tol=1e-6):
                # Further polish
                U, V, W = _optimize_fixed_rank(U, V, W, T, best_rank, max_iters=200)
                U, V, W = _round_to_half(U, V, W)
                if _verify_exact(U, V, W, T, tol=1e-6):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    # Keep unrounded if exact before rounding
                    U2, V2, W2 = _optimize_fixed_rank(U, V, W, T, best_rank, max_iters=100)
                    if _verify_exact(U2, V2, W2, T, tol=1e-8):
                        best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
                    else:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                # Try unrounded
                U2, V2, W2 = _optimize_fixed_rank(U, V, W, T, best_rank, max_iters=200)
                if _verify_exact(U2, V2, W2, T, tol=1e-8):
                    best_rank = best_rank
                    best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
    
    # Final attempt: if we have a valid decomposition, return it
    # Convert to lists of floats (the evaluator accepts integer/half-integer entries)
    def to_list(arr):
        return [[float(arr[i, j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
