"""Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition.

Combines the schoolbook rank-60 baseline with an alternating-minimization
refinement: fix two factor matrices, solve for the third via a linear least-
squares subproblem, iterate, and verify exactness on random matrices.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _solve_W(U, V, n, m, p):
    """For fixed U, V, find W minimizing the tensor error (linear LS)."""
    W = np.zeros((p * n, U.shape[1]))
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, U.shape[1]))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            W[k * n + i, :] = sol
    return W


def _solve_U(W, V, n, m, p):
    """For fixed W, V, find U minimizing the tensor error (linear LS)."""
    U = np.zeros((n * m, W.shape[1]))
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, W.shape[1]))
            for k in range(p):
                A[k, :] = W[k * n + i, :] * V[j * p + k, :]
            b = np.ones(p)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            U[i * m + j, :] = sol
    return U


def _solve_V(U, W, n, m, p):
    """For fixed U, W, find V minimizing the tensor error (linear LS)."""
    V = np.zeros((m * p, U.shape[1]))
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, U.shape[1]))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b = np.ones(n)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            V[j * p + k, :] = sol
    return V


def _verify(U, V, W, n, m, p, ntests=50, tol=1e-9):
    """Check exactness: for random A, B, does C = A@B equal the tensor contraction?"""
    rng = np.random.default_rng(12345)
    for _ in range(ntests):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # Build via einsum over the flattened factors
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                C_calc[i, k] = s
        if np.max(np.abs(C_calc - C_true)) > tol:
            return False
    return True


def _normalize_columns(M):
    """Normalize columns of M to unit norm to prevent scale drift."""
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms

def _try_rank(r, n, m, p, deadline, max_iter=500, n_trials=30):
    """Attempt to find an exact rank-r decomposition via alternating LS."""
    best = None
    best_err = float('inf')
    
    # Pre-compute indices for faster error calculation
    idx_U = [(i, j) for i in range(n) for j in range(m)]
    idx_V = [(j, k) for j in range(m) for k in range(p)]
    idx_W = [(k, i) for k in range(p) for i in range(n)]
    # Mapping from (i,j,k) to indices in U, V, W
    # We need to compute sum over r of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is equivalent to summing over c.
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        rng = np.random.default_rng(1000 + trial)
        
        # Choose initialization strategy
        # 0: Gaussian, 1: Sparse, 2: Signed Sparse
        strat = trial % 3
        if strat == 0:
            U = rng.standard_normal((n * m, r))
            V = rng.standard_normal((m * p, r))
        elif strat == 1:
            U = (rng.random((n * m, r)) > 0.7) * (rng.choice([-1, 1], size=(n * m, r)))
            V = (rng.random((m * p, r)) > 0.7) * (rng.choice([-1, 1], size=(m * p, r)))
        else:
            U = rng.choice([-1, 0, 1], size=(n * m, r))
            V = rng.choice([-1, 0, 1], size=(m * p, r))
            
        # Solve for W initially
        W = _solve_W(U, V, n, m, p)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Alternating updates
            U = _solve_U(W, V, n, m, p)
            V = _solve_V(U, W, n, m, p)
            W = _solve_W(U, V, n, m, p)
            
            # Normalize to maintain conditioning
            U = _normalize_columns(U)
            V = _normalize_columns(V)
            W = _normalize_columns(W)
            
            # Compute error efficiently
            # err = sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
            # We can compute the tensor T[i,j,k] = sum_c ... and then sum squares.
            # Using einsum is faster.
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # This is tricky with einsum directly on flattened arrays without reshaping.
            # Let's stick to the loop for clarity and safety, but vectorize the dot product.
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    u_row = U[i * m + j, :]
                    for k in range(p):
                        v_row = V[j * p + k, :]
                        w_row = W[k * n + i, :]
                        val = np.dot(u_row, v_row * w_row)
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-12:
                # Found a very good candidate, verify and return immediately
                Uv, Vv, Wv = best
                if _verify(Uv, Vv, Wv, n, m, p):
                    return Uv, Vv, Wv
                break
            # If error is not decreasing, break early?
            # Simple stagnation check
            if it > 50 and err > best_err * 1.01:
                break
        # If loop finished without breaking on err < 1e-12, check best
        if best is not None and best_err < 1e-9:
            Uv, Vv, Wv = best
            if _verify(Uv, Vv, Wv, n, m, p):
                return Uv, Vv, Wv
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit

    # Baseline: schoolbook rank 60
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Theoretical lower bound: n*m + m*p + n*p - m = 6 + 30 + 20 - 3 = 53
    lower_bound = n * m + m * p + n * p - m

    # Try to improve from rank 59 down toward the lower bound
    for r in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            # Stop at the lower bound
            if r <= lower_bound:
                break

    # Convert to lists of lists of numbers (integers or floats)
    def to_list(M):
        return [[float(x) for x in row] for row in M]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
