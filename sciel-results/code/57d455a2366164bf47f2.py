"""Matrix multiplication 3x3 x 3x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using einsum. T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Compute via einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=tol))


def _verify_exact_rational(U, V, W, n, m, p, tol=1e-9):
    """Verify with exact rational arithmetic using fractions (for small sizes)."""
    from fractions import Fraction
    Uf = [[Fraction(U[i, r]).limit_denominator(10**9) for r in range(U.shape[1])] for i in range(U.shape[0])]
    Vf = [[Fraction(V[i, r]).limit_denominator(10**9) for r in range(V.shape[1])] for i in range(V.shape[0])]
    Wf = [[Fraction(W[i, r]).limit_denominator(10**9) for r in range(W.shape[1])] for i in range(W.shape[0])]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(U.shape[1]):
                    s += Uf[i * m + j][r] * Vf[j * p + k][r] * Wf[k * n + i][r]
                if s != Fraction(1):
                    return False
    return True


def _als_solve(U, V, W, n, m, p, max_iter=200, tol=1e-12, time_limit=None, start_time=None):
    """Alternating least squares to minimize ||T_computed - 1||_F^2.
    Returns (U, V, W, residual)."""
    if time_limit is None:
        time_limit = 10.0
    if start_time is None:
        start_time = time.time()
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        # Update V: for each (j,k), solve min_v ||A v - 1|| where A[i,r] = U[i*m+j,r]*W[k*n+i,r]
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                # Solve min ||A v - 1||_2
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except np.linalg.LinAlgError:
                    return U, V, W, np.inf
        V = V_new
        
        # Update W: for each (k,i), solve min_w ||B w - 1|| where B[j,r] = U[i*m+j,r]*V[j*p+k,r]
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                B = np.zeros((m, rank))
                for j in range(m):
                    B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except np.linalg.LinAlgError:
                    return U, V, W, np.inf
        W = W_new
        
        # Update U: for each (i,j), solve min_u ||C u - 1|| where C[k,r] = V[j*p+k,r]*W[k*n+i,r]
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                C = np.zeros((p, rank))
                for k in range(p):
                    C[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    u, _, _, _ = np.linalg.lstsq(C, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except np.linalg.LinAlgError:
                    return U, V, W, np.inf
        U = U_new
    
    # Compute residual
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = float(np.linalg.norm(T - 1.0, 'fro'))
    return U, V, W, residual


def _snap_half_integer(X):
    """Snap matrix to nearest half-integer lattice."""
    return np.round(2.0 * X) / 2.0


def _polish_exact(U, V, W, n, m, p, time_limit=10.0, start_time=None):
    """Try to polish to exact half-integer solution via local linear solve."""
    if start_time is None:
        start_time = time.time()
    U = _snap_half_integer(U)
    V = _snap_half_integer(V)
    W = _snap_half_integer(W)
    
    # Try a few ALS iterations with the snapped values as start
    U, V, W, residual = _als_solve(U, V, W, n, m, p, max_iter=50, time_limit=time_limit, start_time=start_time)
    
    # Snap again and check
    U_s = _snap_half_integer(U)
    V_s = _snap_half_integer(V)
    W_s = _snap_half_integer(W)
    
    if _verify_fast(U_s, V_s, W_s, n, m, p, tol=1e-7):
        # Try exact verification
        if _verify_exact_rational(U_s, V_s, W_s, n, m, p):
            return U_s, V_s, W_s, 0.0
    return U, V, W, residual


def _structured_init_3x3(rank, n, m, seed):
    """Structured initialization exploiting 3x3 block structure."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    U /= np.sqrt(rank)
    return U


def _kron_structured_init(rank, n, m, p, seed):
    """Kronecker-structured initialization: treat as 3x3 block times 3x1 vector."""
    rng = np.random.RandomState(seed)
    # Decompose rank into rank_block * rank_vec if possible
    # For now, just use random with a bias toward structured patterns
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    # Scale
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    return U, V, W


def _try_rank(rank, n, m, p, time_limit_per_attempt=15.0, num_restarts=5):
    """Try to find a valid rank-r decomposition."""
    start_time = time.time()
    best = None
    best_residual = np.inf
    
    for attempt in range(num_restarts):
        if time.time() - start_time > time_limit_per_attempt:
            break
        seed = 42 + rank * 1000 + attempt * 7
        U, V, W = _kron_structured_init(rank, n, m, p, seed)
        U, V, W, residual = _als_solve(U, V, W, n, m, p, max_iter=100, time_limit=time_limit_per_attempt, start_time=start_time)
        
        if residual < best_residual:
            best_residual = residual
            best = (U.copy(), V.copy(), W.copy())
        
        if residual < 1e-8:
            # Try exact polish
            U_p, V_p, W_p, res_p = _polish_exact(U, V, W, n, m, p, time_limit=min(10.0, time_limit_per_attempt), start_time=start_time)
            if res_p < 1e-9:
                return U_p, V_p, W_p
            if _verify_fast(U_p, V_p, W_p, n, m, p, tol=1e-7):
                return U_p, V_p, W_p
    
    # Final check on best
    if best is not None:
        U, V, W = best
        if _verify_fast(U, V, W, n, m, p, tol=1e-7):
            return U, V, W
        # Try polish
        U_p, V_p, W_p, res_p = _polish_exact(U, V, W, n, m, p, time_limit=10.0, start_time=start_time)
        if _verify_fast(U_p, V_p, W_p, n, m, p, tol=1e-7):
            return U_p, V_p, W_p
    
    return None


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Try ranks from 24 upward
    for target_rank in range(24, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank(target_rank, n, m, p, time_limit_per_attempt=min(15.0, max(5.0, deadline - time.time() - 10)))
        
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify_fast(U, V, W, n, m, p, tol=1e-7):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best_result

# EVOLVE-BLOCK-END
