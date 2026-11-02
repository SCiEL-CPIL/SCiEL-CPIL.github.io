"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize, least_squares

_start_time = time.time()
DEADLINE = 143.0


def _verify_full(U, V, W, n=2, m=2, p=5, tol=1e-8):
    """Verify the full 2x2x5 tensor decomposition. Returns True if exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    target = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j][c] * V[jp * p + k][c] * W[k * n + i][c]
                    if abs(val - target) > tol:
                        return False
    return True


def _get_rank_18_solution(n=2, m=2, p=5):
    """Fallback: 2 Strassen blocks (rank 7 each) for cols 0-3, schoolbook (rank 4) for col 4."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(t_offset, k_start):
        U[0][t_offset + 0] = 1
        U[3][t_offset + 0] = 1
        U[2][t_offset + 1] = 1
        U[3][t_offset + 1] = 1
        U[0][t_offset + 2] = 1
        U[3][t_offset + 3] = 1
        U[0][t_offset + 4] = 1
        U[1][t_offset + 4] = 1
        U[2][t_offset + 5] = 1
        U[0][t_offset + 5] = -1
        U[1][t_offset + 6] = 1
        U[3][t_offset + 6] = -1

        k0 = k_start
        k1 = k_start + 1
        V[0 * p + k0][t_offset + 0] = 1
        V[1 * p + k1][t_offset + 0] = 1
        V[0 * p + k0][t_offset + 1] = 1
        V[0 * p + k1][t_offset + 2] = 1
        V[1 * p + k1][t_offset + 2] = -1
        V[1 * p + k0][t_offset + 3] = 1
        V[0 * p + k0][t_offset + 3] = -1
        V[1 * p + k1][t_offset + 4] = 1
        V[0 * p + k0][t_offset + 5] = 1
        V[0 * p + k1][t_offset + 5] = 1
        V[1 * p + k0][t_offset + 6] = 1
        V[1 * p + k1][t_offset + 6] = 1

        W[k0 * n + 0][t_offset + 0] = 1
        W[k1 * n + 1][t_offset + 0] = 1
        W[k0 * n + 1][t_offset + 1] = 1
        W[k1 * n + 1][t_offset + 1] = -1
        W[k1 * n + 0][t_offset + 2] = 1
        W[k1 * n + 1][t_offset + 2] = 1
        W[k0 * n + 0][t_offset + 3] = 1
        W[k0 * n + 1][t_offset + 3] = 1
        W[k0 * n + 0][t_offset + 4] = -1
        W[k1 * n + 0][t_offset + 4] = 1
        W[k1 * n + 1][t_offset + 5] = 1
        W[k0 * n + 0][t_offset + 6] = 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)

    t = 14
    U[0][t] = 1; V[0 * p + 4][t] = 1; W[4 * n + 0][t] = 1; t += 1
    U[1][t] = 1; V[1 * p + 4][t] = 1; W[4 * n + 0][t] = 1; t += 1
    U[2][t] = 1; V[0 * p + 4][t] = 1; W[4 * n + 1][t] = 1; t += 1
    U[3][t] = 1; V[1 * p + 4][t] = 1; W[4 * n + 1][t] = 1

    return {"U": U, "V": V, "W": W}


def _get_rank_20_solution(n=2, m=2, p=5):
    """Schoolbook for 2x2x5. Rank 20."""
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    t = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1

    return {"U": U, "V": V, "W": W}


def _build_target_tensor(n=2, m=2, p=5):
    """Build the target tensor T[i,j,k,jp] = delta(j,jp)."""
    T = np.zeros((n, m, p, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k, j] = 1.0
    return T


def _tensor_error(U, V, W, n=2, m=2, p=5):
    """Compute the error vector for the decomposition."""
    rank = U.shape[1]
    errors = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    target = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[jp * p + k, c] * W[k * n + i, c]
                    errors.append(val - target)
    return np.array(errors)


def _smooth_error(U, V, W, n=2, m=2, p=5, temp=0.01):
    """Smooth surrogate using log-sum-exp for better gradient behavior."""
    errs = _tensor_error(U, V, W, n, m, p)
    # Use soft-max of squared errors for smoothness
    sq = errs ** 2
    max_sq = np.max(sq)
    if max_sq < 1e-12:
        return 0.0
    log_sum = np.log(np.sum(np.exp(sq - max_sq))) + max_sq
    return np.exp(log_sum / len(errs)) ** (len(errs) / 2)  # geometric mean of sq errors


def _snap_to_half_int(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    result = arr.copy()
    for idx in np.ndindex(result.shape):
        v = result[idx]
        # Try snapping to nearest half-integer
        nearest = np.round(v * 2) / 2
        if abs(v - nearest) < tol:
            result[idx] = nearest
    return result


def _polish_decomposition(U, V, W, n=2, m=2, p=5, max_iters=50, tol=1e-10):
    """
    Polish a near-exact decomposition using alternating linear solves.
    Treats one factor at a time as fixed and solves for the others.
    """
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    rank = U.shape[1]
    
    T = _build_target_tensor(n, m, p)
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Solve for U: for each (i,j), U[i*m+j,:] = sum over k,jp of T[i,j,k,jp] * (V[jp*p+k,:] * W[k*n+i,:])^T
        # Actually: sum_c U[i*m+j,c] * V[jp*p+k,c] * W[k*n+i,c] = T[i,j,k,jp]
        # Fix V,W, solve for U rows
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build linear system: for each (k,jp), sum_c U[i*m+j,c] * (V[jp*p+k,:] * W[k*n+i,:])_c = T[i,j,k,jp]
                # This is: M @ U[i*m+j,:] = rhs where M[k,jp,c] = V[jp*p+k,c] * W[k*n+i,c]
                M = np.zeros((p * m, rank))
                rhs = np.zeros(p * m)
                for jp in range(m):
                    for k in range(p):
                        M[jp * p + k, :] = V[jp * p + k, :] * W[k * n + i, :]
                        rhs[jp * p + k] = T[i, j, k, jp]
                # Least squares solve
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    new_U[i * m + j, :] = sol
                except:
                    new_U[i * m + j, :] = U[i * m + j, :]
        U = new_U
        
        # Solve for V
        new_V = np.zeros_like(V)
        for jp in range(m):
            for k in range(p):
                M = np.zeros((n * m, rank))
                rhs = np.zeros(n * m)
                for i in range(n):
                    for j in range(m):
                        M[i * m + j, :] = U[i * m + j, :] * W[k * n + i, :]
                        rhs[i * m + j] = T[i, j, k, jp]
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    new_V[jp * p + k, :] = sol
                except:
                    new_V[jp * p + k, :] = V[jp * p + k, :]
        V = new_V
        
        # Solve for W
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.zeros((n * m, rank))
                rhs = np.zeros(n * m)
                for jp in range(m):
                    for j in range(m):
                        M[i * m + j, :] = U[i * m + j, :] * V[jp * p + k, :]
                        rhs[i * m + j] = T[i, j, k, jp]
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    new_W[k * n + i, :] = sol
                except:
                    new_W[k * n + i, :] = W[k * n + i, :]
        W = new_W
        
        # Check convergence
        err = np.max(np.abs(_tensor_error(U, V, W, n, m, p)))
        if err < tol:
            break
    
    return U, V, W


def _search_rank15(n=2, m=2, p=5, time_budget=60):
    """
    Search for a rank-15 decomposition using structured random initialization
    and alternating linear solves (syndrome-based approach).
    """
    rank = 15
    best_err = float('inf')
    best_sol = None
    
    T = _build_target_tensor(n, m, p)
    
    np.random.seed(42)
    
    # Try multiple random initializations
    n_attempts = 8
    for attempt in range(n_attempts):
        if time.time() > DEADLINE - 30:
            break
        
        # Structured initialization: mix of Strassen-like patterns and random
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Add some structured components
        if attempt < 2:
            # Initialize with Strassen-like structure for first 14 columns
            U[:4, :14] = 0
            V[:10, :14] = 0
            W[:10, :14] = 0
            
            # First Strassen block
            U[0, 0] = 1; U[3, 0] = 1
            U[2, 1] = 1; U[3, 1] = 1
            U[0, 2] = 1; U[3, 3] = 1
            U[0, 4] = 1; U[1, 4] = 1
            U[2, 5] = 1; U[0, 5] = -1
            U[1, 6] = 1; U[3, 6] = -1
            
            V[0, 0] = 1; V[7, 0] = 1
            V[0, 1] = 1
            V[0, 2] = 1; V[7, 2] = -1
            V[7, 3] = 1; V[0, 3] = -1
            V[7, 4] = 1
            V[0, 5] = 1; V[7, 5] = 1
            V[7, 6] = 1; V[7, 6] = 1
            
            W[0, 0] = 1; W[3, 0] = 1
            W[1, 1] = 1; W[3, 1] = -1
            W[2, 2] = 1; W[3, 2] = 1
            W[0, 3] = 1; W[1, 3] = 1
            W[0, 4] = -1; W[2, 4] = 1
            W[3, 5] = 1
            W[0, 6] = 1
            
            # Second Strassen block for columns 2-3
            U[0, 7] = 1; U[3, 7] = 1
            U[2, 8] = 1; U[3, 8] = 1
            U[0, 9] = 1; U[3, 10] = 1
            U[0, 11] = 1; U[1, 11] = 1
            U[2, 12] = 1; U[0, 12] = -1
            U[1, 13] = 1; U[3, 13] = -1
            
            V[2, 7] = 1; V[9, 7] = 1
            V[2, 8] = 1
            V[2, 9] = 1; V[9, 9] = -1
            V[9, 10] = 1; V[2, 10] = -1
            V[9, 11] = 1
            V[2, 12] = 1; V[9, 12] = 1
            V[9, 13] = 1; V[9, 13] = 1
            
            W[4, 7] = 1; W[7, 7] = 1
            W[5, 8] = 1; W[7, 8] = -1
            W[6, 9] = 1; W[7, 9] = 1
            W[4, 10] = 1; W[5, 10] = 1
            W[4, 11] = -1; W[6, 11] = 1
            W[7, 12] = 1
            W[4, 13] = 1
        
        # Add random perturbation for the last few columns
        if rank > 14:
            U[:, 14:] += np.random.randn(n * m, rank - 14) * 0.3
            V[:, 14:] += np.random.randn(m * p, rank - 14) * 0.3
            W[:, 14:] += np.random.randn(p * n, rank - 14) * 0.3
        
        # Polish using alternating linear solves
        U_p, V_p, W_p = _polish_decomposition(U, V, W, n, m, p, max_iters=30, tol=1e-8)
        
        err = np.max(np.abs(_tensor_error(U_p, V_p, W_p, n, m, p)))
        if err < best_err:
            best_err = err
            best_sol = (U_p.copy(), V_p.copy(), W_p.copy())
        
        if best_err < 1e-6:
            break
    
    if best_sol is not None and best_err < 1e-4:
        U, V, W = best_sol
        # Final polish with tighter tolerance
        U, V, W = _polish_decomposition(U, V, W, n, m, p, max_iters=50, tol=1e-12)
        err = np.max(np.abs(_tensor_error(U, V, W, n, m, p)))
        if err < 1e-5:
            # Snap to half-integers
            U = _snap_to_half_int(U, tol=1e-4)
            V = _snap_to_half_int(V, tol=1e-4)
            W = _snap_to_half_int(W, tol=1e-4)
            # Verify
            if _verify_full(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-4):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def _search_rank16(n=2, m=2, p=5, time_budget=45):
    """
    Search for a rank-16 decomposition.
    """
    rank = 16
    best_err = float('inf')
    best_sol = None
    
    T = _build_target_tensor(n, m, p)
    
    np.random.seed(123)
    
    n_attempts = 6
    for attempt in range(n_attempts):
        if time.time() > DEADLINE - 30:
            break
        
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Structured initialization for first 14 columns (Strassen)
        if attempt < 3:
            U[:4, :14] = 0
            V[:10, :14] = 0
            W[:10, :14] = 0
            
            # First Strassen block
            U[0, 0] = 1; U[3, 0] = 1
            U[2, 1] = 1; U[3, 1] = 1
            U[0, 2] = 1; U[3, 3] = 1
            U[0, 4] = 1; U[1, 4] = 1
            U[2, 5] = 1; U[0, 5] = -1
            U[1, 6] = 1; U[3, 6] = -1
            
            V[0, 0] = 1; V[7, 0] = 1
            V[0, 1] = 1
            V[0, 2] = 1; V[7, 2] = -1
            V[7, 3] = 1; V[0, 3] = -1
            V[7, 4] = 1
            V[0, 5] = 1; V[7, 5] = 1
            V[7, 6] = 1
            
            W[0, 0] = 1; W[3, 0] = 1
            W[1, 1] = 1; W[3, 1] = -1
            W[2, 2] = 1; W[3, 2] = 1
            W[0, 3] = 1; W[1, 3] = 1
            W[0, 4] = -1; W[2, 4] = 1
            W[3, 5] = 1
            W[0, 6] = 1
            
            # Second Strassen block
            U[0, 7] = 1; U[3, 7] = 1
            U[2, 8] = 1; U[3, 8] = 1
            U[0, 9] = 1; U[3, 10] = 1
            U[0, 11] = 1; U[1, 11] = 1
            U[2, 12] = 1; U[0, 12] = -1
            U[1, 13] = 1; U[3, 13] = -1
            
            V[2, 7] = 1; V[9, 7] = 1
            V[2, 8] = 1
            V[2, 9] = 1; V[9, 9] = -1
            V[9, 10] = 1; V[2, 10] = -1
            V[9, 11] = 1
            V[2, 12] = 1; V[9, 12] = 1
            V[9, 13] = 1
            
            W[4, 7] = 1; W[7, 7] = 1
            W[5, 8] = 1; W[7, 8] = -1
            W[6, 9] = 1; W[7, 9] = 1
            W[4, 10] = 1; W[5, 10] = 1
            W[4, 11] = -1; W[6, 11] = 1
            W[7, 12] = 1
            W[4, 13] = 1
        
        # Random for remaining columns
        U[:, 14:] += np.random.randn(n * m, rank - 14) * 0.3
        V[:, 14:] += np.random.randn(m * p, rank - 14) * 0.3
        W[:, 14:] += np.random.randn(p * n, rank - 14) * 0.3
        
        U_p, V_p, W_p = _polish_decomposition(U, V, W, n, m, p, max_iters=30, tol=1e-8)
        
        err = np.max(np.abs(_tensor_error(U_p, V_p, W_p, n, m, p)))
        if err < best_err:
            best_err = err
            best_sol = (U_p.copy(), V_p.copy(), W_p.copy())
        
        if best_err < 1e-6:
            break
    
    if best_sol is not None and best_err < 1e-4:
        U, V, W = best_sol
        U, V, W = _polish_decomposition(U, V, W, n, m, p, max_iters=50, tol=1e-12)
        err = np.max(np.abs(_tensor_error(U, V, W, n, m, p)))
        if err < 1e-5:
            U = _snap_to_half_int(U, tol=1e-4)
            V = _snap_to_half_int(V, tol=1e-4)
            W = _snap_to_half_int(W, tol=1e-4)
            if _verify_full(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-4):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 15 first
    sol = _search_rank15(n, m, p, time_budget=60)
    if sol is not None:
        return sol
    
    # Try rank 16
    sol = _search_rank16(n, m, p, time_budget=45)
    if sol is not None:
        return sol
    
    # Fallback to rank 18
    sol = _get_rank_18_solution(n, m, p)
    if _verify_full(sol["U"], sol["V"], sol["W"], n, m, p, tol=1e-6):
        return sol
    
    # Last resort: schoolbook
    return _get_rank_20_solution(n, m, p)

# EVOLVE-BLOCK-END
