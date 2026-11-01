"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    Uses ALS with exact linear solves for each factor, combined with multiple
    random initializations and a final rounding/polish pass.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    
    # Schoolbook fallback (rank 12)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    deadline = time.time() + 130  # Leave 20s margin
    
    # Try ranks from 6 upward (lower bound is max(n*m, m*p, n*p) = 6)
    for r in range(6, rank_sb):
        if time.time() > deadline:
            break
        
        result = _find_decomposition_als(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found minimal rank
    
    return best_result


def _compute_loss(n, m, p, U, V, W):
    """Compute the loss sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2."""
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                loss += (val - 1.0) ** 2
    return loss


def _als_solve_U(n, m, p, r, V, W, max_iter=200, tol=1e-12):
    """
    Solve for U given V and W using least squares.
    For each (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k.
    This is a linear system in U[i*m+j,:].
    """
    U = np.zeros((n * m, r))
    
    for i in range(n):
        for j in range(m):
            # Build the linear system: A @ U[i*m+j,:] = b
            # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
            # b[k] = 1
            A = np.zeros((p, r))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b = np.ones(p)
            
            # Solve least squares
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j, :] = sol
            except np.linalg.LinAlgError:
                U[i * m + j, :] = 0
    
    return U


def _als_solve_V(n, m, p, r, U, W, max_iter=200, tol=1e-12):
    """
    Solve for V given U and W using least squares.
    For each (j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i.
    This is a linear system in V[j*p+k,:].
    """
    V = np.zeros((m * p, r))
    
    for j in range(m):
        for k in range(p):
            # Build the linear system: A @ V[j*p+k,:] = b
            # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
            # b[i] = 1
            A = np.zeros((n, r))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b = np.ones(n)
            
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k, :] = sol
            except np.linalg.LinAlgError:
                V[j * p + k, :] = 0
    
    return V


def _als_solve_W(n, m, p, r, U, V, max_iter=200, tol=1e-12):
    """
    Solve for W given U and V using least squares.
    For each (k,i): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j.
    This is a linear system in W[k*n+i,:].
    """
    W = np.zeros((p * n, r))
    
    for k in range(p):
        for i in range(n):
            # Build the linear system: A @ W[k*n+i,:] = b
            # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
            # b[j] = 1
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = sol
            except np.linalg.LinAlgError:
                W[k * n + i, :] = 0
    
    return W


def _round_to_half_integers(U, V, W):
    """Round to nearest half-integer and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r


def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Verify that the decomposition exactly reproduces the tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _polish(n, m, p, r, U, V, W, deadline):
    """
    High-precision polish: do more ALS iterations with tighter tolerance,
    then try rounding to half-integers.
    """
    for it in range(500):
        if time.time() > deadline:
            break
        
        U = _als_solve_U(n, m, p, r, V, W)
        V = _als_solve_V(n, m, p, r, U, W)
        W = _als_solve_W(n, m, p, r, U, V)
        
        loss = _compute_loss(n, m, p, U, V, W)
        if loss < 1e-14:
            # Try rounding
            U_r, V_r, W_r = _round_to_half_integers(U, V, W)
            if _verify_decomposition(n, m, p, U_r, V_r, W_r, tol=1e-10):
                return U_r, V_r, W_r
    
    # Try rounding even if not perfectly converged
    U_r, V_r, W_r = _round_to_half_integers(U, V, W)
    if _verify_decomposition(n, m, p, U_r, V_r, W_r, tol=1e-8):
        return U_r, V_r, W_r
    
    return None


def _find_decomposition_als(n, m, p, r, deadline):
    """
    Try to find a rank-r decomposition using ALS with exact linear solves.
    Multiple random initializations.
    """
    np.random.seed(42)
    
    num_attempts = 200
    for attempt in range(num_attempts):
        if time.time() > deadline:
            return None
        
        # Random initialization with varying scales
        scale = 0.5 + np.random.random() * 1.0
        U = np.random.randn(n * m, r) * scale
        V = np.random.randn(m * p, r) * scale
        W = np.random.randn(p * n, r) * scale
        
        # Run ALS
        converged = False
        for it in range(300):
            if time.time() > deadline:
                break
            
            U = _als_solve_U(n, m, p, r, V, W)
            V = _als_solve_V(n, m, p, r, U, W)
            W = _als_solve_W(n, m, p, r, U, V)
            
            loss = _compute_loss(n, m, p, U, V, W)
            if loss < 1e-10:
                converged = True
                break
        
        if converged:
            # Polish and round
            result = _polish(n, m, p, r, U, V, W, deadline)
            if result is not None:
                U_r, V_r, W_r = result
                U_list = [[float(U_r[i, c]) for c in range(r)] for i in range(n * m)]
                V_list = [[float(V_r[i, c]) for c in range(r)] for i in range(m * p)]
                W_list = [[float(W_r[i, c]) for c in range(r)] for i in range(p * n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
