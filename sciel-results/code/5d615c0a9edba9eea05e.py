"""Combined program: structural baseline + constrained optimization attempt for 3x4x5 matmul tensor rank."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x5 matrix. Uses schoolbook (rank 60) as the guaranteed baseline.
    Attempts to find lower-rank decompositions via constrained least-squares
    optimization with integer/half-integer rounding and verification.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5-second safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 4, 5

    # Build target tensor: T[i,j,k] = 1 for all valid indices
    T = np.ones((n, m, p), dtype=float)

    # --- Schoolbook decomposition (rank = n*m*p = 60) ---
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # --- Verification function ---
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition exactly reproduces the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - T[i, j, k]) > tol:
                        return False
        return True

    # --- Half-integer rounding and verification ---
    def round_and_verify(U, V, W):
        """Round to nearest half-integer and check if still valid."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        if verify(U_r, V_r, W_r, tol=1e-6):
            return U_r, V_r, W_r, True
        return None, None, None, False

    # --- Residuals function for optimization ---
    def residuals(params, r):
        """Compute residuals for a given rank r decomposition."""
        U_flat = params[:n * m * r]
        V_flat = params[n * m * r : n * m * r + m * p * r]
        W_flat = params[n * m * r + m * p * r : n * m * r + m * p * r + p * n * r]
        U = U_flat.reshape(n * m, r)
        V = V_flat.reshape(m * p, r)
        W = W_flat.reshape(p * n, r)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    res.append(val - T[i, j, k])
        return np.array(res)

    # --- Attempt optimization for ranks below schoolbook ---
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # In practice, known bounds for 3x4x5 are not well established below 60,
    # but we attempt a search from a reasonable lower bound upward.
    
    # We'll try a limited set of ranks to keep within time budget
    # Start from 30 (half of schoolbook) and go up in steps
    candidate_ranks = list(range(30, 60, 2))  # 30, 32, 34, ..., 58
    
    np.random.seed(42)  # Reproducibility
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Multiple random restarts for each rank
        max_restarts = 3
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_init = np.random.randn(n * m, target_rank)
            V_init = np.random.randn(m * p, target_rank)
            W_init = np.random.randn(p * n, target_rank)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(
                    residuals, x0, args=(target_rank,),
                    method='lm', max_nfev=500, xtol=1e-12, ftol=1e-12
                )
                
                if result.success or result.cost < 1e-10:
                    U_opt = result.x[:n * m * target_rank].reshape(n * m, target_rank)
                    V_opt = result.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
                    W_opt = result.x[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
                    
                    # Try rounding to half-integers
                    U_r, V_r, W_r, valid = round_and_verify(U_opt, V_opt, W_opt)
                    if valid:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break
                    
                    # Also check if the unrounded solution is close enough
                    # and if rounding to integers works
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if verify(U_int, V_int, W_int, tol=1e-6):
                        best_U, best_V, best_W = U_int, V_int, W_int
                        best_rank = target_rank
                        found = True
                        break
                        
            except Exception:
                continue
        
        if found:
            break  # Found a valid lower-rank decomposition
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-6):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb

    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
