"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np
from itertools import product
import scipy.optimize as optimize

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] where:
    T[i,j,k] = 1 if this corresponds to a valid product term.
    For multiplying A (n x m) by B (m x p) to get C (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor has entries T[i, j, k] = 1 for all valid (i,j,k).
    """
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    
    For exact decomposition: for all i,j,k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check each (i, j, k) triple
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                # sum over c of u_val[c] * v_val[c] * w_val[c]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _solve_rank_r(n, m, p, r, max_restarts=50, time_budget=10.0):
    """
    Try to find a rank-r decomposition for n x m x p matmul tensor.
    Uses scipy.optimize.least_squares on the sum of squares of residuals.
    """
    n_eq = n * m * p
    n_vars = (n*m + m*p + p*n) * r
    
    def objective(params):
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r :].reshape(p*n, r)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    residuals.append(s - 1.0)
        return residuals
    
    best_U, best_V, best_W = None, None, None
    best_loss = float('inf')
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > time_budget:
            break
            
        # Random initialization
        # Scale to avoid large initial gradients
        U0 = np.random.randn(n*m, r) * 0.5
        V0 = np.random.randn(m*p, r) * 0.5
        W0 = np.random.randn(p*n, r) * 0.5
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = optimize.least_squares(objective, x0, method='lm', max_nfev=1000)
        except Exception:
            continue
            
        if res.cost < best_loss:
            best_loss = res.cost
            U_sol = res.x[:n*m*r].reshape(n*m, r)
            V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_sol = res.x[n*m*r + m*p*r :].reshape(p*n, r)
            
            # Snap to half-integers
            U_snap = _snap_to_half_integers(U_sol)
            V_snap = _snap_to_half_integers(V_sol)
            W_snap = _snap_to_half_integers(W_sol)
            
            # Verify snapped solution
            ok, _ = _verify_decomposition(U_snap, V_snap, W_snap, n, m, p)
            if ok:
                return U_snap, V_snap, W_snap
            
            # Also check unsnapped if it's very close, but we need half-integers
            # If unsnapped is exact but not half-integer, it's invalid for the output format
            # But maybe we can try to find a nearby half-integer solution?
            # For now, only accept snapped solutions that verify.
            
            # Store best found (even if not snapped) in case we need to debug, 
            # but we can only return valid half-integer solutions.
            
    return None

def _try_block_decomposition(n, m, p):
    """
    Try to decompose 2x3x6 as two 2x3x3 blocks.
    Rank(2x3x6) = 2 * Rank(2x3x3).
    If we find rank 17 for 2x3x3, we get rank 34 for 2x3x6.
    """
    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Try rank 17 for 2x3x3
    result_17 = _solve_rank_r(n_sub, m_sub, p_sub, 17, max_restarts=100, time_budget=20.0)
    if result_17 is not None:
        U_sub, V_sub, W_sub = result_17
        # Expand to 2x3x6
        # U is same
        U_full = np.zeros((2*3, 34))
        U_full[:, :17] = U_sub
        U_full[:, 17:] = U_sub
        
        # V and W are split
        V_full = np.zeros((3*6, 34))
        W_full = np.zeros((6*2, 34))
        
        for k in range(3):
            V_full[k*3 + 0, :17] = V_sub[k*3 + 0, :] # Wait, V index is j*p+k
            # In sub-problem, p=3. V_sub index is j*3 + k_sub.
            # In full problem, p=6.
            # For first block (k=0,1,2):
            # V_full[j*6 + k, c] = V_sub[j*3 + k, c] for c in 0..16
            # For second block (k=3,4,5):
            # V_full[j*6 + k, c] = V_sub[j*3 + (k-3), c-17]
            
            # Let's do it properly
            pass
            
        # Re-do expansion logic
        V_full = np.zeros((18, 34))
        W_full = np.zeros((12, 34))
        
        for j in range(3):
            for k_sub in range(3):
                # First block: k = k_sub
                V_full[j*6 + k_sub, :17] = V_sub[j*3 + k_sub, :]
                # Second block: k = k_sub + 3
                V_full[j*6 + k_sub + 3, 17:] = V_sub[j*3 + k_sub, :]
                
        for i in range(2):
            for k_sub in range(3):
                # First block: k = k_sub
                # W index in sub: k_sub * 2 + i
                # W index in full: k_sub * 2 + i
                W_full[k_sub * 2 + i, :17] = W_sub[k_sub * 2 + i, :]
                # Second block: k = k_sub + 3
                # W index in full: (k_sub + 3) * 2 + i
                W_full[(k_sub + 3) * 2 + i, 17:] = W_sub[k_sub * 2 + i, :]
                
        U_full = np.zeros((6, 34))
        U_full[:, :17] = U_sub
        U_full[:, 17:] = U_sub
        
        ok, _ = _verify_decomposition(U_full, V_full, W_full, 2, 3, 6)
        if ok:
            return U_full, V_full, W_full
            
    # Try rank 18 for 2x3x3 (should work, gives 36)
    # But schoolbook is already 36.
    
    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try block decomposition strategy
    if time.time() - START_TIME < 80.0:
        result = _try_block_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            ok, _ = _verify_decomposition(U, V, W, n, m, p)
            if ok:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    
    # Safety: ensure we have a valid decomposition
    ok, _ = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not ok:
        # Fall back to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
    
    # Convert to lists for the return format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
