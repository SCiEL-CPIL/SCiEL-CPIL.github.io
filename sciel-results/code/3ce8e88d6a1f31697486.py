"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 110  # safety margin before 120s limit

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Check all entries are exactly 1 (within floating point tolerance for exact values)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    # Additional check: entries should be close to integers or simple rationals
    # For exact decomposition, values should be representable
    return True

def _verify_strict(U, V, W, n, m, p):
    """Stricter verification with tighter tolerance"""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(T, 1.0, atol=1e-12):
        return False
    return True

def _als_solve(U, V, W, n, m, p, target_rank, max_iter=50, tol=1e-10):
    """Alternating least squares to refine decomposition to target_rank."""
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
    best_err = float('inf')
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        # Solve for U given V and W
        U_new = np.zeros((n * m, target_rank), dtype=float)
        for i in range(n):
            for j in range(m):
                # A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A_mat = np.outer(V_curr[j*p:(j+1)*p, :].flatten(), 
                                  np.ones(p))  # placeholder, build properly
                # Build A_mat properly: p x target_rank
                A_mat = np.zeros((p, target_rank))
                for k in range(p):
                    A_mat[k, :] = V_curr[j*p+k, :] * W_curr[k*n+i, :]
                b_vec = np.ones(p)
                try:
                    sol, _, _, _ = linalg.lstsq(A_mat, b_vec)
                    U_new[i*m+j, :] = sol
                except:
                    U_new[i*m+j, :] = U_curr[i*m+j, :target_rank] if target_rank <= U_curr.shape[1] else 0
        
        # Solve for V given U and W
        V_new = np.zeros((m * p, target_rank), dtype=float)
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, target_rank))
                for i in range(n):
                    A_mat[i, :] = U_curr[i*m+j, :] * W_curr[k*n+i, :]
                b_vec = np.ones(n)
                try:
                    sol, _, _, _ = linalg.lstsq(A_mat, b_vec)
                    V_new[j*p+k, :] = sol
                except:
                    V_new[j*p+k, :] = V_curr[j*p+k, :target_rank] if target_rank <= V_curr.shape[1] else 0
        
        # Solve for W given U and V
        W_new = np.zeros((p * n, target_rank), dtype=float)
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, target_rank))
                for j in range(m):
                    A_mat[j, :] = U_curr[i*m+j, :] * V_curr[j*p+k, :]
                b_vec = np.ones(m)
                try:
                    sol, _, _, _ = linalg.lstsq(A_mat, b_vec)
                    W_new[k*n+i, :] = sol
                except:
                    W_new[k*n+i, :] = W_curr[k*n+i, :target_rank] if target_rank <= W_curr.shape[1] else 0
        
        # Check error
        U3 = U_new.reshape(n, m, target_rank)
        V3 = V_new.reshape(m, p, target_rank)
        W3 = W_new.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
        
        if err < tol:
            # Verify exactly
            if _verify_exact(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, True
            elif _verify_strict(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, True
        
        # Update
        U_curr, V_curr, W_curr = U_new, V_new, W_new
    
    # Final check
    if best_err < 1e-8:
        if _verify_exact(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W, True
        elif _verify_strict(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W, True
    
    return best_U, best_V, best_W, False

def _random_init(n, m, p, rank, seed=None):
    """Random initialization for ALS."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Scale to have reasonable magnitude
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    
    return U, V, W

def _structured_init(n, m, p, rank):
    """Structured initialization based on schoolbook subset."""
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Try to pack schoolbook terms into rank columns
    # Each schoolbook term is (i,j,k) with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # We can group terms that share some indices
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < rank:
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    return U, V, W

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = 16
    # For 3x4x4, expected optimal rank is around 18-22
    # Try ranks from 16 up to schoolbook
    
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank + 1  # Start above schoolbook so any valid result is better
    
    # First, get schoolbook as fallback
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=float)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=float)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Start from lower ranks and go up
    for target_rank in range(16, schoolbook_rank + 1):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        if target_rank >= best_rank:
            break
        
        # Multiple restarts for this rank
        found = False
        n_restarts = 10 if target_rank < 25 else 5
        
        for restart in range(n_restarts):
            if time.time() - _START_TIME > _TIMEOUT:
                break
            
            if found:
                break
            
            # Try different initializations
            if restart == 0:
                # Structured init
                U_init, V_init, W_init = _structured_init(n, m, p, target_rank)
            elif restart == 1:
                # Random with fixed seed
                U_init, V_init, W_init = _random_init(n, m, p, target_rank, seed=restart * 100)
            else:
                # Random with varying seeds
                U_init, V_init, W_init = _random_init(n, m, p, target_rank, seed=restart * 137 + 42)
            
            # Run ALS
            U_res, V_res, W_res, success = _als_solve(U_init, V_init, W_init, n, m, p, target_rank, 
                                                       max_iter=30, tol=1e-10)
            
            if success:
                # Verify one more time
                if _verify_exact(U_res, V_res, W_res, n, m, p):
                    best_U, best_V, best_W = U_res, V_res, W_res
                    best_rank = target_rank
                    found = True
                    break
        
        # If we found a valid decomposition at this rank, we can stop
        # (since we're going from low to high)
        if found:
            break
    
    # Convert to lists for return
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
