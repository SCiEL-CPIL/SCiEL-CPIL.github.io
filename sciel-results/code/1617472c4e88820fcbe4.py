import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """
    Verify if U, V, W form a valid decomposition of T.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Condition: sum_r U[i, r] V[j, r] W[k, r] == T[i, j, k]
    """
    try:
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if not np.allclose(T_recon, T, atol=tol, rtol=tol):
            return False
    except Exception:
        return False
    return True

def _is_half_integer(arr, tol=1e-9):
    """
    Check if all entries in arr are half-integers (i.e., 2*entry is an integer).
    """
    scaled = 2.0 * np.array(arr, dtype=np.float64)
    rounded = np.round(scaled)
    return np.allclose(scaled, rounded, atol=tol)

def _round_to_half_integer(arr, tol=1e-9):
    """
    Round entries to nearest half-integer.
    """
    scaled = 2.0 * np.array(arr, dtype=np.float64)
    rounded = np.round(scaled)
    return rounded / 2.0

def _find_rank_via_svd(T, max_rank_to_check=None):
    """
    Estimate the rank of the tensor by checking the rank of its flattening.
    The tensor T of shape (I, J, K) can be flattened to a matrix of size (I, J*K).
    The rank of this matrix is an upper bound on the tensor rank.
    If the rank is R, we can try to find a decomposition of rank R.
    """
    I, J, K = T.shape
    T_flat = T.reshape(I, J * K)
    # Compute SVD
    U, s, Vt = np.linalg.svd(T_flat, full_matrices=False)
    # Determine rank based on singular values
    # We want the smallest R such that the reconstruction error is negligible
    # Since T is 0/1, the true rank should be an integer.
    # Let's find the rank where the singular values drop to near zero.
    tol = 1e-10
    rank = 0
    for val in s:
        if val > tol:
            rank += 1
        else:
            break
    
    # The rank of the flattening is an upper bound.
    # For matrix multiplication tensors, the rank is often equal to the rank of the flattening
    # if the tensor is "generic" enough, but for specific tensors it can be lower.
    # However, we know that the tensor rank <= rank(flattening).
    # Let's return this rank as a candidate.
    return rank

def _als_decompose(T, R, max_iter=500, tol=1e-8, seed=0):
    """
    Alternating Least Squares (ALS) for tensor decomposition.
    T: (I, J, K)
    Returns U, V, W of shapes (I, R), (J, R), (K, R) or None if fails.
    """
    I, J, K = T.shape
    
    # Initialize randomly
    np.random.seed(seed)
    U = np.random.randn(I, R)
    V = np.random.randn(J, R)
    W = np.random.randn(K, R)
    
    # Normalize to avoid divergence
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    T_flat = T.reshape(I * J, K)
    
    for it in range(max_iter):
        # Update U
        A = np.zeros((J * K, R))
        for r in range(R):
            A[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
        try:
            U_new, _, _, _ = np.linalg.lstsq(A, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass
            
        # Update V
        B = np.zeros((I * K, R))
        for r in range(R):
            B[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
        T_perm_V = np.transpose(T, (1, 0, 2)).reshape(J, I * K)
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(B, T_perm_V.T, rcond=None)
            V = V_new.T
        except:
            pass
            
        # Update W
        C = np.zeros((I * J, R))
        for r in range(R):
            C[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
        T_perm_W = np.transpose(T, (2, 0, 1)).reshape(K, I * J)
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(C, T_perm_W.T, rcond=None)
            W = W_new.T
        except:
            pass
        
        # Check convergence
        if it % 10 == 0:
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < tol:
                break
                
    # Final check
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.linalg.norm(T_recon - T)
    if err < 1e-6:
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0 # Leave 10s margin
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    T = _build_tensor(n, m, p)
    
    # Estimate rank via SVD of flattening
    # This gives an upper bound.
    # For 2x4x7, let's see what SVD says.
    # If SVD rank is R_svd, the tensor rank is <= R_svd.
    # We will search from R_svd down to some limit.
    
    # However, the SVD rank of the flattening might be higher than the tensor rank.
    # But it's a good starting point.
    
    # Let's just try ranks from 55 down to 40 as before, but maybe use the SVD rank to set the upper bound.
    # Actually, for small tensors, the SVD rank of the flattening is often the tensor rank.
    
    R_svd = _find_rank_via_svd(T)
    
    # We know the rank is at most 56.
    # Let's start from min(R_svd, 55) down to 40.
    # If R_svd is 56, we start from 55.
    
    current_rank = min(R_svd, schoolbook_rank - 1)
    min_rank_to_try = 40
    
    # It is possible that the SVD rank is lower than the true tensor rank? No, tensor rank <= flattening rank.
    # So if SVD rank is 50, tensor rank <= 50.
    
    while current_rank >= min_rank_to_try:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Try this rank
        found = False
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            np.random.seed(restart * 1000 + current_rank)
            res = _als_decompose(T, current_rank, max_iter=500, tol=1e-8, seed=restart * 1000 + current_rank)
            if res is not None:
                U, V, W = res
                # Verify
                if _verify(U, V, W, T, tol=1e-6):
                    # Try to round
                    U_r = _round_to_half_integer(U)
                    V_r = _round_to_half_integer(V)
                    W_r = _round_to_half_integer(W)
                    
                    if _verify(U_r, V_r, W_r, T, tol=1e-9):
                        # Found a valid half-integer solution
                        best_result = {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        best_rank = current_rank
                        found = True
                        break # Move to lower rank
                    else:
                        # Rounding failed.
                        # Maybe we can try to solve for exact rationals?
                        # Or just accept that this rank might not have a half-integer solution.
                        pass
        
        if found:
            current_rank -= 1
        else:
            current_rank -= 1
            
    return best_result
