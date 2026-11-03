import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    Uses vectorized operations for speed.
    """
    rank = U.shape[1]
    # T_ijk = sum_r U_ij_r * V_jk_r * W_ki_r
    # We can compute this by iterating over r or using einsum.
    # einsum is fast.
    # U: (n*m, R) -> (n, m, R)
    # V: (m*p, R) -> (m, p, R)
    # W: (p*n, R) -> (p, n, R)
    
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Product: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Result shape: (n, m, p)
    # Note: W is indexed by k, i. So W_tensor[k, i, r].
    
    # We need to align indices.
    # U: i, j, r
    # V: j, k, r
    # W: k, i, r
    
    # einsum('ijr,jkr,kir->ijp')? No, W is k,i,r.
    # Let's swap W axes to i,k,r? No, W is given as (p*n, R) -> (p, n, R).
    # So W[k, i, r].
    
    # We want result[i, j, k].
    # term = U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    # Using einsum:
    # 'ijr,jkr,kir->ijk'
    
    try:
        product = np.einsum('ijr,jkr,kir->ijk', U_tensor, V_tensor, W_tensor)
    except Exception:
        return False
        
    # Check if product is all 1s
    # Matrix multiplication tensor T is 1 for all i,j,k.
    # Wait, the tensor for matrix multiplication is T_{ijk} = 1 for all i,j,k?
    # Yes, the "multiplication tensor" for $A \times B$ has entries $T_{ijk} = 1$ if the indices are valid.
    # Since we are multiplying a full 2x3 by 3x7 matrix, all entries are 1.
    
    if np.allclose(product, 1.0, atol=1e-9):
        return True
    return False

def _als_decomposition(n, m, p, rank, iterations=100, seed=0):
    """
    Perform ALS to find a rank 'rank' decomposition.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    T = np.ones((n, m, p))
    
    for it in range(iterations):
        # Update U
        # For each i, j, solve A[i,j] @ u = T[i,j,:]
        # A[i,j] is (p, rank) where A[i,j][k, r] = V[j,k,r] * W[k,i,r]
        for i in range(n):
            for j in range(m):
                # Construct A
                # V[j, :, :] is (p, rank)
                # W[:, i, :] is (p, rank)
                # A[k, r] = V[j, k, r] * W[k, i, r]
                A = V_tensor[j, :, :] * W_tensor[:, i, :] # (p, rank)
                b = T[i, j, :] # (p,)
                
                # Solve least squares
                try:
                    u, residuals, rank_u, s = np.linalg.lstsq(A, b, rcond=None)
                    U_tensor[i, j, :] = u
                except:
                    pass
                    
        # Update V
        # For each j, k, solve A[j,k] @ v = T[:, j, k]
        # A[j,k] is (n, rank) where A[j,k][i, r] = U[i,j,r] * W[k,i,r]
        for j in range(m):
            for k in range(p):
                # U[:, j, :] is (n, rank)
                # W[k, :, :] is (n, rank)
                A = U_tensor[:, j, :] * W_tensor[k, :, :] # (n, rank)
                b = T[:, j, k] # (n,)
                
                try:
                    v, residuals, rank_v, s = np.linalg.lstsq(A, b, rcond=None)
                    V_tensor[j, k, :] = v
                except:
                    pass
                    
        # Update W
        # For each k, i, solve A[k,i] @ w = T[i, :, k]
        # A[k,i] is (m, rank) where A[k,i][j, r] = U[i,j,r] * V[j,k,r]
        for k in range(p):
            for i in range(n):
                # U[i, :, :] is (m, rank)
                # V[:, k, :] is (m, rank)
                A = U_tensor[i, :, :] * V_tensor[:, k, :] # (m, rank)
                b = T[i, :, k] # (m,)
                
                try:
                    w, residuals, rank_w, s = np.linalg.lstsq(A, b, rcond=None)
                    W_tensor[k, i, :] = w
                except:
                    pass
                    
        # Check convergence / validity
        if it % 10 == 0:
            # Quick check
            pass

    return U_tensor.reshape(n*m, rank), V_tensor.reshape(m*p, rank), W_tensor.reshape(p*n, rank)

def _snap_to_half_int(U, V, W):
    """
    Snap entries to the nearest 0.5.
    """
    return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Try to find a low rank decomposition
    # Known rank for 2x3x7 is likely < 42.
    # Let's try ranks from 30 to 41.
    # If we find one that snaps and verifies, return it.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Search for lower ranks
    for target_rank in range(30, 42): # Try 30 to 41
        if time.time() > deadline:
            break
            
        # Try a few seeds for each rank
        for seed in range(5):
            if time.time() > deadline:
                break
                
            U, V, W = _als_decomposition(n, m, p, target_rank, iterations=50, seed=seed)
            
            # Snap to half-integers
            U_snap, V_snap, W_snap = _snap_to_half_int(U, V, W)
            
            # Verify
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                best_rank = target_rank
                best_U = U_snap
                best_V = V_snap
                best_W = W_snap
                break # Found a valid one for this rank, try lower? No, we iterate down?
                # We iterate from 30 to 41. So 30 is best.
                # If we find one at 30, we can break the outer loop?
                # Yes, since we want minimal rank.
                # But we might find a valid one at 30, then later find a valid one at 31?
                # We iterate 30, 31, ... so the first one found is the lowest rank.
                break
        
        if best_rank == target_rank:
            break # Found the lowest rank in this range
            
    # If no low rank found, use schoolbook
    # The schoolbook is already set in best_U etc.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
