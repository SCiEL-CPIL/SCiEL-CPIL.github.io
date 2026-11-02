import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p, rank):
    """
    Verify that U, V, W form a valid tensor decomposition of the n x m x p matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    if rank == 0:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T_decomp[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can compute this efficiently using broadcasting and einsum or loops.
    # Given small dimensions, a simple loop over c is fine, but we can vectorize over i,j,k.
    
    T_decomp = np.zeros((n, m, p))
    
    for c in range(rank):
        u_col = U_arr[:, c]
        v_col = V_arr[:, c]
        w_col = W_arr[:, c]
        
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        
        # T_decomp += u_2d[:,:,None] * v_2d[None,:,:] * w_2d[:,None,:] ?
        # u_2d is (n, m), v_2d is (m, p), w_2d is (p, n)
        # We need T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        
        # Let's use einsum for clarity and speed
        # 'ij, jk, ki -> ijk'
        # np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
        
        T_decomp += np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
        
    if not np.allclose(T_decomp, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook():
    """Generate the standard schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
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
    
    return U, V, W, rank

def _try_eliminate_columns(U, V, W, n, m, p, max_iters=100):
    """
    Try to reduce rank by finding linear dependencies between the tensor terms.
    If T_r = sum_{c != r} alpha_c T_c, we can remove r and scale others.
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    
    current_rank = U.shape[1]
    
    for _ in range(max_iters):
        if current_rank == 0:
            break
            
        eliminated = False
        # We try to eliminate columns in order of index
        for r in range(current_rank):
            if current_rank == 1:
                # If only 1 column left, it must equal the identity tensor
                # Check if it does
                if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, 1):
                     # Can't eliminate the last one if it's the only one and it's wrong?
                     # But we start with valid schoolbook, so it should be fine.
                     pass
                else:
                    break # Can't eliminate the last column
            
            # Construct T_r
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            T_r = np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
            T_r_flat = T_r.flatten()
            
            # Construct matrix of other tensors
            mask = [c for c in range(current_rank) if c != r]
            if not mask:
                continue
                
            M = np.zeros((n*m*p, len(mask)))
            for idx, c in enumerate(mask):
                u_c = U[:, c].reshape(n, m)
                v_c = V[:, c].reshape(m, p)
                w_c = W[:, c].reshape(p, n)
                T_c = np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
                M[:, idx] = T_c.flatten()
            
            # Solve M alpha = T_r_flat
            try:
                alpha, residuals, rank_m, sv = np.linalg.lstsq(M, T_r_flat, rcond=None)
                # Check if the solution is accurate
                if np.allclose(M @ alpha, T_r_flat, atol=1e-9):
                    # We can eliminate r
                    # Update other columns: W[:, c] *= (1 + alpha[idx])
                    # Actually, T_new = sum_{c != r} (1 + alpha_c) T_c
                    # So we scale W (or U or V) by (1 + alpha_c)
                    for idx, c in enumerate(mask):
                        scale = 1.0 + alpha[idx]
                        # To keep numbers small, we can distribute the scaling or just apply to W
                        W[:, c] *= scale
                        # Optional: round to nearest 0.5 if close? 
                        # For now, keep float. The verification will check exactness.
                    
                    # Remove column r
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    current_rank -= 1
                    eliminated = True
                    break # Restart loop with new indices
            except np.linalg.LinAlgError:
                continue
        
        if not eliminated:
            break
            
    return U, V, W, current_rank

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the tensor decomposition of 3x5x6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 90
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook()
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank via linear dependency elimination
    # This is fast for small ranks, but might be slow if it tries many columns
    # We limit the time
    if time.time() < deadline - 10:
        U_red, V_red, W_red, rank_red = _try_eliminate_columns(U, V, W, n, m, p, max_iters=50)
        if rank_red < best_rank:
            # Verify the reduced version
            if _verify(U_red.tolist(), V_red.tolist(), W_red.tolist(), n, m, p, rank_red):
                best_U, best_V, best_W, best_rank = U_red, V_red, W_red, rank_red
    
    # If we have time, try a random search for even lower ranks?
    # Given the difficulty, we stick to the best found so far.
    
    # Final verification
    if not _verify(best_U.tolist(), best_V.tolist(), W_red.tolist() if best_rank == rank_red else best_W.tolist(), n, m, p, best_rank):
         # Fallback to schoolbook if verification fails due to numerical issues
         U, V, W, rank = _schoolbook()
         best_U, best_V, best_W, best_rank = U, V, W, rank

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
