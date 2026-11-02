"""Matrix multiplication 2,2,11 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 2, 2, 11
    rank = U.shape[1]
    if T is None:
        T = _build_target_tensor()
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-10)

def _build_strassen_rank39():
    """Build the rank-39 Strassen-based construction."""
    n, m, p = 2, 2, 11
    rank = 39
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1
    
    # Last column k=10: schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    U[0, r] += 1; V[v_e2, r] += 1; W[w_c02, r] += 1; r += 1
    U[1, r] += 1; V[v_g2, r] += 1; W[w_c02, r] += 1; r += 1
    U[2, r] += 1; V[v_e2, r] += 1; W[w_c12, r] += 1; r += 1
    U[3, r] += 1; V[v_g2, r] += 1; W[w_c12, r] += 1; r += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, T, max_time=60.0):
    """Try to reduce the rank by finding redundant terms."""
    start_time = time.time()
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try removing each term one at a time, iteratively
    improved = True
    while improved:
        if time.time() - start_time > max_time:
            break
        improved = False
        current_rank = best_U.shape[1]
        
        for r_remove in range(current_rank):
            if time.time() - start_time > max_time:
                break
            
            # Compute residual: target - sum of all terms except r_remove
            residual = T.copy()
            for r_keep in range(current_rank):
                if r_keep == r_remove:
                    continue
                residual -= np.einsum('a,b,c->abc', 
                                       best_U[:, r_keep], 
                                       best_V[:, r_keep], 
                                       best_W[:, r_keep])
            
            # Check if residual is zero (within tolerance)
            if np.max(np.abs(residual)) < 1e-10:
                # Term r_remove is redundant!
                new_U = np.delete(best_U, r_remove, axis=1)
                new_V = np.delete(best_V, r_remove, axis=1)
                new_W = np.delete(best_W, r_remove, axis=1)
                
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_U.shape[1]
                improved = True
                break
    
    return best_U, best_V, best_W, best_rank

def _try_merge_terms(U, V, W, T, max_time=30.0):
    """Try to merge pairs of terms that might combine."""
    start_time = time.time()
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # For each pair of terms, check if they can be merged into one term
    # This is complex, so we'll try a simpler heuristic
    # Check if any two terms are proportional
    
    for i in range(rank):
        if time.time() - start_time > max_time:
            break
        for j in range(i+1, rank):
            if time.time() - start_time > max_time:
                break
            
            # Check if term i and term j are proportional
            # i.e., U_i * V_i * W_i = c * U_j * V_j * W_j for some scalar c
            # This is hard to check directly, so we'll skip for now
            pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 11
    
    # Build the rank-39 Strassen-based construction
    U, V, W = _build_strassen_rank39()
    
    # Verify the construction
    T = _build_target_tensor()
    if not _verify(U, V, W, T):
        # Fallback: Schoolbook (Rank 44)
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook))
        V_sb = np.zeros((m*p, rank_schoolbook))
        W_sb = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to reduce rank
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        U, V, W, new_rank = _try_reduce_rank(U, V, W, T, max_time=min(remaining_time - 5, 60))
        if new_rank < U.shape[1]:
            pass  # Already updated
        # Re-verify
        if not _verify(U, V, W, T):
            # Reset to original
            U, V, W = _build_strassen_rank39()
    
    # Final verification
    if not _verify(U, V, W, T):
        # Fallback: Schoolbook (Rank 44)
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook))
        V_sb = np.zeros((m*p, rank_schoolbook))
        W_sb = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
