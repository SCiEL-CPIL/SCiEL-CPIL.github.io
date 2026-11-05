"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 by 3x8 matrix multiplication.
    
    Uses iterative hard thresholding (IHT) formulated as sparse tensor completion.
    The factor matrices are treated as a block-sparse tensor and IHT is applied
    to minimize the number of non-zero rank-1 terms.
    """
    n, m, p = DIMENSIONS
    
    # Build target tensor T with shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 iff j==j' and k==k' and i==i'
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    def try_iht(rank, max_iter=200, lr=0.01, threshold=1e-8, n_restarts=3):
        """
        Try to find a rank-`rank` decomposition using IHT.
        Returns (U, V, W) arrays or None if failed.
        """
        for restart in range(n_restarts):
            # Initialize with random sparse matrices
            np.random.seed(42 + restart)
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Initialize with small random values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            for it in range(max_iter):
                # Compute residual: R = T - sum_r U[:,r] o V[:,r] o W[:,r]
                # Efficiently compute current tensor estimate
                # T_est[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                
                # For efficiency, compute gradient with respect to U, V, W
                # dT/du = V o W (for each r)
                # Gradient for U: grad_U[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] - sum_{b,c} T_est[a,b,c] * V[b,r] * W[c,r]
                # But we can simplify: since T is sparse, we only need to consider nonzero entries
                
                # Compute current estimate at nonzero positions
                # For each rank r, the contribution to T[a,b,c] is U[a,r]*V[b,r]*W[c,r]
                
                # Gradient computation:
                # grad_U = (T * (V o W)) - (T_est * (V o W))  but this is expensive
                # Instead: grad_U[a, r] = sum_{(b,c): T[a,b,c]=1} T[a,b,c] * V[b,r] * W[c,r] - sum_{b,c} T_est[a,b,c] * V[b,r] * W[c,r]
                
                # More efficient: 
                # Let M_r = V[:,r] o W[:,r] (shape m*p x p*n)
                # Then grad_U[:, r] = T[:, :, :] contracted with M_r - T_est contracted with M_r
                # = (T - T_est) contracted with M_r
                
                # Compute T_est efficiently
                T_est = np.zeros((n*m, m*p, p*n))
                for r in range(rank):
                    T_est += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                # Residual
                R = T - T_est
                
                # Compute gradients
                # grad_U[a, r] = sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
                # grad_V[b, r] = sum_{a,c} R[a,b,c] * U[a,r] * W[c,r]
                # grad_W[c, r] = sum_{a,b} R[a,b,c] * U[a,r] * V[b,r]
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(rank):
                    # grad_U[:, r] = R @ (V[:, r] o W[:, r]) flattened
                    # R is (n*m, m*p, p*n), V[:,r] is (m*p,), W[:,r] is (p*n,)
                    # grad_U[a, r] = sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
                    grad_U[:, r] = np.einsum('abc,b,c->a', R, V[:, r], W[:, r])
                    grad_V[:, r] = np.einsum('abc,a,c->b', R, U[:, r], W[:, r])
                    grad_W[:, r] = np.einsum('abc,a,b->c', R, U[:, r], V[:, r])
                
                # IHT: update and threshold
                U_new = U - lr * grad_U
                V_new = V - lr * grad_V
                W_new = W - lr * grad_W
                
                # Hard thresholding: keep top-k values per column
                # For each column r, keep the largest |values|
                U_thresh = np.zeros_like(U_new)
                V_thresh = np.zeros_like(V_new)
                W_thresh = np.zeros_like(W_new)
                
                for r in range(rank):
                    # For U column r, keep entries with largest absolute values
                    u_col = U_new[:, r]
                    v_col = V_new[:, r]
                    w_col = W_new[:, r]
                    
                    # Keep all non-negligible entries (soft thresholding)
                    mask_u = np.abs(u_col) > threshold
                    mask_v = np.abs(v_col) > threshold
                    mask_w = np.abs(w_col) > threshold
                    
                    U_thresh[:, r] = np.where(mask_u, u_col, 0.0)
                    V_thresh[:, r] = np.where(mask_v, v_col, 0.0)
                    W_thresh[:, r] = np.where(mask_w, w_col, 0.0)
                
                U, V, W = U_thresh, V_thresh, W_thresh
                
                # Check convergence
                if np.max(np.abs(R)) < 1e-6:
                    # Verify exactness
                    T_final = np.zeros((n*m, m*p, p*n))
                    for r in range(rank):
                        T_final += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                    if np.allclose(T_final, T, atol=1e-8):
                        return U, V, W
            
            # After all iterations, check if we got close
            T_final = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T_final += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            if np.allclose(T_final, T, atol=1e-8):
                return U, V, W
        
        return None
    
    # Try structured ranks below schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    lower_bound = max(n*m, m*p, p*n)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound up to schoolbook
    # Focus on ranks that are multiples of small numbers or have structure
    candidate_ranks = sorted(set([lower_bound, 25, 28, 30, 32, 36, 40, 45, 48, 50, 54, 56, 60, 64, 66, 68, 70, 71, 72]))
    candidate_ranks = [r for r in candidate_ranks if r <= schoolbook_rank]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        result = try_iht(rank, max_iter=150, lr=0.008, threshold=1e-7, n_restarts=2)
        if result is not None:
            U, V, W = result
            actual_rank = U.shape[1]
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_result = (U, V, W)
                break
    
    # If IHT didn't find anything better, use schoolbook
    if best_result is None:
        rank = schoolbook_rank
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    else:
        U, V, W = best_result
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
