"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy: Decompose the 2x3 * 3x6 problem into two independent 2x3 * 3x3 problems.
    B is split into B1 (cols 0-2) and B2 (cols 3-5).
    C1 = A * B1 and C2 = A * B2.
    We find a rank-r decomposition for the 2x3 * 3x3 tensor and duplicate it.
    Total rank = 2 * r.
    
    We use Alternating Least Squares (ALS) to find a low-rank decomposition of the 2x3 * 3x3 tensor.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Sub-problem dimensions: n=2, m=3, p_sub=3
    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Build the 2x3 * 3x3 tensor T_sub
    # Indices: a = i*m + j, b = j*p_sub + k, c = k*n + i
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub), dtype=np.float64)
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                a = i * m_sub + j
                b = j * p_sub + k
                c = k * n_sub + i
                T_sub[a, b, c] = 1.0
    
    vec_T_sub = T_sub.reshape(-1)
    
    # Try ranks r for the sub-problem
    # Lower bound is 3. Try 6, 7, 8...
    # Max rank is n_sub*m_sub*p_sub = 18
    best_rank_sub = None
    best_U_sub = None
    best_V_sub = None
    best_W_sub = None
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Search for minimal rank r
    for r in range(6, 19):  # Start from 6, go up to 18
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random initializations for this rank
        for trial in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n_sub * m_sub, r)
            V = np.random.randn(m_sub * p_sub, r)
            W = np.random.randn(p_sub * n_sub, r)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Solve for U
                # T = sum_r U_r V_r^T W_r^T
                # vec(T) = sum_r U_r (V_r (x) W_r)^T ? 
                # vec(T) = sum_r (V_r (kron) W_r)^T U_r ? No.
                # Standard ALS for tensor:
                # Fix V, W. Solve for U.
                # T = sum_r U[:,r] * (V[:,r] (kron) W[:,r])
                # This is not quite right. 
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # vec(T) = sum_r U[:,r] (V[:,r] (kron) W[:,r])
                # So vec(T) = M_U U_flat
                # M_U[:, r] = V[:,r] (kron) W[:,r]
                
                M_U = np.zeros((m_sub * p_sub * p_sub * n_sub, r))
                for r_idx in range(r):
                    M_U[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx])
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M_U, vec_T_sub, rcond=None)
                except:
                    break
                U = U_new.reshape(n_sub * m_sub, r)
                
                # Solve for V
                M_V = np.zeros((n_sub * m_sub * p_sub * n_sub, r))
                for r_idx in range(r):
                    M_V[:, r_idx] = np.kron(W[:, r_idx], U[:, r_idx])
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_V, vec_T_sub, rcond=None)
                except:
                    break
                V = V_new.reshape(m_sub * p_sub, r)
                
                # Solve for W
                M_W = np.zeros((n_sub * m_sub * m_sub * p_sub, r))
                for r_idx in range(r):
                    M_W[:, r_idx] = np.kron(U[:, r_idx], V[:, r_idx])
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_W, vec_T_sub, rcond=None)
                except:
                    break
                W = W_new.reshape(p_sub * n_sub, r)
                
                # Check error
                # Reconstruct T_sub
                recon = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
                for r_idx in range(r):
                    recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                
                error = np.max(np.abs(recon - T_sub))
                if error < 1e-8:
                    best_rank_sub = r
                    best_U_sub = U.copy()
                    best_V_sub = V.copy()
                    best_W_sub = W.copy()
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # If we found a sub-decomposition, expand to 2x3 * 3x6
    if best_rank_sub is not None:
        r_total = 2 * best_rank_sub
        U_total = np.zeros((n * m, r_total))
        V_total = np.zeros((m * p, r_total))
        W_total = np.zeros((p * n, r_total))
        
        # U is the same for both blocks
        U_total[:, 0:best_rank_sub] = best_U_sub
        U_total[:, best_rank_sub:2*best_rank_sub] = best_U_sub
        
        # V: split k into k1 (0-2) and k2 (3-5)
        for j in range(m):
            for k in range(p):
                if k < p_sub:
                    # Block 1
                    V_total[j * p + k, 0:best_rank_sub] = best_V_sub[j * p_sub + k, 0:best_rank_sub]
                else:
                    # Block 2
                    k_sub = k - p_sub
                    V_total[j * p + k, best_rank_sub:2*best_rank_sub] = best_V_sub[j * p_sub + k_sub, 0:best_rank_sub]
        
        # W: split k into k1 (0-2) and k2 (3-5)
        for k in range(p):
            for i in range(n):
                if k < p_sub:
                    # Block 1
                    W_total[k * n + i, 0:best_rank_sub] = best_W_sub[k * n_sub + i, 0:best_rank_sub]
                else:
                    # Block 2
                    k_sub = k - p_sub
                    W_total[k * n + i, best_rank_sub:2*best_rank_sub] = best_W_sub[k_sub * n_sub + i, 0:best_rank_sub]
        
        # Final verification
        T_full = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_full[a, b, c] = 1.0
        
        recon_full = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r_total):
            recon_full += np.einsum('a,b,c->abc', U_total[:, r_idx], V_total[:, r_idx], W_total[:, r_idx])
        
        if np.max(np.abs(recon_full - T_full)) < 1e-8:
            return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    
    # Fallback: Schoolbook
    rank = n * m * p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
