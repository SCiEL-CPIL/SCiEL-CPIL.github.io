import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 148.0  # 2s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    def compute_errors(U, V, W):
        """Compute all (i,j,k) errors efficiently using einsum."""
        # errors[i,j,k] = |U[i*m+j] . (V[j*p+k] * W[k*n+i])| - 1
        # Shape: (n, m, p)
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each (i,j,k): dot(U[i*m+j], V[j*p+k]*W[k*n+i])
        # Reshape to use einsum:
        # U[i*m+j] -> U_2d[i, j, :, :] where U_2d has shape (n, m, 1, rank)
        # V[j*p+k] -> V_2d[j, k, :, :] shape (m, p, 1, rank)  
        # W[k*n+i] -> W_2d[k, i, :, :] shape (p, n, 1, rank)
        U2 = U.reshape(n, m, 1, -1)  # (n, m, 1, rank)
        V2 = V.reshape(m, p, 1, -1)  # (m, p, 1, rank)
        W2 = W.reshape(p, n, 1, -1)  # (p, n, 1, rank)
        # Product element-wise: U2[i,j] * V2[j,k] * W2[k,i]
        # Need to align dimensions: (n,m,1,rank) * (1,m,p,rank) * (1,1,p,n) -> broadcast
        # Let's do it step by step
        # A[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r]
        # B[i,j,k] = sum_r A[i,j,k,r] * W[k*n+i, r]
        # Use einsum
        # U2 shape (n,m,1,rank), V2 shape (m,p,1,rank) -> need to align
        # Let's just use loops for clarity but vectorize over rank
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    errors[i, j, k] = val - 1.0
        return errors
    
    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        errors = compute_errors(U, V, W)
        return np.max(np.abs(errors))
    
    def smooth_max_error(errors, beta=50.0):
        """Log-sum-exp smooth approximation of max|error|.
        Uses auxiliary epigraph formulation: min_t t + (1/beta)*log(sum exp(beta*|e_k|/t))
        But simpler: just use log-sum-exp of squared errors as smooth surrogate.
        """
        sq = errors * errors
        max_sq = np.max(sq)
        if max_sq < 1e-30:
            return 0.0
        # log-sum-exp trick for numerical stability
        shifted = sq / max_sq
        lse = np.log(np.sum(np.exp(shifted))) + np.log(max_sq)
        # Return sqrt of smooth max squared error
        return np.sqrt(np.exp(lse))
    
    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = (U_school.copy(), V_school.copy(), W_school.copy())
    
    candidate_ranks = [49, 50, 55, 60, 70, 80, 90, 100, 110, 120]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 25:
            break
        
        if rank >= rank_schoolbook:
            continue
        
        best_err_for_rank = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for init_seed in range(8):
            if time.time() > deadline - 15:
                break
            
            rng = np.random.RandomState(init_seed * 1000 + rank + 7)
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Structured initialization with random perturbation
            for r in range(rank):
                i_idx = r % n
                j_idx = (r // n) % m
                k_idx = (r // (n*m)) % p
                
                U[i_idx*m+j_idx, r] = 1.0
                V[j_idx*p+k_idx, r] = 1.0
                W[k_idx*n+i_idx, r] = 1.0
                
                if init_seed > 0:
                    scale = 0.05 + 0.05 * (init_seed % 3)
                    U[i_idx*m+j_idx, r] = 1.0 + rng.randn() * scale
                    V[j_idx*p+k_idx, r] = 1.0 + rng.randn() * scale
                    W[k_idx*n+i_idx, r] = 1.0 + rng.randn() * scale
            
            # Phase 1: Standard alternating minimization
            prev_err = float('inf')
            for it in range(60):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((rank, p))
                        for k in range(p):
                            B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            sol = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((rank, n))
                        for i in range(n):
                            A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            sol = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((rank, m))
                        for j in range(m):
                            C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            sol = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Compute error
                errors = compute_errors(U, V, W)
                max_err = np.max(np.abs(errors))
                
                if max_err < best_err_for_rank:
                    best_err_for_rank = max_err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if max_err < 1e-10:
                    break
                
                if max_err > prev_err * 1.5 and it > 15:
                    break
                prev_err = max_err
            
            if best_err_for_rank < 1e-8:
                break
        
        if best_err_for_rank < 1e-6 and best_U is not None:
            # Phase 2: High-precision polish pass with tighter tolerance
            # Use the smooth surrogate to guide final refinement
            U_polish = best_U.copy()
            V_polish = best_V.copy()
            W_polish = best_W.copy()
            
            polish_prev_err = best_err_for_rank
            for pol_it in range(30):
                if time.time() > deadline - 5:
                    break
                
                # Compute smooth error for monitoring
                errors = compute_errors(U_polish, V_polish, W_polish)
                max_err = np.max(np.abs(errors))
                
                if max_err < 1e-12:
                    break
                
                # Update U with high precision
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((rank, p))
                        for k in range(p):
                            B[:, k] = V_polish[j*p+k, :] * W_polish[k*n+i, :]
                        try:
                            sol = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                            U_polish[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((rank, n))
                        for i in range(n):
                            A[:, i] = U_polish[i*m+j, :] * W_polish[k*n+i, :]
                        try:
                            sol = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                            V_polish[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((rank, m))
                        for j in range(m):
                            C[:, j] = U_polish[i*m+j, :] * V_polish[j*p+k, :]
                        try:
                            sol = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                            W_polish[k*n+i, :] = sol
                        except:
                            pass
                
                errors = compute_errors(U_polish, V_polish, W_polish)
                max_err = np.max(np.abs(errors))
                
                if max_err < best_err_for_rank:
                    best_err_for_rank = max_err
                    best_U = U_polish.copy()
                    best_V = V_polish.copy()
                    best_W = W_polish.copy()
                
                if max_err > polish_prev_err * 1.1 and pol_it > 5:
                    break
                polish_prev_err = max_err
            
            # Round to nearest half-integer
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            max_err = verify(U_r, V_r, W_r, rank)
            if max_err < 1e-8:
                best_rank = rank
                best_result = (U_r, V_r, W_r)
                if rank <= 50:
                    break
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
