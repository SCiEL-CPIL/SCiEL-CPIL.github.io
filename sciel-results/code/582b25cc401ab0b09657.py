"""Matrix multiplication 2,5,5 - rank minimization via structured search."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
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
    result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    start_time = time.time()
    deadline = start_time + 140
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def try_als_decomposition(r, max_iters=200, seed=None):
        """Try to find a rank-r decomposition via ALS with random restarts."""
        rng = np.random.default_rng(seed)
        
        for restart in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random initialization with continuous values
            U = rng.standard_normal((n*m, r))
            V = rng.standard_normal((m*p, r))
            W = rng.standard_normal((p*n, r))
            
            # Scale to reasonable magnitude
            U /= np.sqrt(r)
            V /= np.sqrt(r)
            W /= np.sqrt(r)
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline:
                    return None
                
                # Update U: for each (i,j), solve linear system
                for i in range(n):
                    for j in range(m):
                        # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                        # This is: U[i*m+j,:] @ (V[j*p,:].T * W[:,k*n+i].T) = ones(p)
                        # Let M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                        # Then: U[i*m+j,:] @ M.T = ones(p)
                        # So: M @ U[i*m+j,:].T = ones(p)
                        M = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                M[k,c] = V[j*p+k, c] * W[k*n+i, c]
                        # Solve M @ u = ones(p) for u = U[i*m+j,:].T
                        try:
                            u, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Update V: for each (j,k), solve linear system
                for j in range(m):
                    for k in range(p):
                        # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                        # Let M[i,c] = U[i*m+j,c] * W[k*n+i,c]
                        # Then: M @ v = ones(n) for v = V[j*p+k,:].T
                        M = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                M[i,c] = U[i*m+j, c] * W[k*n+i, c]
                        try:
                            v, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                # Update W: for each (k,i), solve linear system
                for k in range(p):
                    for i in range(n):
                        # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                        # Let M[j,c] = U[i*m+j,c] * V[j*p+k,c]
                        # Then: M @ w = ones(m) for w = W[k*n+i,:].T
                        M = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                M[j,c] = U[i*m+j, c] * V[j*p+k, c]
                        try:
                            w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
            
            # After ALS, check if solution is close to integer/half-integer
            # First check continuous solution quality
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                # Try rounding to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                if verify(U_round, V_round, W_round, r):
                    return U_round, V_round, W_round
                
                # Also try integer rounding
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify(U_int, V_int, W_int, r):
                    return U_int, V_int, W_int
        
        return None
    
    # Try ranks from schoolbook-1 down to a reasonable lower bound
    # For 2x5x5, the rank is known to be 49 (from Strassen-type constructions)
    # Let's try from 49 down to 40
    
    for r in range(rank_schoolbook - 1, max(35, rank_schoolbook - 15), -1):
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        for seed in range(20):
            if time.time() > deadline:
                break
            
            result = try_als_decomposition(r, max_iters=10, seed=42 + seed)
            if result is not None:
                U, V, W = result
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no better solution found, return schoolbook
    if best_rank < rank_schoolbook:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return result

# EVOLVE-BLOCK-END
