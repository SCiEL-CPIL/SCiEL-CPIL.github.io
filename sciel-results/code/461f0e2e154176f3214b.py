import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the constraint matrix: for each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a multilinear system. Let's try to find a low-rank solution.
    
    # First, return the schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank solutions using random initialization + optimization
    # The constraint is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using random restarts."""
        nonlocal best_result, best_rank
        
        if rank >= best_rank:
            return False
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return False
            
            # Random initialization
            U = [[float(np.random.choice([-1, 0, 1])) for _ in range(rank)] for _ in range(n*m)]
            V = [[float(np.random.choice([-1, 0, 1])) for _ in range(rank)] for _ in range(m*p)]
            W = [[float(np.random.choice([-1, 0, 1])) for _ in range(rank)] for _ in range(p*n)]
            
            # Normalize: for each column r, scale so product sums work
            # Actually, let's try a smarter approach: fix W to be identity-like
            # and solve for U, V
            
            # Alternative: use the fact that we can set W[k*n+i, r] = 1 for all
            # Then we need sum_r U[i*m+j,r]*V[j*p+k,r] = 1
            # This is a bilinear system.
            
            # Let's try: set W[r] to be a basis vector pattern
            # For rank r, let W[k*n+i, r'] = delta_{(k,i), some_pair(r')}
            
            # Actually, let me try a different approach: 
            # Fix W to have 1s in specific positions, then solve linear system for U,V
            
            # Strategy: For each rank, try structured decompositions
            # where W has a simple form (like permutation matrices or diagonal)
            
            # Try: W[k*n+i, r] = 1 if (k,i) matches pattern for r, else 0
            # Then for each r, we need U[i*m+j,r]*V[j*p+k,r] to contribute to the right places
            
            # Let's try a more systematic approach with numpy
            if rank <= 12:
                # Try random real-valued initialization and gradient-free search
                U = np.random.randn(n*m, rank) * 0.5
                V = np.random.randn(m*p, rank) * 0.5
                W = np.random.randn(p*n, rank) * 0.5
                
                # Simple iterative refinement
                for it in range(200):
                    if time.time() > deadline:
                        break
                    
                    # Update U: for each (i,j), we want sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all k
                    # This is linear in U for fixed V,W
                    for i in range(n):
                        for j in range(m):
                            # For fixed (i,j), we need: for all k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                            # This is a system in U[i*m+j, :]: 
                            # Let M_k[r] = V[j*p+k,r]*W[k*n+i,r]
                            # Then sum_r U[i*m+j,r]*M_k[r] = 1 for all k
                            # This is M^T * U[i*m+j,:] = 1 (vector of ones)
                            M = np.zeros((p, rank))
                            for k in range(p):
                                for r in range(rank):
                                    M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                            # Solve M^T * x = ones(p)
                            try:
                                x = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                                U[i*m+j, :] = x
                            except:
                                pass
                    
                    # Update V similarly
                    for j in range(m):
                        for k in range(p):
                            # For fixed (j,k), need: for all i, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                            # M_i[r] = U[i*m+j,r]*W[k*n+i,r]
                            M = np.zeros((n, rank))
                            for i in range(n):
                                for r in range(rank):
                                    M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                            try:
                                x = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                                V[j*p+k, :] = x
                            except:
                                pass
                    
                    # Update W similarly
                    for k in range(p):
                        for i in range(n):
                            # For fixed (k,i), need: for all j, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                            # M_j[r] = U[i*m+j,r]*V[j*p+k,r]
                            M = np.zeros((m, rank))
                            for j in range(m):
                                for r in range(rank):
                                    M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                            try:
                                x = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                                W[k*n+i, :] = x
                            except:
                                pass
                
                # Check
                if check_decomposition(U.tolist(), V.tolist(), W.tolist(), rank):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    return True
        
        return False
    
    # Try ranks from low to high
    for rank in range(1, best_rank):
        if time.time() > deadline:
            break
        if try_rank(rank):
            break
    
    return best_result
