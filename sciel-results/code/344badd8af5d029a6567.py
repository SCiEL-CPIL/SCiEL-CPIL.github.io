import time
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
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
    
    # Try different rank values with a structured approach
    # For each candidate rank, use a different initialization strategy
    
    # Known lower bounds: for 5x5x5, Strassen-like methods might help, but 125 is schoolbook.
    # We try to find a decomposition with rank < 125.
    
    # Candidate ranks: We know rank >= n^2 = 25. 
    # Let's try a range of ranks, prioritizing those that might be achievable.
    # 5x5x5 matrix multiplication tensor rank is unknown, but likely < 125.
    # Let's try 50, 60, 70, 80, 90, 100, 110.
    
    candidate_ranks = [50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 120]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        if rank >= rank_schoolbook:
            continue
        
        # Strategy: Use a structured initialization based on blocks
        # For 5x5x5, try to exploit the fact that 5 = 2+3
        
        rng = np.random.RandomState(rank * 7 + 42)
        
        # Try multiple initialization strategies
        best_err_for_rank = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for init_seed in range(10):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(init_seed * 1000 + rank)
            
            # Initialize with structured patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Strategy 1: Block-based initialization
            # Divide the rank into blocks corresponding to (i,j) or (j,k) or (k,i) pairs
            # Or just random initialization with small integers/halves
            
            if init_seed < 5:
                # Structured: map rank indices to (i,j,k) tuples
                for r in range(rank):
                    i_idx = r % n
                    j_idx = (r // n) % m
                    k_idx = (r // (n*m)) % p
                    
                    # Create a structured factor
                    U[i_idx*m+j_idx, r] = 1.0
                    V[j_idx*p+k_idx, r] = 1.0
                    W[k_idx*n+i_idx, r] = 1.0
                    
                    # Add some randomness to break symmetry
                    if init_seed > 0:
                        U[i_idx*m+j_idx, r] = 1.0 + rng.randn() * 0.1
                        V[j_idx*p+k_idx, r] = 1.0 + rng.randn() * 0.1
                        W[k_idx*n+i_idx, r] = 1.0 + rng.randn() * 0.1
            else:
                # Random initialization with small integers/halves
                # This might help escape local minima
                U = np.random.choice([0, 0.5, 1, -0.5, -1], size=(n*m, rank))
                V = np.random.choice([0, 0.5, 1, -0.5, -1], size=(m*p, rank))
                W = np.random.choice([0, 0.5, 1, -0.5, -1], size=(p*n, rank))
            
            # Alternating minimization
            prev_err = float('inf')
            for it in range(100):
                if time.time() > deadline - 5:
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
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < best_err_for_rank:
                    best_err_for_rank = max_err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if max_err < 1e-8:
                    break
                
                if max_err > prev_err * 1.5 and it > 10:
                    break
                prev_err = max_err
            
            if best_err_for_rank < 1e-6:
                break
        
        if best_err_for_rank < 1e-6 and best_U is not None:
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
