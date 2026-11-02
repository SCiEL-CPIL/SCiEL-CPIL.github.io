import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    def verify_fast(U, V, W):
        """Fast vectorized verification using einsum."""
        U_mat = U.reshape(n, m, -1)  # (n, m, rank)
        V_mat = V.reshape(m, p, -1)  # (m, p, rank)
        W_mat = W.reshape(p, n, -1)  # (p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        max_err = float(np.max(np.abs(T - 1.0)))
        return max_err
    
    def alternating_minimize(U, V, W, max_iters=200, tol=1e-10, deadline=145.0):
        """Alternating minimization with proper convergence."""
        rank = U.shape[1]
        prev_err = float('inf')
        stuck_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            max_err = verify_fast(U, V, W)
            
            if max_err < tol:
                return U, V, W, max_err
            
            if max_err < prev_err * 0.999:
                stuck_count = 0
            else:
                stuck_count += 1
            
            if stuck_count > 30 and it > 50:
                break
            prev_err = max_err
        
        return U, V, W, verify_fast(U, V, W)
    
    def polish(U, V, W, max_iters=100, tol=1e-12, deadline=145.0):
        """High-precision polish pass."""
        rank = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            for i in range(n):
                for j in range(m):
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
            
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
            
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
            
            max_err = verify_fast(U, V, W)
            if max_err < tol:
                break
        
        return U, V, W, verify_fast(U, V, W)
    
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
    best_result = (U_school, V_school, W_school)
    
    # Try various ranks with structured initializations
    # Use tensor slicing: initialize with patterns that respect tensor structure
    candidate_ranks = list(range(25, 80))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        found_valid = False
        max_seeds = 15 if rank < 40 else 8
        
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(seed + rank * 1000)
            
            # Structured initialization: use block patterns
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            
            # Add structured components: for each (i,j,k), add a small rank-1 component
            for i in range(min(n, 3)):
                for j in range(min(m, 3)):
                    for k in range(min(p, 3)):
                        c = (i*m+j)*p+k
                        if c < rank:
                            U[i*m+j, c] += 1.0
                            V[j*p+k, c] += 1.0
                            W[k*n+i, c] += 1.0
            
            result = alternating_minimize(U, V, W, max_iters=150, tol=1e-8, deadline=deadline)
            if result is not None:
                U_r, V_r, W_r, err = result
                if err < 1e-6:
                    # Round to nearest half-integer
                    U_rounded = np.round(U_r * 2) / 2
                    V_rounded = np.round(V_r * 2) / 2
                    W_rounded = np.round(W_r * 2) / 2
                    max_err = verify_fast(U_rounded, V_rounded, W_rounded)
                    if max_err < 1e-8:
                        best_rank = rank
                        best_result = (U_rounded, V_rounded, W_rounded)
                        found_valid = True
                        break
        
        if found_valid and best_rank <= 35:
            break
    
    # Polish the best result
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        polished = polish(U.copy(), V.copy(), W.copy(), max_iters=50, tol=1e-12, deadline=deadline)
        if polished is not None:
            U_p, V_p, W_p, err_p = polished
            U_pr = np.round(U_p * 2) / 2
            V_pr = np.round(V_p * 2) / 2
            W_pr = np.round(W_p * 2) / 2
            max_err = verify_fast(U_pr, V_pr, W_pr)
            if max_err < 1e-8:
                best_result = (U_pr, V_pr, W_pr)
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
