import time
import numpy as np

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
    
    def try_decomposition(rank, max_iters=150, seed=0):
        """Alternating minimization with structured initialization."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Structured initialization: exploit block structure
        # For 5x5x5, use a Kronecker-like structure
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Try different initialization strategies
        if rank >= 25:
            # Block-diagonal style: assign rank to (i,j,k) blocks
            block_size = rank // (n*m*p) if rank >= n*m*p else 1
            if block_size >= 1:
                # Dense initialization
                U = rng.randn(n*m, rank) * (1.0 / np.sqrt(rank))
                V = rng.randn(m*p, rank) * (1.0 / np.sqrt(rank))
                W = rng.randn(p*n, rank) * (1.0 / np.sqrt(rank))
            else:
                U = rng.randn(n*m, rank) * 0.3
                V = rng.randn(m*p, rank) * 0.3
                W = rng.randn(p*n, rank) * 0.3
        else:
            U = rng.randn(n*m, rank) * 0.3
            V = rng.randn(m*p, rank) * 0.3
            W = rng.randn(p*n, rank) * 0.3
        
        prev_err = float('inf')
        best_local = None
        best_local_err = float('inf')
        
        for it in range(max_iters):
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
            
            if max_err < best_local_err:
                best_local_err = max_err
                best_local = (U.copy(), V.copy(), W.copy(), max_err)
            
            if max_err < 1e-8:
                return U, V, W, max_err
            
            if max_err > prev_err * 1.2 and it > 20:
                break
            prev_err = max_err
        
        if best_local is not None and best_local[3] < 1e-6:
            return best_local
        return None
    
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
    
    # Try ranks from low to high, with more seeds for lower ranks
    candidate_ranks = list(range(20, min(55, rank_schoolbook)))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        # More seeds for lower ranks (harder to find)
        num_seeds = 10 if rank < 30 else 5
        
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            result = try_decomposition(rank, max_iters=120, seed=seed + rank*1000)
            if result is not None:
                U, V, W, err = result
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                max_err = verify(U_r, V_r, W_r, rank)
                if max_err < 1e-8:
                    best_rank = rank
                    best_result = (U_r, V_r, W_r)
                    break
        
        if best_rank < rank_schoolbook:
            if best_rank <= 25:
                break
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
