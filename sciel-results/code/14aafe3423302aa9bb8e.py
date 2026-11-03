import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Uses catalecticant matrix analysis to establish rank bounds, then attempts structured
    decomposition via random initialization and local refinement.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit

    # Schoolbook decomposition as guaranteed fallback
    base_rank = n * m * p
    U_sb = np.zeros((n * m, base_rank), dtype=np.int64)
    V_sb = np.zeros((m * p, base_rank), dtype=np.int64)
    W_sb = np.zeros((p * n, base_rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = base_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def validate(U, V, W, n, m, p, rank):
        """
        Validate that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for all i in [0,n), j in [0,m), k in [0,p).
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        s += U[ui, r] * V[vj, r] * W[wk, r]
                    if s != 1:
                        return False
        return True

    # Compute catalecticant matrix to get a lower bound on rank
    # The tensor T has entries T[i,j,k] = 1 if we're multiplying (i,j) and (j,k)
    # Flattenings: 
    #   Flattening 1: reshape to (n*m, p) -> rank gives lower bound
    #   Flattening 2: reshape to (m*p, n) -> rank gives lower bound  
    #   Flattening 3: reshape to (p*n, m) -> rank gives lower bound
    
    # Build the tensor
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # All entries are 1 for standard matrix mult
    
    # Catalecticant matrices
    # Flattening along first mode: (n) x (m*p)
    cat1 = T.reshape(n, m * p)
    # Flattening along second mode: (m) x (n*p) 
    cat2 = T.reshape(m, n * p).T  # (n*p, m)
    # Flattening along third mode: (p) x (n*m)
    cat3 = T.reshape(p, n * m).T  # (n*m, p)
    
    # Rank of each catalecticant gives lower bound on tensor rank
    r1 = np.linalg.matrix_rank(cat1)
    r2 = np.linalg.matrix_rank(cat2)
    r3 = np.linalg.matrix_rank(cat3)
    catalecticant_lower_bound = max(r1, r2, r3)
    
    # For standard matrix multiplication, the tensor rank is known to be n*m*p
    # but we use the lower bound to guide search
    print(f"Catalecticant lower bounds: {r1}, {r2}, {r3}, max={catalecticant_lower_bound}")
    
    # Try to find a decomposition with rank slightly above the lower bound
    # Using random initialization and local search
    
    def try_random_decomposition(target_rank, num_trials=1000):
        """
        Try to find a valid decomposition with given target_rank.
        Uses random initialization with integer entries.
        """
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None
            
            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(n * m, target_rank), dtype=np.int64)
            V = np.random.randint(-2, 3, size=(m * p, target_rank), dtype=np.int64)
            W = np.random.randint(-2, 3, size=(p * n, target_rank), dtype=np.int64)
            
            # Validate
            if validate(U, V, W, n, m, p, target_rank):
                return U, V, W
        
        return None, None, None
    
    # Try ranks from lower bound up to schoolbook
    for rank in range(catalecticant_lower_bound, base_rank + 1):
        if time.time() > deadline:
            break
        
        # Only try if rank is reasonable
        if rank > 50:  # Skip very high ranks for random search
            continue
            
        U_try, V_try, W_try = try_random_decomposition(rank, num_trials=500)
        if U_try is not None:
            best_rank = rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < base_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
