import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Pre-compute indices for the 60 tensor entries
    # We need to check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    idx_u = np.zeros(n * m * p, dtype=np.int32)
    idx_v = np.zeros(n * m * p, dtype=np.int32)
    idx_w = np.zeros(n * m * p, dtype=np.int32)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u[idx] = i * m + j
                idx_v[idx] = j * p + k
                idx_w[idx] = k * n + i
                idx += 1

    def check_decomposition_fast(U, V, W):
        """
        Vectorized check using pre-computed indices.
        U: (nm, R), V: (mp, R), W: (pn, R)
        """
        # Advanced indexing: U[idx_u] has shape (60, R)
        # Element-wise multiply and sum over R
        # This is faster than einsum for this specific small structure
        vals = U[idx_u] * V[idx_v] * W[idx_w]
        sums = vals.sum(axis=1)
        return np.all(sums == 1.0)

    # Strategy:
    # 1. Try to find a rank < 60 using fast random search.
    # 2. If found, return it.
    # 3. Otherwise, return schoolbook (rank 60).
    
    # Theoretical lower bound is 30.
    # We start from 59 down to 30.
    # We allocate time based on the rank. Higher ranks are "closer" to schoolbook 
    # in terms of degrees of freedom vs constraints, but random search is still unlikely.
    # However, if a solution exists, it might have small integer coefficients.
    
    # Let's try to be aggressive with the search.
    # Total time budget ~ 110s.
    # We will try each rank for a fixed number of attempts, or until time runs out.
    
    # Estimate: check_decomposition_fast takes ~5-10 microseconds for R=30?
    # Let's benchmark mentally: 60 * 30 = 1800 multiplications, sum, check. Very fast.
    # Maybe 100,000 attempts per second?
    # So in 100 seconds, we can do 10,000,000 attempts per rank.
    # That is a lot.
    
    # Instead of random search, we can try to find a solution by solving a system of equations
    # for small ranks, or use a more structured search.
    
    # For now, let's try a more efficient random search with larger batches and early exit.
    
    def try_rank_random_batch(R, num_batches, batch_size):
        """Try random search for a valid decomposition of rank R using batches."""
        for _ in range(num_batches):
            if time.time() > deadline:
                return None
            
            # Generate a batch of random matrices
            U_batch = np.random.randint(-2, 3, size=(batch_size, nm, R))
            V_batch = np.random.randint(-2, 3, size=(batch_size, mp, R))
            W_batch = np.random.randint(-2, 3, size=(batch_size, pn, R))
            
            # Check each in the batch
            # U_batch[:, idx_u] -> (batch, 60, R)
            U_sel = U_batch[:, idx_u]
            V_sel = V_batch[:, idx_v]
            W_sel = W_batch[:, idx_w]
            
            vals = U_sel * V_sel * W_sel
            sums = vals.sum(axis=2) # (batch, 60)
            
            # Find indices where all 60 sums are 1
            valid_mask = np.all(sums == 1.0, axis=1)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                idx_valid = valid_indices[0]
                return U_batch[idx_valid], V_batch[idx_valid], W_batch[idx_valid]
        
        return None

    for R in range(59, 29, -1):
        if time.time() > deadline - 2:
            break
        
        # Allocate time for this rank
        time_per_rank = 1.0
        start_rank_time = time.time()
        
        # Use batch search
        batch_size = 100
        num_batches = int(time_per_rank * 1000000 / (batch_size * 60 * R * 1e-6)) # Rough estimate
        num_batches = max(num_batches, 100)
        
        result = try_rank_random_batch(R, num_batches, batch_size)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if time.time() > deadline - 5:
            break

    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
