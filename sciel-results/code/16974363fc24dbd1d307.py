import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p # 96
    
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # Construct target tensor T with shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1.0
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Precompute flattened views for faster ALS updates
    # We need to solve for U, V, W given the others.
    # T_ijk = sum_r U_ir V_jr W_kr
    # This is equivalent to:
    # T_flat_U (size nm x (mp*pn)) = U @ (V x W)
    # Where (V x W) is the Khatri-Rao product of V and W, reshaped.
    
    def verify(U, V, W, tol=1e-9):
        # S = np.einsum('ir,jr,kr->ijk', U, V, W)
        # To be faster, we can check the norm of the difference
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(S, T, atol=tol)

    def round_to_half_int(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None

    def try_als(rank, seed, max_iters=50, time_budget=10.0):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        start = time.time()
        for it in range(max_iters):
            if time.time() > start + time_budget or time.time() > deadline:
                break
                
            # Update U
            # M_U: rank x (m*p*p*n)
            # We want U @ M_U = T_flat_U
            # M_U[r, (j,k)] = V[j,r] * W[k,r]
            # T_flat_U[i, (j,k)] = T[i,j,k]
            
            # Efficient update using einsum and reshape
            # V_W = np.einsum('jr,kr->rjk', V, W).reshape(rank, m*p*p*n)
            # T_flat_U = T.reshape(n*m, m*p*p*n)
            # U_new = lstsq(V_W.T, T_flat_U.T).T
            
            # To save time, we can use a smaller subset of equations if rank is high? No, must be exact.
            
            V_W = np.einsum('jr,kr->rjk', V, W).reshape(rank, -1)
            T_flat_U = T.reshape(n * m, -1)
            try:
                U_new, _, _, _ = np.linalg.lstsq(V_W.T, T_flat_U.T, rcond=None)
                U = U_new.T
            except:
                return None

            U_W = np.einsum('ir,kr->rik', U, W).reshape(rank, -1)
            T_flat_V = np.transpose(T, (1, 0, 2)).reshape(m * p, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(U_W.T, T_flat_V.T, rcond=None)
                V = V_new.T
            except:
                return None

            U_V = np.einsum('ir,jr->rij', U, V).reshape(rank, -1)
            T_flat_W = np.transpose(T, (2, 0, 1)).reshape(p * n, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(U_V.T, T_flat_W.T, rcond=None)
                W = W_new.T
            except:
                return None
                
            if verify(U, V, W, tol=1e-9):
                return U, V, W
                
        if verify(U, V, W, tol=1e-9):
            return U, V, W
        return None

    best_rank = rank_schoolbook
    best_result = {
        "U": np.eye(n*m, rank_schoolbook)[:, :rank_schoolbook].tolist(), # Placeholder, will be overwritten
        "V": np.eye(m*p, rank_schoolbook)[:, :rank_schoolbook].tolist(),
        "W": np.eye(p*n, rank_schoolbook)[:, :rank_schoolbook].tolist()
    }
    
    # Initialize with schoolbook
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Try to find lower rank
    # Start from 95 down to 80
    # We allocate time dynamically.
    
    for rank in range(rank_schoolbook - 1, 79, -1):
        if time.time() > deadline - 5: # Stop if close to deadline
            break
            
        # Try a few restarts for this rank
        # Fewer restarts for lower ranks as they are harder and we have less time?
        # Actually, we want to find the *minimum* rank, but finding *any* lower rank improves score.
        # So we try to find the lowest rank possible within time.
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
                
            result = try_als(rank, seed, max_iters=30, time_budget=5.0)
            if result is not None:
                U, V, W = result
                # Try to round to half-integers for cleanliness, but not required if floats are exact enough
                rounded = round_to_half_int(U, V, W)
                if rounded is not None:
                    U, V, W = rounded
                else:
                    # Verify again to be sure
                    if not verify(U, V, W, tol=1e-9):
                        continue
                        
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break # Found a valid decomposition at this rank, try next lower rank
        
        # If we found a decomposition at this rank, we continue to see if we can go lower.
        # If not, we continue to next rank.

    return best_result
