import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 20s safety margin
    
    # Target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # a in [0, nm), b in [0, mp), c in [0, pn)
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W, rank):
        # Check if sum_r U_r (x) V_r (x) W_r == T
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # We can compute the tensor efficiently?
        # T_recon = np.einsum('ar, br, cr -> abc', U, V, W)
        # This might be slow for large rank.
        # But rank is small (< 75).
        # nm=15, mp=25, pn=15.
        # 15*25*15 = 5625 entries.
        # Loop over r:
        T_recon = np.zeros((n * m, m * p, p * n), dtype=int)
        for r in range(rank):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.array_equal(T_recon, T)

    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook

    # Try to find lower rank
    # We will try ranks from 74 down to 50
    # Use random sparse integer matrices
    
    for target_rank in range(rank_schoolbook - 1, 49, -1):
        if time.time() > deadline:
            break
            
        # Number of trials per rank
        trials = 200
        
        for t in range(trials):
            if time.time() > deadline:
                break
            
            # Generate random sparse matrices
            # Density: schoolbook has 1 non-zero per column.
            # Let's try density 0.2 - 0.5
            density = 0.3
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            W = np.zeros((p * n, target_rank), dtype=int)
            
            # Randomly place non-zeros
            # For each column, pick a few indices
            for r in range(target_rank):
                # U
                num_nonzero_u = np.random.choice([1, 2], p=[0.7, 0.3])
                indices_u = np.random.choice(n * m, num_nonzero_u, replace=False)
                U[indices_u, r] = np.random.choice([-1, 1], num_nonzero_u)
                
                # V
                num_nonzero_v = np.random.choice([1, 2], p=[0.7, 0.3])
                indices_v = np.random.choice(m * p, num_nonzero_v, replace=False)
                V[indices_v, r] = np.random.choice([-1, 1], num_nonzero_v)
                
                # W
                num_nonzero_w = np.random.choice([1, 2], p=[0.7, 0.3])
                indices_w = np.random.choice(p * n, num_nonzero_w, replace=False)
                W[indices_w, r] = np.random.choice([-1, 1], num_nonzero_w)
            
            if verify(U, V, W, target_rank):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break # Found valid for this rank, try lower
        
        # If we found a valid one for this rank, continue to lower ranks
        # If not, continue to next rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
