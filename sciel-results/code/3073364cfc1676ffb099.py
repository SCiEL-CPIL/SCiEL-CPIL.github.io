import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135  # safety margin before 150s kill

    # Build target tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W, rank):
        T_recon = np.zeros((n * m, m * p, p * n), dtype=int)
        for r in range(rank):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.array_equal(T_recon, T)

    # Schoolbook decomposition
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

    # Strategy: Try structured decompositions based on Strassen-like ideas
    # Partition the 3x5 matrix into blocks and use lower-rank tensor decompositions
    
    # First, try a direct sum approach:
    # Split the 3 rows into 2+1, and use:
    # - A 2x5x5 part with rank R1
    # - A 1x5x5 part with rank R2 (schoolbook = 25, or potentially lower)
    # Total rank = R1 + R2
    
    # For the 1x5x5 part (scalar times 5x5 matrix times 5x5 matrix),
    # this is just element-wise product, so rank = 25 (schoolbook)
    # Actually, for 1x5x5: T[a,b,c] where a=j, b=j*p+k, c=k*n+i=5k+0
    # This is the standard matrix multiplication tensor for 5x5, rank at most 25
    
    # For the 2x5x5 part, we can try to use Strassen-like methods on 2x2 blocks
    # but with 5 columns, it's more complex
    
    # Let's try a different approach: use known rank bounds
    # For 3x5x5, the rank is at most 75 (schoolbook)
    # Lower bounds: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # Strassen for 2x2x2 gives rank 7 instead of 8
    
    # Try to find a decomposition with rank < 75 using structured random search
    # Focus on ranks from 70 down to 50
    
    # Use a more systematic search with structured matrices
    # Each column of U, V, W should be sparse (few non-zeros)
    
    for target_rank in range(74, 49, -1):
        if time.time() > deadline:
            break
        
        # Number of trials per rank - adapt based on time remaining
        remaining = deadline - time.time()
        trials = max(10, min(300, int(remaining * 5)))
        
        for t in range(trials):
            if time.time() > deadline:
                break
            
            # Generate structured sparse matrices
            # For each column r, pick a small number of non-zero entries
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            W = np.zeros((p * n, target_rank), dtype=int)
            
            for r in range(target_rank):
                # U: pick 1-3 non-zeros from 15 entries
                num_u = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                idx_u = np.random.choice(n * m, num_u, replace=False)
                U[idx_u, r] = np.random.choice([-1, 1], num_u)
                
                # V: pick 1-3 non-zeros from 25 entries
                num_v = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                idx_v = np.random.choice(m * p, num_v, replace=False)
                V[idx_v, r] = np.random.choice([-1, 1], num_v)
                
                # W: pick 1-3 non-zeros from 15 entries
                num_w = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                idx_w = np.random.choice(p * n, num_w, replace=False)
                W[idx_w, r] = np.random.choice([-1, 1], num_w)
            
            if verify(U, V, W, target_rank):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break  # Found valid for this rank, try lower
        
        # If we found a valid one for this rank, continue to lower ranks
        # If not, continue to next rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
