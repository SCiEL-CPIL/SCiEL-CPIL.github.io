import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for 2x4x7 matrix multiplication.
    Falls back to the schoolbook decomposition (rank 56) if a lower rank is not found within the time limit.
    """
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0

    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    
    # Construct the target multiplication tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Function to check if U, V, W form a valid decomposition
    def check_decomposition(U, V, W):
        # T_recon[a, b, c] = sum_l U[a, l] * V[b, l] * W[c, l]
        T_recon = np.einsum('al,bl,cl->abc', U, V, W)
        return np.allclose(T_recon, T_target, atol=1e-6)

    # Generate schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school), dtype=int)
    V_school = np.zeros((dim_v, rank_school), dtype=int)
    W_school = np.zeros((dim_w, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # Search for lower rank decompositions
    # We try ranks from rank_school - 1 down to some limit
    # Using random integer entries from {-1, 0, 1}
    # This is a heuristic search
    
    max_search_rank = 55
    min_search_rank = 50  # Don't go too low, probability is very low
    
    random.seed(42) # Fixed seed for reproducibility in evaluation if needed, but random is fine too
    
    for rank in range(max_search_rank, min_search_rank - 1, -1):
        if time.time() + safety_margin >= deadline:
            break
            
        # Number of attempts per rank
        # As rank decreases, probability of random success drops, but we can try a few
        attempts = 20 if rank > 53 else 5
        
        for _ in range(attempts):
            if time.time() + safety_margin >= deadline:
                break
                
            # Initialize with small random integers
            # Using sparse random matrices might help, but dense {-1,0,1} is a good start
            U = np.random.randint(-1, 2, (dim_u, rank))
            V = np.random.randint(-1, 2, (dim_v, rank))
            W = np.random.randint(-1, 2, (dim_w, rank))
            
            if check_decomposition(U, V, W):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                # If we find a very low rank, we can stop early? 
                # For now, we continue to see if we can go lower, but time is precious.
                # Breaking immediately is safer for time, but might miss a better rank.
                # Given the low probability of finding rank < 54 randomly, we might break if we find 55.
                # Let's break if we find rank <= 54 to save time for verification or just return.
                if rank <= 54:
                    return best_result

    # If we didn't find a better one, return the best found (which is schoolbook)
    return best_result
