import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook decomposition (rank 24)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(dim_U)]
    V_sb = [[0] * rank_sb for _ in range(dim_V)]
    W_sb = [[0] * rank_sb for _ in range(dim_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Known rank for 2x3x4 is 19. We will try to find it.
    # Strategy:
    # 1. Try small integer sets {-1, 0, 1} with random restarts.
    # 2. If not found, try {-2, -1, 0, 1, 2} with random restarts.
    # 3. If still not found, return schoolbook.
    
    # Pre-compute indices for the 24 non-zero entries of T
    # This allows for faster verification
    t_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_indices.append((i * m + j, j * p + k, k * n + i))
    
    def check_solution(U, V, W):
        # U, V, W are numpy arrays of shape (dim, rank)
        # Check if sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c] for all a,b,c
        # Since T is sparse (only 24 non-zeros), we only need to check those.
        # But we also need to ensure that for all OTHER (a,b,c), the sum is 0.
        # This is the hard part. 
        # However, if we assume the solution is "tight" or we just check the support first, 
        # we might miss errors. 
        # A full check is O(rank * dim_U * dim_V * dim_W) which is 24 * 6 * 12 * 8 = 13824 ops per check.
        # This is very fast.
        
        # Let's do a full check using numpy broadcasting for speed.
        # Construct the tensor from the decomposition
        # T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a 6x12x8 tensor.
        # Then check T_decomp == T
        
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if not np.array_equal(T_decomp, T):
            return False
        return True

    # Search strategy
    # We will iterate through candidate ranks starting from 19 up to 23.
    # For each rank, we try several random initializations.
    
    # To make the search more effective, we can use a heuristic:
    # The entries of U, V, W in known optimal decompositions for small matrices are often small integers.
    
    # Let's try rank 19 first, as it is the known minimum.
    # If we find a solution for 19, we stop.
    
    ranks_to_try = [19, 20, 21, 22, 23]
    
    # Define sets of values to try
    # Start with small integers
    val_sets = [
        [-1, 0, 1],
        [-2, -1, 0, 1, 2],
        [-3, -2, -1, 0, 1, 2, 3]
    ]
    
    # We will use a time-boxed search.
    # For each rank, we try a certain number of random restarts.
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try different value sets
        for v_set in val_sets:
            if time.time() > deadline - 10:
                break
                
            # Number of attempts
            # We can afford roughly 1000-5000 attempts per second with numpy checks.
            # Let's try 2000 attempts per value set/rank combination if time permits.
            max_attempts = 2000
            
            for _ in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                # Random initialization
                # Using random integers from the set
                U = np.array([[random.choice(v_set) for _ in range(rank)] for _ in range(dim_U)], dtype=int)
                V = np.array([[random.choice(v_set) for _ in range(rank)] for _ in range(dim_V)], dtype=int)
                W = np.array([[random.choice(v_set) for _ in range(rank)] for _ in range(dim_W)], dtype=int)
                
                # Quick check: 
                # The trace of the multiplication for each (i,j,k) must be 1.
                # We can check this faster than full tensor equality if we want, but einsum is fast enough.
                
                if check_solution(U, V, W):
                    # Found a solution!
                    # Convert to lists of lists as required by interface
                    best_rank = rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    return {"U": best_U, "V": best_V, "W": best_W}
    
    # If no lower rank found, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
