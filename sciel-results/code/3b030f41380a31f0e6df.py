import time
import numpy as np
import random

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Define the target tensor indices (i, j, k) -> (a, b, c)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
                
    # Initialize Schoolbook Decomposition
    # Rank R = n * m * p
    R0 = n * m * p
    U = np.zeros((nm, R0), dtype=float)
    V = np.zeros((mp, R0), dtype=float)
    W = np.zeros((pn, R0), dtype=float)
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col_idx] = 1.0
                V[j * p + k, col_idx] = 1.0
                W[k * n + i, col_idx] = 1.0
                col_idx += 1
                
    best_U, best_V, best_W, best_R = U, V, W, R0
    
    # Heuristic Search for Rank Reduction
    # We try to remove columns if the tensor constraint is still satisfied.
    # This is a simple greedy approach.
    R = R0
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    # Function to check if the current decomposition is valid
    def is_valid(U_mat, V_mat, W_mat, rank):
        # Check all target indices
        for (a, b, c) in target_indices:
            # Sum U[a,r] * V[b,r] * W[c,r] for r in range(rank)
            # Use dot product for speed
            # u = U_mat[a, :rank]
            # v = V_mat[b, :rank]
            # w = W_mat[c, :rank]
            # s = np.sum(u * v * w)
            s = np.sum(U_mat[a, :rank] * V_mat[b, :rank] * W_mat[c, :rank])
            if abs(s - 1.0) > 1e-9:
                return False
        return True

    # Greedy removal
    # Try removing each column, check validity, if valid, keep removed
    # To avoid O(R * N_target * R) complexity being too high, we limit iterations
    # N_target = 52. R=52.
    # Check cost: 52 * 52 * 52 ~ 140k ops. Very fast.
    # We can try all columns in random order
    
    columns_to_try = list(range(R0))
    random.shuffle(columns_to_try)
    
    current_rank = R0
    U_work = U_curr
    V_work = V_curr
    W_work = W_curr
    
    for col in columns_to_try:
        if time.time() > deadline:
            break
        if col >= current_rank:
            continue
            
        # Try zeroing out this column
        # Since we are removing, we effectively set it to 0 and then slice later
        # But to check, we can just check if the sum is 0 for this column?
        # No, the sum is over all r.
        # If we remove column col, the new sum is old_sum - U[a,col]*V[b,col]*W[c,col].
        # We need new_sum = 1.
        # So we need U[a,col]*V[b,col]*W[c,col] = 0 for all (a,b,c) in target?
        # No, the other columns must compensate.
        # So we must check the full sum.
        
        U_test = U_work.copy()
        V_test = V_work.copy()
        W_test = W_work.copy()
        
        U_test[:, col] = 0
        V_test[:, col] = 0
        W_test[:, col] = 0
        
        if is_valid(U_test, V_test, W_test, current_rank):
            # Valid, remove the column
            # Slice out the column
            U_work = np.delete(U_work, col, axis=1)
            V_work = np.delete(V_work, col, axis=1)
            W_work = np.delete(W_work, col, axis=1)
            current_rank -= 1
            best_U, best_V, best_W, best_R = U_work, V_work, W_work, current_rank
            
    # Convert to lists as required by interface? 
    # Interface: {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), ...
    # The current program returns lists. I will return lists to be safe, or numpy arrays if accepted.
    # The prompt says "integer/half-integer entries". Numpy floats are fine if they are exact.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
