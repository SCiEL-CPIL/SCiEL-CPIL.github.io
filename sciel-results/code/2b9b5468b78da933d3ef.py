import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 72
    schoolbook_rank = n * m * p
    U_school = np.zeros((n * m, schoolbook_rank), dtype=object)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=object)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = schoolbook_rank
    
    def check_decomposition(U, V, W):
        """Check if the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=object)
        V = np.array(V, dtype=object)
        W = np.array(W, dtype=object)
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if s != 1:
                        return False
        return True
    
    # Try to find lower-rank decompositions using structured approaches
    # Strategy: exploit the block structure - 8 columns of B share the same A
    
    # Approach: For each of the 8 columns, we need a 3x3x1 decomposition.
    # If we could find a rank-7 decomposition for 3x3x1, total would be 56.
    # But 3x3x1 has rank 9, so this won't work directly.
    
    # Alternative: use linearity. If we write B = sum of rank-1 matrices B_t,
    # then AB = sum_t A B_t. Each A B_t where B_t is 3x1 can be done in rank 9.
    # But that's 8*9=72 again.
    
    # Let's try a direct random search for rank < 72
    random.seed(12345)
    
    def try_random_decomposition(target_rank, num_trials=50):
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            U = np.zeros((n * m, target_rank), dtype=object)
            V = np.zeros((m * p, target_rank), dtype=object)
            W = np.zeros((p * n, target_rank), dtype=object)
            
            # Initialize with sparse random values
            for r in range(target_rank):
                # Each rank-1 term should contribute to some (i,j,k) pattern
                # Use a structured approach: pick random (i,j,k) and assign
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                
                # Assign values from a small set
                u_val = random.choice([1, -1, 1, -1, 2, -2])
                v_val = random.choice([1, -1, 1, -1, 2, -2])
                w_val = random.choice([1, -1, 1, -1, 2, -2])
                
                U[i * m + j, r] = u_val
                V[j * p + k, r] = v_val
                W[k * n + i, r] = w_val
            
            if check_decomposition(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from 71 down to 60
    for target_rank in range(schoolbook_rank - 1, 59, -1):
        if time.time() > deadline:
            break
        
        result = try_random_decomposition(target_rank, num_trials=20)
        if result is not None:
            U_list, V_list, W_list = result
            best_U = U_list
            best_V = V_list
            best_W = W_list
            best_rank = target_rank
            break  # Found a valid decomposition, stop
    
    # If no improvement found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
