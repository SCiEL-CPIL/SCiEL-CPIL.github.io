"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook decomposition
    rank_sb = n * m * p  # 54
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    best_rank = rank_sb
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find lower rank via structured search
    # The tensor condition: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def verify(U_arr, V_arr, W_arr):
        """Verify the decomposition is exact."""
        r = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j] * V_arr[j*p+k], W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Try to construct a lower-rank solution by exploiting structure
    # For 3x3x6, we can view it as two 3x3x3 multiplications side by side
    # If we have a rank-r 3x3x3 algorithm, we get rank 2r for 3x3x6
    
    # Let's try to find a 3x3x3 algorithm of rank < 27
    # Known: Strassen gives 7 for 2x2x2, and 3x3x3 can be done in rank <= 23
    
    # Attempt: Use a block-based construction
    # Split 3x3 into 2x2 + 1x1 blocks
    
    # For now, try random search with integer/half-integer constraints
    # Scale by 2: search for integer entries, divide by 2 at the end
    # Condition becomes: sum U_int * V_int * W_int = 8 (since (U/2)(V/2)(W/2) = 1 => U*V*W = 8)
    
    def try_random_rank(r, max_attempts=50):
        """Try to find a valid decomposition of rank r with integer entries (scaled by 2)."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random integer entries in range [-2, 2] for scaled values
            U_int = np.random.randint(-2, 3, size=(n*m, r))
            V_int = np.random.randint(-2, 3, size=(m*p, r))
            W_int = np.random.randint(-2, 3, size=(p*n, r))
            
            # Check condition: sum U_int[i*m+j, c] * V_int[j*p+k, c] * W_int[k*n+i, c] = 8
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_int[i*m+j] * V_int[j*p+k] * W_int[k*n+i])
                        if val != 8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U_float = (U_int / 2.0).tolist()
                V_float = (V_int / 2.0).tolist()
                W_float = (W_int / 2.0).tolist()
                return {"U": U_float, "V": V_float, "W": W_float}
        
        return None
    
    # Try ranks from 53 down to 30
    for r in range(rank_sb - 1, 29, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_random_rank(r, max_attempts=20)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
