"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition
    # For 2x5x6, try various ranks starting from a reasonable lower bound
    # The tensor rank of M(2,m,p) is known to be at most m*p + m + p - 2 in some constructions
    # Let's try rank = 40 first, then increase if needed
    
    def try_rank(target_rank):
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random integer/half-integer entries
        # Use a structured initialization based on the schoolbook pattern but with sharing
        
        # Strategy: For each (j,k) pair, we need contributions to both i=0 and i=1
        # We can share the V factor entries between the two rows
        
        # Let's try a block structure:
        # For each j in range(m), k in range(p):
        #   We have two equations: one for i=0, one for i=1
        #   U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r] should contribute to result[0,j]
        #   U[1*m+j, r] * V[j*p+k, r] * W[k*n+1, r] should contribute to result[1,j]
        
        # This is complex. Let me try a simpler approach:
        # Use the fact that for n=2, we can potentially use rank = m*p + 2*m + 2*p - 4 or similar
        
        # Actually, let me try a direct construction:
        # Rank = m*p + m + p - 2 = 30 + 5 + 6 - 2 = 39
        
        U = np.zeros((n * m, target_rank), dtype=float)
        V = np.zeros((m * p, target_rank), dtype=float)
        W = np.zeros((p * n, target_rank), dtype=float)
        
        # Build a structured decomposition
        # For each (j,k), assign a unique column for the "diagonal" part
        col = 0
        for j in range(m):
            for k in range(p):
                # This handles the case where we need U[i*m+j] * V[j*p+k] * W[k*n+i]
                # For i=0: U[j][col] * V[j*p+k][col] * W[k*n][col]
                # For i=1: U[m+j][col] * V[j*p+k][col] * W[k*n+1][col]
                
                # Set up so that this column contributes to both i=0 and i=1
                U[j][col] = 1
                U[m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n][col] = 1
                W[k*n+1][col] = 1
                col += 1
        
        # Now we have m*p = 30 columns. We need to add more to handle the cross terms.
        # The issue is that the above gives:
        # result[i,j] = sum_k U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For the above construction, for a fixed (i,j):
        # result[i,j] = sum_k 1 * 1 * 1 = p (for each k where V[j*p+k] is nonzero)
        # This is not correct.
        
        # Let me try a different approach. Use the schoolbook but try to merge columns.
        
        # Start with schoolbook
        U_s = np.zeros((n * m, n * m * p), dtype=float)
        V_s = np.zeros((m * p, n * m * p), dtype=float)
        W_s = np.zeros((p * n, n * m * p), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        
        # Now try to reduce the rank by finding linear dependencies
        # The tensor identity requires:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
        
        # This is a nonlinear system. Let me try a random projection approach.
        
        # Try random rank-r decompositions and check if they satisfy the constraints
        import random
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U_rand = np.random.randint(-2, 3, (n * m, target_rank)) / 2.0
            V_rand = np.random.randint(-2, 3, (m * p, target_rank)) / 2.0
            W_rand = np.random.randint(-2, 3, (p * n, target_rank)) / 2.0
            
            # Check if this satisfies the tensor identity
            # For each (i,j,k), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list format
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rand.tolist()
                return U_list, V_list, W_list
        
        return None
    
    # Try ranks from 39 upward
    for target_rank in range(39, 61):
        result = try_rank(target_rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
