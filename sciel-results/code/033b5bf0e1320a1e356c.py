import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds for search
    
    # Allowed values for entries
    # Integer or half-integer
    # Let's try a small set: 0, 1, -1, 0.5, -0.5
    # Maybe 2, -2, 1.5, -1.5 if needed?
    # Start with {0, 1, -1}
    
    def compute_error(U, V, W):
        # U: (n*m, r)
        # V: (m*p, r)
        # W: (p*n, r)
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want T[i, j, k] == 1 for all i, j, k
        
        # Reshape for easier indexing?
        # U_reshaped: (n, m, r)
        # V_reshaped: (m, p, r)
        # W_reshaped: (p, n, r)
        
        # We can compute the tensor T of shape (n, m, p)
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        
        # Efficient computation:
        # For each c, we have u_c (n*m), v_c (m*p), w_c (p*n)
        # This is a rank-1 tensor u_c \otimes v_c \otimes w_c?
        # No, the indices are coupled.
        
        # Let's just loop over i, j, k and sum over c.
        # n*m*p = 32. r is up to 31.
        # 32 * 31 = 992 operations. Very fast.
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(len(U[0])):
                        val += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    err += (val - 1.0)**2
        return err

    def try_find_solution(rank):
        # Randomized local search
        # U, V, W are lists of lists
        # We will use numpy arrays for speed in error calc?
        # No, the loop is small. Python lists are fine.
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        allowed_vals = [0, 1, -1, 0.5, -0.5]
        
        for attempt in range(100): # 100 restarts
            if time.time() > deadline:
                return None
            
            # Initialize with zeros
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            
            # Randomly set some entries
            # Sparsity: let's try to set ~30% of entries
            num_entries_U = int(0.3 * n * m * rank)
            num_entries_V = int(0.3 * m * p * rank)
            num_entries_W = int(0.3 * p * n * rank)
            
            for _ in range(num_entries_U):
                i = random.randint(0, n*m - 1)
                c = random.randint(0, rank - 1)
                U[i][c] = random.choice(allowed_vals)
            
            for _ in range(num_entries_V):
                i = random.randint(0, m*p - 1)
                c = random.randint(0, rank - 1)
                V[i][c] = random.choice(allowed_vals)
            
            for _ in range(num_entries_W):
                i = random.randint(0, p*n - 1)
                c = random.randint(0, rank - 1)
                W[i][c] = random.choice(allowed_vals)
            
            current_error = compute_error(U, V, W)
            
            # Local search
            for step in range(1000):
                if time.time() > deadline:
                    return None
                
                if current_error == 0:
                    return U, V, W
                
                # Pick a random matrix and entry
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                    size = n * m
                elif mat_idx == 1:
                    mat = V
                    size = m * p
                else:
                    mat = W
                    size = p * n
                
                i = random.randint(0, size - 1)
                c = random.randint(0, rank - 1)
                
                old_val = mat[i][c]
                new_val = random.choice(allowed_vals)
                
                if old_val == new_val:
                    continue
                
                mat[i][c] = new_val
                new_error = compute_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                    # Keep new_val
                else:
                    # Revert
                    mat[i][c] = old_val
                    
                if current_error < best_error:
                    best_error = current_error
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                    
        # If we didn't find 0, return None for this rank
        # Or return the best found if it's very small?
        # No, must be exact.
        return None

    # Try ranks from 31 down to 25
    for r in range(31, 24, -1):
        if time.time() > deadline:
            break
        
        result = try_find_solution(r)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # If no better solution found, return schoolbook
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
