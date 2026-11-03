"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Returns a dictionary with 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook rank is n*m*p = 4*4*9 = 144.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find a lower rank decomposition using random search
    # We'll try various ranks from 143 down to a reasonable minimum
    for target_rank in range(rank - 1, 100, -1):
        if time.time() > deadline:
            break
            
        # Try random initialization with integer/half-integer entries
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Generate random U, V, W with integer/half-integer entries
            U_trial = np.zeros((n * m, target_rank))
            V_trial = np.zeros((m * p, target_rank))
            W_trial = np.zeros((p * n, target_rank))
            
            # Randomly set some entries to 1, -1, or 0.5, -0.5
            for r in range(target_rank):
                # U entries
                num_nonzero_u = random.randint(1, 4)
                rows_u = random.sample(range(n * m), num_nonzero_u)
                for row in rows_u:
                    U_trial[row, r] = random.choice([1, -1, 0.5, -0.5])
                
                # V entries
                num_nonzero_v = random.randint(1, 4)
                rows_v = random.sample(range(m * p), num_nonzero_v)
                for row in rows_v:
                    V_trial[row, r] = random.choice([1, -1, 0.5, -0.5])
                
                # W entries
                num_nonzero_w = random.randint(1, 4)
                rows_w = random.sample(range(p * n), num_nonzero_w)
                for row in rows_w:
                    W_trial[row, r] = random.choice([1, -1, 0.5, -0.5])
            
            # Check if this decomposition is valid
            # The decomposition is valid if for all i, j, k:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i,j,k are "connected" in the matrix multiplication
            # Actually, for matrix multiplication, the tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor)
            # Wait, no. For matrix multiplication C = A * B where A is n x m and B is m x p,
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # The tensor T[i,j,k] has T[i,j,k] = 1 for all valid i,j,k.
            # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U_trial[i*m+j, r] * V_trial[j*p+k, r] * W_trial[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_result = {
                    "U": U_trial.tolist(),
                    "V": V_trial.tolist(),
                    "W": W_trial.tolist()
                }
                break  # Found a valid decomposition at this rank, move to next lower rank
    
    return best_result
# EVOLVE-BLOCK-END
