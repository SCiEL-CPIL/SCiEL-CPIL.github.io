import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Precompute indices for the tensor identity
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # We need this to hold for all i, j, k in {0, 1, 2}
    
    # Fallback Rank 27
    U_fb = [[0]*27 for _ in range(9)]
    V_fb = [[0]*27 for _ in range(9)]
    W_fb = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    # Try to find a lower rank solution with entries in {-1, 0, 1}
    # We'll try ranks 20 to 26
    for rank in range(20, 27):
        if time.time() - start_time > time_limit:
            break
            
        # Random search with some local improvement?
        # Just random sampling is very slow.
        # Let's try a few hundred random samples.
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random -1, 0, 1
            U = np.random.choice([-1, 0, 1], size=(9, rank))
            V = np.random.choice([-1, 0, 1], size=(9, rank))
            W = np.random.choice([-1, 0, 1], size=(9, rank))
            
            # Check identity
            # Reshape to tensors
            # U: 3x3xR, V: 3x3xR, W: 3x3xR
            U_t = U.reshape(3, 3, rank)
            V_t = V.reshape(3, 3, rank)
            W_t = W.reshape(3, 3, rank)
            
            # Compute T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Note: W is indexed by k, i. So W_t[k, i, r].
            # Let's use einsum
            # U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
            # Sum over r.
            # Result shape: (3, 3, 3)
            
            # Efficient check:
            # Multiply U and V first?
            # M[j, i, k] = sum_r U[i, j, r] V[j, k, r] ? No, indices are tricky.
            
            # Let's just loop for simplicity in the check, it's small (27 checks).
            valid = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        val = 0
                        # u_idx = i*3 + j
                        # v_idx = j*3 + k
                        # w_idx = k*3 + i
                        # Use numpy dot products for speed?
                        # val = np.dot(np.dot(U[i*3+j, :], V[j*3+k, :]), W[k*3+i, :])
                        # But we need element-wise product sum.
                        # val = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                        val = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        # If not found for this rank, try next (higher) rank?
        # The loop goes 20 to 26. If 20 fails, try 21, etc.
        
    return {
        "U": U_fb,
        "V": V_fb,
        "W": W_fb
    }
