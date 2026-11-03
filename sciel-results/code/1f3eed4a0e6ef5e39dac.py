"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for matrix multiplication.
    Starts with the schoolbook rank-50 solution and attempts to reduce rank via
    random perturbation and local optimization.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 50
    
    # Initialize with schoolbook solution
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists for return
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    best_result = {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    best_rank = rank_schoolbook
    
    # Try to reduce rank
    deadline = time.time() + 110  # Safety margin
    
    for target_rank in range(40, rank_schoolbook):  # Try ranks from 40 to 49
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U_r = np.random.randn(n * m, target_rank)
            V_r = np.random.randn(m * p, target_rank)
            W_r = np.random.randn(p * n, target_rank)
            
            # Normalize
            for c in range(target_rank):
                norm_u = np.linalg.norm(U_r[:, c])
                norm_v = np.linalg.norm(V_r[:, c])
                norm_w = np.linalg.norm(W_r[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    U_r[:, c] /= norm_u
                    V_r[:, c] /= norm_v
                    W_r[:, c] /= norm_w
            
            # Check if this random initialization is close to a valid decomposition
            # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # For matrix multiplication, T[i,j,k] = 1 if i,j,k are valid indices, else 0
            # Actually, the tensor for matrix multiplication is defined such that
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # The tensor T has T[i,j,k] = 1 for all valid i,j,k (since each term contributes)
            
            # Let's verify the decomposition
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1  # All entries of the multiplication tensor are 1
                        actual = sum(U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c] for c in range(target_rank))
                        err = abs(expected - actual)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                # Found a valid decomposition with lower rank
                best_rank = target_rank
                best_result = {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
