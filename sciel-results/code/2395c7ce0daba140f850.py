import time
import random
import copy
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal (or near-minimal) exact tensor decomposition 
    for multiplying a 3x3 matrix by a 3x5 matrix.
    
    The interface requires returning {'U', 'V', 'W'} with shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Entries must be integers or half-integers.
    Score is the rank (lower is better).
    
    Strategy:
    1. Start with the schoolbook decomposition (Rank = n*m*p = 45).
    2. Perform a randomized search for lower rank decompositions.
       - Try ranks from 44 down to 30.
       - For each rank, generate random matrices U, V, W with integer/half-integer entries.
       - Check if the tensor decomposition is exact.
       - If a valid lower rank is found, return it.
    3. If no lower rank is found, return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard: 150s limit, we return well before.
    deadline = time.time() + 115  # 5s safety margin
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def check_decomposition(U, V, W, tol=1e-10):
        """
        Check if U, V, W form an exact tensor decomposition.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        rank = U.shape[1]
        
        # Reshape to tensors
        # U_tensor[i, j, r] = U[i*m + j, r]
        U_tensor = U.reshape(n, m, rank)
        # V_tensor[j, k, r] = V[j*p + k, r]
        V_tensor = V.reshape(m, p, rank)
        # W_tensor[k, i, r] = W[k*n + i, r]
        W_tensor = W.reshape(p, n, rank)
        
        # For each (i, j, k), check if sum_r U[i,j,r] * V[j,k,r] == W[k,i,r]
        # This is equivalent to checking if the tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]
        # matches the target tensor which has T[i,j,k] = 1 if we're decomposing identity-like,
        # but actually the matrix multiplication tensor is defined such that:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor M[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
        # which is 1 for all i,j,k.
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] == 1 for all i,j,k
        # And W[k*n+i, r] should be the "output" part. Wait, the standard tensor decomposition
        # for matrix multiplication is:
        # T = sum_r U_r \otimes V_r \otimes W_r
        # where T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] ? No.
        
        # Let's re-read the task contract:
        # "exactly decompose the matrix multiplication tensor"
        # The matrix multiplication tensor T is defined by T[i,j,k] = 1 for all i,j,k
        # (since C[i,k] = sum_j A[i,j]B[j,k]).
        
        # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r
        # This means T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        
        # Let's compute this sum for all i,j,k and check if it equals 1.
        
        # Efficient check using einsum or loops
        # Since dimensions are small, loops are fine.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    for rank in range(n * m * p - 1, 29, -1):  # Try from 44 down to 30
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        num_attempts = 50
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Generate random matrices with integer/half-integer entries
            # Use small values to keep numbers manageable
            scale = 2.0
            U = np.round(np.random.uniform(-scale, scale, (n * m, rank)) * 2.0) / 2.0
            V = np.round(np.random.uniform(-scale, scale, (m * p, rank)) * 2.0) / 2.0
            W = np.round(np.random.uniform(-scale, scale, (p * n, rank)) * 2.0) / 2.0
            
            if check_decomposition(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook decomposition
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
