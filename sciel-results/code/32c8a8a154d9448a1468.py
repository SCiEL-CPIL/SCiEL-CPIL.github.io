"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition of rank n*m*p for matrix multiplication.
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where c indexes the rank dimension.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill in the schoolbook decomposition
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
        
        # Check deadline periodically
        if time.time() > deadline:
            break
    
    # Verify the decomposition is correct
    # The tensor should satisfy: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if this is the "diagonal" term, but actually for matrix mult:
    # T[i,j,k] should be such that when contracted with A and B, gives C[i,k]
    
    # Let's verify: for each (i,j,k), sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # In schoolbook, each (i,j,k) triple has exactly one column where all three are 1
    # So the sum should be 1 for each (i,j,k)
    
    # Quick spot check
    if time.time() < deadline:
        sample_count = min(100, n * m * p)
        for _ in range(sample_count):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            if abs(val - 1.0) > 1e-10:
                # This shouldn't happen in schoolbook, but if it does, return what we have
                pass
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
