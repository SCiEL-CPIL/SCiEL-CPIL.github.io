"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = n*m*p = 54
    rank_sb = n * m * p
    
    # Try to find a lower-rank decomposition using tensor network methods
    # For 2x3 * 3x9, the tensor is T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}
    # Actually T[i, j, k] = 1 if indices match, 0 otherwise
    # T is a 2x3x9 tensor where T[i,j,k] = 1 if j is the shared index
    
    # The tensor multiplication T for (n,m,p) has T[i,j,k] = 1 if the "middle" index matches
    # Specifically for matrix multiplication C = A * B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 (it's the identity in the middle dimension)
    
    # Known lower bounds:
    # For 2x3 * 3x9:
    # - Schoolbook: 54
    # - Can we do better?
    
    # Let's try a structured approach:
    # The 2x3 * 3x9 multiplication can be seen as:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # One approach: use the fact that n=2 is small
    # We can write this as a block operation
    
    # Let's try rank 36: use a 2x2 block structure
    # Actually, let me think about this more carefully.
    
    # For n x m * m x p, the tensor rank is at most n*m*p (schoolbook)
    # Lower bounds include max(n*p, n*m, m*p) = max(18, 6, 27) = 27
    
    # Let's try to achieve rank 27 or lower using structured decompositions
    
    # Strategy: Since n=2, we can potentially use the structure
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # This is essentially two independent vector-matrix multiplications of size 1x3 * 3x9
    # Each 1x3 * 3x9 has tensor rank 3 (since it's just scaling)
    # Wait, no. A 1x3 vector times a 3x9 matrix:
    # C[k] = sum_j v[j] * M[j,k]
    # This is a 1x3x9 tensor with T[j,k] = M[j,k]... 
    # Actually the tensor for 1x3 * 3x9 multiplication is T[j,k] where the "input" is v[j] and M[j,k]
    # The tensor is T[j, j', k] = delta_{j,j'} * 1... 
    # No, the tensor is T[j, k] for the bilinear form... 
    
    # Let me be more careful. The matrix multiplication tensor for n x m * m x p is:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor in the middle)
    
    # A rank-r decomposition gives:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[i,k,c]
    # Wait, the convention is W[k*n+i, r] so W is indexed by (k,i)
    
    # So: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For schoolbook: U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    
    # Let's try to find a lower rank using optimization
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, leave margin
    
    # Try various ranks from 27 up to 54
    # Use a simple iterative method
    
    # First, let's try a known good construction
    # For 2x3 * 3x9, I'll try rank 36
    
    # Actually, let me just return schoolbook for now since we need to ensure correctness
    # and the search may not find anything better in the time available
    
    rank = rank_sb
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
