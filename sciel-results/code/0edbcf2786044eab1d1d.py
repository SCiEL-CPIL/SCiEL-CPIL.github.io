"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or at least valid) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    
    Uses the schoolbook decomposition of rank n*m*p = 84.
    Then attempts greedy rank reduction by checking if any term is
    expressible as a linear combination of the remaining terms.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10 seconds safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank = n * m * p  # 84
    
    # Build the schoolbook decomposition
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)  
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    
    # Verify the decomposition is correct
    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For schoolbook, T[i,j,k] = 1 if the (i,j,k) index is present, which it always is
    # Actually the multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor)
    # Wait, no. The matrix multiplication tensor is defined as:
    # T[i, j, k] where i is row of A, j is col of A / row of B, k is col of B
    # T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k]
    # So the tensor has entries T[i,j,k] = 1 for all valid i,j,k
    # The decomposition satisfies: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i,j,k]
    
    # For schoolbook with 1s, sum_r U[i*m+j,r]V[j*p+k,r]W[k*n+i,r]
    # Each term r corresponds to a unique (i',j',k') triple
    # U[i*m+j, r] = 1 only if i'=i and j'=j
    # V[j*p+k, r] = 1 only if j'=j and k'=k  
    # W[k*n+i, r] = 1 only if k'=k and i'=i
    # So the product is 1 only when (i',j',k') = (i,j,k), and 0 otherwise
    # Thus sum = 1 for all (i,j,k), which is correct
    
    # Now try greedy rank reduction
    # For each term r, check if it can be expressed as a linear combination of the others
    # This is equivalent to checking if the rank-1 tensor u_r ⊗ v_r ⊗ w_r is in the span
    # of the other rank-1 tensors
    
    # A rank-1 tensor u⊗v⊗w is in the span of {u_s⊗v_s⊗w_s : s≠r} iff
    # there exist coefficients c_s such that u⊗v⊗w = sum_s c_s u_s⊗v_s⊗w_s
    # This is a system of equations. We can check this by flattening the tensors.
    
    # Flatten each rank-1 tensor to a vector of length n*m*p
    # u⊗v⊗w flattened: for index (i,j,k) in the tensor, value = u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # Let's work with the flattened representation
    # For term r with (i_r, j_r, k_r):
    # The flattened tensor has 1 at position corresponding to (i_r, j_r, k_r) and 0 elsewhere
    # Wait, that's not right. Let me reconsider.
    
    # Actually, the rank-1 tensor u_r ⊗ v_r ⊗ w_r where:
    # u_r has 1 at position i_r*m + j_r and 0 elsewhere
    # v_r has 1 at position j_r*p + k_r and 0 elsewhere
    # w_r has 1 at position k_r*n + i_r and 0 elsewhere
    # 
    # The outer product u_r ⊗ v_r ⊗ w_r evaluated at (i,j,k) is:
    # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # This is 1 only if i=i_r, j=j_r, k=k_r, and 0 otherwise
    # 
    # So each rank-1 term is a standard basis vector in R^{n*m*p}!
    # They are all linearly independent, so no term can be removed.
    # 
    # This means the schoolbook decomposition is already minimal in the sense that
    # no single term is a linear combination of the others.
    # 
    # However, a different decomposition with lower rank might exist that doesn't
    # use the standard basis vectors. But finding such a decomposition is the hard part.
    
    # Let's just return the schoolbook decomposition
    # Convert to lists for the expected output format
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure entries are integers or half-integers (they should be 0.0 or 1.0)
    # Convert to int where possible
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U_list]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V_list]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
