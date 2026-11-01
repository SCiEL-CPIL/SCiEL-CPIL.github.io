"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank: 42
    
    # Build schoolbook decomposition
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
    
    # Verify the decomposition is correct
    # The tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # should equal 1 if a=(i,j), b=(j',k), c=(k',i') with j=j', i=i', k=k'
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: for each (i,j,j',k,k',i'), the product should be 1 iff j=j' and i=i' and k=k'
    # This is expensive, so just spot-check a few
    def tensor_entry(a, b, c):
        return np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
    
    # Spot check
    # T[(i,j), (j,k), (k,i)] should be 1
    assert abs(tensor_entry(0, 0, 0) - 1.0) < 1e-10  # i=0,j=0, j'=0,k=0, k'=0,i'=0
    # T[(i,j), (j',k), (k',i')] should be 0 if j != j'
    assert abs(tensor_entry(0, 1, 0) - 0.0) < 1e-10  # j=0, j'=1
    
    # Try to reduce rank by checking for linear dependencies
    # If the rank-1 tensors are linearly dependent, we can reduce the rank
    
    # Reshape into a 3D tensor
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # The tensor has shape (n*m, m*p, p*n) = (6, 21, 14)
    # We want to find if there's a rank-r decomposition with r < 42
    
    # Approach: Try to find a non-trivial linear relation among the rank-1 terms
    # Each rank-1 term is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # If sum_r alpha_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0 for some nonzero alpha,
    # then we can reduce the rank.
    
    # This is equivalent to checking if the matrix whose columns are the 
    # vectorized rank-1 tensors has a non-trivial null space.
    
    # Vectorize each rank-1 term: (n*m*m*p*p*n) = 6*21*14 = 1764 dimensional
    # We have 42 such vectors, so the matrix is 1764 x 42
    # The null space will always be trivial since 42 < 1764, but the vectors might be dependent.
    
    # Actually, since we have 42 vectors in a 1764-dimensional space, they are generically independent.
    # But for this specific tensor, they might be dependent.
    
    # Let's check the rank of the matrix of vectorized rank-1 terms
    vec_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
        vec_terms.append(term.flatten())
    
    vec_matrix = np.array(vec_terms).T  # Shape (1764, 42)
    
    # Compute the rank
    try:
        mat_rank = np.linalg.matrix_rank(vec_matrix)
    except:
        mat_rank = rank
    
    if mat_rank < rank:
        # There are linear dependencies, so we can reduce the rank
        # Find a basis for the column space
        Q, R = np.linalg.qr(vec_matrix)
        # The first mat_rank columns of Q form a basis
        # But this doesn't directly give us a reduced decomposition
        # We need to find a new set of rank-1 terms that span the same tensor
        
        # For now, just note that the rank might be reducible
        # This is complex, so let's try a different approach
    
    # Alternative: Try specific structured reductions
    # For matrix multiplication, one known technique is to use the fact that
    # certain combinations of terms can be eliminated.
    
    # Let's try to see if we can reduce by 1 by finding a linear relation
    # among the 42 rank-1 terms.
    
    # Since the vectors are in a 1764-dim space and we have 42 of them,
    # they are almost certainly linearly independent, so the rank is 42.
    
    # Return the schoolbook decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
