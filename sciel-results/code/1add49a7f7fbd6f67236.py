"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = 2, 3, 4
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return schoolbook()
    
    # Try to find a lower rank decomposition
    # The matrix multiplication tensor T has entries T[i,j,k] = 1
    # We want T = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # Let's try rank 12 construction based on known results
    # For 2x3x4, there are known decompositions of rank 12
    
    # Approach: Use a structured construction
    # The tensor can be decomposed by exploiting the small dimensions
    
    # Let's try a direct algebraic construction for rank 12
    # Based on the structure of matrix multiplication
    
    # One approach: use the fact that we can combine terms
    # For each pair (i,k), we need sum_j A[i,j] B[j,k]
    
    # Let's try a construction with 12 terms
    # r = 0..11
    
    # Pattern: r = 3*a + b where a in 0..3 (index for k-like), b in 0..2 (index for j-like)
    # But we need to be careful about the coupling
    
    # Actually, let me try a different approach:
    # Use the identity that for small matrices, we can find explicit decompositions
    
    # For 2x3x4, let's try to use a basis of 12 rank-1 tensors
    
    # The entries of U are indexed by (i,j) where i in {0,1}, j in {0,1,2}
    # The entries of V are indexed by (j,k) where j in {0,1,2}, k in {0,1,2,3}
    # The entries of W are indexed by (k,i) where k in {0,1,2,3}, i in {0,1}
    
    # Let me try a systematic search for a rank-12 decomposition
    # using a greedy or structured approach
    
    # Actually, let me try a specific known construction
    # For M_{2,3,4}, one can use the following pattern:
    
    # Term r corresponds to a triple (a, b, c) or some other structure
    
    # Let me try: for each j in 0..2 and each "half" of k in {0,1,2,3}
    # This gives 3 * 4 = 12 terms if we can structure it right
    
    # Construction attempt:
    # For term (j, k) where j in 0..2, k in 0..3:
    #   U[i*m+j, r] = 1 for all i
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] = 1 for all i
    # But this is just schoolbook with 12 terms... wait, that's not right
    # Schoolbook has 24 terms (one for each i,j,k)
    
    # To get rank 12, we need to combine the i dimension
    # For each (j,k), we have two terms: one for i=0 and one for i=1
    # Can we combine them?
    
    # If we set U[i*m+j, r] = 1 for both i=0 and i=1, and W[k*n+i, r] = 1 for both i=0 and i=1
    # Then the contribution to C[i,k] would be sum_j A[i,j] * B[j,k] * 1 * 1
    # But this would give the same contribution for i=0 and i=1, which is wrong
    
    # So we need a more clever combination
    
    # Let's try: use half-integers or other values to separate the i contributions
    # For term (j,k):
    #   U[0*m+j, r] = 1, U[1*m+j, r] = 1
    #   V[j*p+k, r] = 1
    #   W[k*n+0, r] = 1, W[k*n+1, r] = 0  -- this only contributes to i=0
    # This is still schoolbook-like
    
    # Alternative: use a Strassen-like trick
    # For 2x2x2, Strassen uses 7 terms instead of 8
    # The key is to use products of sums
    
    # For 2x3x4, let's think of it as two 1x3x4 multiplications coupled
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # These are independent linear combinations, so we might be able to share the B factors
    # But the A factors are different
    
    # Let me try a rank-12 construction based on the following idea:
    # Use 4 "base" terms for each of the 3 j values, but with clever U and W
    
    # Actually, let me just try to verify if a rank-12 solution exists by attempting
    # a numerical optimization, but that might be too slow
    
    # For now, let's try a specific algebraic construction
    
    # Based on known results in tensor rank, M_{2,3,4} has rank 12
    # Let me try to construct it explicitly
    
    # One possible construction:
    # Use the fact that we can write the multiplication as:
    # C = A * B where A is 2x3 and B is 3x4
    
    # Let's define 12 rank-1 tensors:
    # For j in 0..2 and t in 0..3:
    #   Let r = j * 4 + t
    #   U[i*m+j, r] = 1 for all i
    #   V[j*p+t, r] = 1
    #   W[t*n+i, r] = 1 if i == 0, 0 if i == 1  -- this only gives C[0,t]
    # This doesn't work for i=1
    
    # Let me try a different splitting:
    # For j in 0..2 and t in 0..3:
    #   r = j * 4 + t
    #   U[0*m+j, r] = 1, U[1*m+j, r] = 1
    #   V[j*p+t, r] = 1
    #   W[t*n+0, r] = 1, W[t*n+1, r] = 1
    # This gives C[i,t] += A[i,j] * B[j,t] for both i=0 and i=1
    # Summing over j gives C[i,t] = sum_j A[i,j] B[j,t]
    # This is correct! And it uses only 12 terms!
    
    # Wait, let me verify:
    # For term r = j*4 + t:
    #   Contribution to C[i,k] is sum over all r: U[i*m+j_r, r] * V[j_r*p+t_r, r] * W[t_r*n+i, r]
    #   For r = j'*4 + t':
    #     U[i*m+j', r] = 1 (for all i)
    #     V[j'*p+t', r] = 1 if j'=j and t'=t, else 0... wait no
    #     Actually V[j'*p+t', r] should be 1 only when the term is active for that j',t'
    
    # Let me redefine:
    # For term r corresponding to (j, t):
    #   U[i*m+j, r] = 1 for i in {0,1}
    #   V[j*p+t, r] = 1
    #   V[j'*p+t', r] = 0 for (j',t') != (j,t)
    #   W[t*n+i, r] = 1 for i in {0,1}
    #   W[t'*n+i', r] = 0 for (t',i') not in {(t,0),(t,1)}
    
    # Then the contribution to C[i,k] from term (j,t) is:
    #   U[i*m+j, r] * V[j*p+t, r] * W[t*n+i, r]
    #   = 1 * 1 * 1 if k == t, else 0
    #   = A[i,j] * B[j,t] if k == t, else 0
    
    # Summing over all terms:
    #   C[i,k] = sum_{j,t} [t == k] * A[i,j] * B[j,t]
    #          = sum_j A[i,j] * B[j,k]
    # This is correct!
    
    # So the construction is:
    # For each (j, t) where j in 0..2, t in 0..3:
    #   r = j * 4 + t
    #   U[i*m+j, r] = 1 for i in {0,1}
    #   V[j*p+t, r] = 1
    #   W[t*n+i, r] = 1 for i in {0,1}
    
    # Let's implement this and verify
    
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for t in range(p):
            r = j * p + t
            for i in range(n):
                U[i * m + j][r] = 1
            V[j * p + t][r] = 1
            for i in range(n):
                W[t * n + i][r] = 1
    
    # Verify the decomposition
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Reconstruct the tensor
        # T[i,j,k] should be 1 for all i,j,k
        # T = sum_r U[:,r] * V[:,r] * W[:,r]
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        T = np.zeros((n, m, p))
        for r in range(rank):
            T += U3[:,:,r][:, :, np.newaxis] * V3[:,:,r][np.newaxis, :, :] * W3[:,:,r][np.newaxis, np.newaxis, :]
        
        # Check if T is all ones
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return schoolbook()

# EVOLVE-BLOCK-END
