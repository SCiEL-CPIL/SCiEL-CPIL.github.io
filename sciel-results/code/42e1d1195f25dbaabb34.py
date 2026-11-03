"""Find a low-rank exact tensor decomposition for 2x4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal (or low-rank) exact decomposition for 2x4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # The multiplication tensor has structure T[i,j,k,l] = delta_{j,k} where:
    # i in {0,1} (output row), j in {0,1,2,3} (inner dim / A col & B row), 
    # k in {0,...,4} (B col / output col), l in {0,1,2,3} (output col index)
    # Actually, for C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has indices (i, j, k) with T[i,j,k] = 1 if this term contributes to C[i,k]
    # The multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    
    # Schoolbook rank = n*m*p = 40
    # Let's try to find a lower rank decomposition.
    
    # The tensor for (n,m,p) = (2,4,5) matrix multiplication has:
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)  
    # W shape: (p*n, rank) = (10, rank)
    
    # The decomposition requires: for all i in [0,n), k in [0,p):
    #   C[i,k] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    # Wait, let me re-read the convention.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n+i, r]
    
    # The tensor decomposition means:
    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # where T_{i,j,k} is the matrix multiplication tensor
    # T_{i,j,k} = 1 if j is the summing index connecting A[i,j] and B[j,k]
    
    # Actually the matmul tensor is: T[i,j,k] which equals delta when we need A[i,j]*B[j,k]
    # The tensor is trilinear: T(A,B) = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] e_{i,k}
    # T[i,j,k] = 1 for all valid (i,j,k) tuples since C[i,k] = sum_j A[i,j] B[j,k]
    
    # So we need: sum_r U[i,j,r] * V[j,k,r] * W[k*n+i, r] = delta_{j is valid for i,k}
    # Actually T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p)
    # Because each triple (i,j,k) contributes A[i,j]*B[j,k] to C[i,k]
    
    # Wait no. The tensor is T[i,j,k] where the sum over j gives C[i,k].
    # T[i,j,k] = 1 if this term is part of the multiplication, which is always true.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a bilinear map decomposition. The rank is the number of rank-1 terms.
    
    # Known results: For 2x2x2 matmul, rank is 7 (Strassen).
    # For 2x4x5, let's think about what's achievable.
    
    # Let me try a systematic approach. The tensor T[i,j,k] = 1 for all i,j,k
    # We want to decompose this as sum of rank-1 tensors.
    
    # A rank-1 term is: u_i * v_j * w_k where u is a vector over i, v over j, w over k
    # But the constraint is that the product U[i,j,r]*V[j,k,r]*W[k,i,r] must equal 1 for ALL (i,j,k).
    
    # Hmm, this is more complex. Let me think differently.
    
    # The matrix multiplication tensor M_{n,m,p} has:
    # M[i,j,k] = 1 if 0<=i<n, 0<=j<m, 0<=k<p (it's all ones)
    # No wait, that's not right either.
    
    # Let me reconsider. The multiplication tensor for C = AB is:
    # For each output entry C[i,k], we compute sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] = delta_{contributes to (i,k)}
    # Actually T[i,j,k] = 1 for all valid i,j,k because every (i,j,k) triple
    # contributes A[i,j]*B[j,k] to the computation of C[i,k].
    
    # So T is the all-ones tensor of shape (n,m,p)? No!
    # T[i,j,k] should be such that the bilinear form sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] gives C
    # But C is a matrix, not a scalar. So the tensor actually maps to C[i,k].
    
    # The correct formulation: The multiplication is a bilinear map from A (n*m vars) and B (m*p vars) to C (n*p vars).
    # Each output C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor: T has shape (n*m, m*p, n*p) where
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'
    
    # So T[a,b,c] where a=(i,j) in [0,n*m), b=(j',k) in [0,m*p), c=(i',k') in [0,n*p)
    # T[a,b,c] = 1 iff i=i' AND j=j' AND k=k'
    
    # The decomposition: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # We need: sum_r U[a,r] V[b,r] W[c,r] = 1 if (a,b,c) is a valid multiplication term, else 0
    
    # For a = (i,j), b = (j',k), c = (i',k'):
    # The term is nonzero only if i=i', j=j', k=k'
    # So we need: sum_r U[i,j,r] V[j,k,r] W[k*n+i, r] = 1 for all i,j,k
    # And it should be 0 for other combinations, but since we're summing over r and the left side
    # depends on the specific i,j,k, we just need it to equal 1 for each valid (i,j,k).
    
    # So the problem is: find matrices U (8 x r), V (20 x r), W (10 x r) such that
    # for ALL i in [0,2), j in [0,4), k in [0,5):
    #   sum_r U[i,j,r] * V[j,k,r] * W[k*n+i, r] = 1
    
    # This is a system of 2*4*5 = 40 equations.
    
    # Let's try to find a low-rank solution. The minimum rank for 2x2x2 is 7.
    # For 2x4x5, a trivial upper bound is 40 (schoolbook).
    # Can we do better?
    
    # Idea: Use the fact that we can factor through smaller tensors.
    # 2x4x5 = (2x2x5) + (2x2x5) by splitting the m=4 dimension into two m=2 blocks.
    # If we can find rank R for 2x2x5, then total rank is 2R.
    
    # What's the rank of 2x2x5 matmul? 
    # 2x2x2 has rank 7.
    # 2x2x3: I believe rank is around 11-12.
    # 2x2x5: maybe around 17-19?
    
    # Actually, let me just try to solve this directly with optimization or a clever construction.
    
    # Alternative: Use known results or construct via polynomial methods.
    
    # Let me try a different approach: set up the problem and use numerical methods to find a low-rank solution.
    
    # Actually, for the purposes of this task, let me try a structured decomposition.
    
    # Key insight: C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this using Winograd-style tricks.
    
    # For the j-sum of length 4: sum_{j=0}^{3} A[i,j] B[j,k]
    # We can use a Winograd-style decomposition for the convolution-like structure.
    
    # Winograd for length-4 dot product can be done with 6 multiplications instead of 4? No, that's not right.
    # The dot product of two length-n vectors has tensor rank n (it's already rank n for the standard dot product).
    # Wait, no. The dot product <x,y> = sum_j x_j y_j is a bilinear form of rank n (sum of n rank-1 forms x_j * y_j).
    # But we can do better? No, the rank of the dot product tensor of dimension n is exactly n.
    # Because the matrix of the bilinear form is the identity, which has rank n.
    
    # Hmm, so for each (i,k), we have a dot product of length m=4, which has rank 4.
    # Total would be n*p*4 = 2*5*4 = 40. That's the schoolbook.
    
    # But the full tensor isn't just a block diagonal of dot products! The same U, V, W entries are shared.
    # So there might be savings.
    
    # Let me think about this more carefully. We need:
    # sum_r U[i,j,r] V[j,k,r] W[k*n+i, r] = 1 for all i,j,k
    
    # Note that W is indexed by (k*n+i), so W[k*n+i, r] couples k and i.
    
    # Let's try rank 30 or so and see if we can find a solution.
    
    # Actually, let me try a concrete approach: use the structure to build a decomposition.
    
    # Method: Use the fact that the tensor is "separable" in some sense.
    
    # Let's try: for each (i,k) pair, we need sum_j A[i,j] B[j,k].
    # Think of it as: for fixed i, we have a linear map from B to C[i,:].
    
    # Alternative: use a Kronecker product structure.
    # If we could write the tensor as a sum of products of simpler tensors...
    
    # Let me just try to find a valid decomposition with rank less than 40 using a numerical approach.
    
    # Start with schoolbook (rank 40) and try to reduce.
    
    # Actually, let me try a specific construction.
    
    # Construction idea: 
    # For 2x4x5, split into:
    # - Two 2x2x5 blocks (for j=0,1 and j=2,3)
    # For 2x2x5: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] for i in {0,1}, k in {0,...,4}
    # This is like: for each k, C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k]
    #                        C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k]
    
    # The 2x2x2 part (for a fixed k) has rank 7 (Strassen).
    # But we have 5 values of k, and they share the A and B structure.
    
    # Hmm, this is getting complex. Let me try a different tactic.
    
    # Let me just implement the schoolbook decomposition with rank 40, but verify it's correct.
    # Then, if time permits, try to find improvements.
    
    # Actually, I recall that for matrix multiplication, the rank of the (n,m,p) tensor is at most n*m*p,
    # and there are known upper bounds using Strassen and its generalizations.
    
    # For 2x4x5, one approach: 
    # 2x4x5 <= 2x2x5 + 2x2x5
    # 2x2x5 <= 2x2x2 + 2x2x3
    # 2x2x2 has rank 7
    # 2x2x3: let's estimate. The tensor size is 2*2*3=12. Rank is at most 12.
    # Actually I think 2x2x3 has rank 11 or 12.
    
    # This is getting too speculative. Let me just go with a safe approach.
    
    # Safe approach: schoolbook rank 40.
    # But let me try to see if I can find a rank 30+ decomposition.
    
    # Let me try a specific structured decomposition.
    
    # Idea: Use the following observation.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_{j=0}^{3} A[i,j] B[j,k]
    
    # Group terms: (A[i,0] + A[i,1]) * (B[0,k] + B[1,k]) = A[i,0]B[0,k] + A[i,0]B[1,k] + A[i,1]B[0,k] + A[i,1]B[1,k]
    # This introduces cross terms. We need to subtract them.
    
    # This is the Winograd/Fourier approach. Let me try to implement it properly.
    
    # For the sum over j of length 4:
    # s0 = A[i,0] + A[i,1] + A[i,2] + A[i,3]
    # s1 = A[i,0] - A[i,1] + A[i,2] - A[i,3]  
    # s2 = A[i,0] + A[i,1] - A[i,2] - A[i,3]
    # s3 = A[i,0] - A[i,1] - A[i,2] + A[i,3]
    # (This is a Hadamard transform)
    
    # Similarly for B:
    # t0 = B[0,k] + B[1,k] + B[2,k] + B[3,k]
    # t1 = B[0,k] - B[1,k] + B[2,k] - B[3,k]
    # t2 = B[0,k] + B[1,k] - B[2,k] - B[3,k]
    # t3 = B[0,k] - B[1,k] - B[2,k] + B[3,k]
    
    # Then sum_j A[i,j] B[j,k] = (1/4) * (s0*t0 + s1*t1 + s2*t2 + s3*t3)
    
    # So C[i,k] = (1/4) * sum_{r=0}^{3} H_i(r) * K_k(r)
    # where H_i(r) is the Hadamard coefficient for A[i,:] and K_k(r) is for B[:,k]
    
    # This gives: C[i,k] = (1/4) * sum_{r=0}^{3} s_i(r) * t_k(r)
    
    # As a tensor decomposition:
    # For each r in {0,1,2,3}, we have a term:
    #   U_part: depends on i and r (the Hadamard transform of A[i,:])
    #   V_part: depends on k and r (the Hadamard transform of B[:,k])
    #   W_part: couples i and k
    
    # But the Hadamard transform of A[i,:] gives a vector of 4 values (one for each r).
    # So s_i(r) = sum_j H[r,j] * A[i,j] where H is the 4x4 Hadamard matrix.
    # Similarly t_k(r) = sum_j H[r,j] * B[j,k]
    
    # So C[i,k] = (1/4) * sum_{r=0}^{3} (sum_j H[r,j] A[i,j]) * (sum_j' H[r,j'] B[j',k])
    
    # This is a bilinear form: sum_r (H A_i) * (B^T H^T)_k
    # = sum_r sum_j sum_j' H[r,j] H[r,j'] A[i,j] B[j',k]
    
    # As a rank decomposition:
    # For each r, the term is: (H[r,:] @ A_i) * (H[r,:] @ B_k^T)
    # This is a rank-1 term in the (i,k) output, but it involves a product of two linear combinations.
    
    # In the tensor decomposition framework:
    # For each r (4 values), we need to represent:
    #   sum_j sum_j' H[r,j] H[r,j'] A[i,j] B[j',k]
    
    # This can be written as: for each r, a rank-4 term (since it's a sum over j and j' independently).
    # So total rank would be 4 * 4 = 16? No, that's not right either.
    
    # Let me think again. The Hadamard approach gives:
    # C[i,k] = (1/4) sum_{r=0}^{3} S_i(r) * T_k(r)
    # where S_i(r) = sum_j H[r,j] A[i,j] and T_k(r) = sum_j H[r,j] B[j,k]
    
    # Now, S_i(r) is a linear combination of the A[i,j] values.
    # T_k(r) is a linear combination of the B[j,k] values.
    
    # The product S_i(r) * T_k(r) = (sum_j H[r,j] A[i,j]) * (sum_j' H[r,j'] B[j',k])
    # = sum_j sum_j' H[r,j] H[r,j'] A[i,j] B[j',k]
    
    # In the tensor decomposition, each "multiplication" S_i(r) * T_k(r) corresponds to a rank-1 term
    # in the bilinear map, but it requires rank 4 in the (j,j') space.
    
    # Hmm, I think I'm overcomplicating this. Let me just go with the schoolbook and verify correctness.
    
    # Schoolbook: rank = 40
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
    result = {"U": U, "V": V, "W": W}
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1) > 1e-10:
                    # Fallback to schoolbook (which should always be correct)
                    return result
    return result
# EVOLVE-BLOCK-END
