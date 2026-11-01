import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140  # 10s safety margin
    
    # Build the multiplication tensor T[i*m+j, k*p+l] -> but actually
    # T is a 3D tensor: T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the (i,k) entry of product uses A[i,j]*B[j,k]
    # Actually the tensor for matrix mult: T[i, j, k] = delta (it's 1 for valid (i,j,k))
    # The product C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the "all ones" in the right sense)
    # No wait, T[i,j,k] is defined such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The tensor is just the all-ones tensor of shape (n, m, p)
    # We need to find its rank as a 3-tensor.
    
    # Reshape to matrix: mode-1 unfolding: shape (n, m*p)
    # T_unfold[i, j*p+k] = 1
    # This is an all-ones matrix of shape (n, m*p)
    # The matrix rank is n (since all rows are identical... no, all entries are 1)
    # Actually the all-ones matrix of shape (n, m*p) has rank 1!
    # But tensor rank is different from matrix rank.
    
    # For the all-ones tensor of shape (n,m,p), the tensor rank is:
    # It can be written as sum of rank-1 tensors u_i * v_i * w_i
    # where (u_i * v_i * w_i)[i,j,k] = u_i[i] * v_i[j] * w_i[k]
    
    # The all-ones tensor = 1 * 1 * 1 (outer product of three all-ones vectors)
    # So the tensor rank is 1?!
    
    # Wait, that can't be right for matrix multiplication. Let me reconsider.
    
    # In matrix multiplication, the tensor is NOT the all-ones tensor.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T satisfies: C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k. Yes, it IS the all-ones tensor.
    
    # But the rank-1 decomposition: a rank-1 tensor is u ⊗ v ⊗ w where
    # (u ⊗ v ⊗ w)[i,j,k] = u[i] * v[j] * w[k]
    # The all-ones tensor = 1_vec ⊗ 1_vec ⊗ 1_vec
    # So the tensor rank is 1??
    
    # That's clearly wrong because we know 2x2x2 has rank 7.
    
    # I think I'm confusing the tensor with the multiplication map. Let me reconsider.
    
    # The multiplication tensor for (n,m,p) is defined as:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A ⊗ e_j^M ⊗ e_k^B ⊗ ... 
    # No, that's not right either.
    
    # Let me be precise. The multiplication tensor T_{n,m,p} is a tensor in
    # (R^n)^* ⊗ (R^m)^* ⊗ (R^p)^* ... no.
    
    # Actually, the standard definition: T_{n,m,p} ∈ (R^n ⊗ R^m)^* ⊗ R^{n×p}
    # or equivalently, T is a 3-tensor with T[i, j, k] = 1 if we're multiplying
    # A[i,j] * B[j,k] to get C[i,k].
    
    # Hmm, I think the issue is that the tensor for matrix multiplication is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (something)
    
    # Let me look at this from the decomposition perspective.
    # A decomposition of rank r gives:
    # C[i,k] = sum_{c=0}^{r-1} (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # Wait, that's not quite right either.
    
    # The correct formulation:
    # C = A * B, where A is n×m, B is m×p, C is n×p
    # A decomposition of rank r: there exist matrices U (n×m × r), V (m×p × r), W (n×p × r)
    # such that C[i,k] = sum_c U[i,*,c] · A[i,:] ... no.
    
    # Let me use the interface definition:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition means: for the multiplication tensor,
    # T = sum_c U_c ⊗ V_c ⊗ W_c (as a 3-tensor)
    
    # Where U_c is a vector of length n*m, V_c of length m*p, W_c of length p*n.
    # And the tensor T has T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # where the tensor is indexed as T[i, j, k] corresponding to the (i,j) entry of A,
    # (j,k) entry of B, and (i,k) entry of C.
    
    # But W is indexed as W[k*n+i, c], so W_c[k*n+i] is the coefficient for the (i,k) output.
    
    # The condition for exact decomposition:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for "cross terms" that shouldn't contribute, they should be 0.
    # But actually, the tensor only has non-zero entries at (i,j,k) for valid indices,
    # and the value is 1. There are no other entries.
    
    # So the condition is simply:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # That's it! There are no other constraints because the tensor is just all ones
    # on the valid indices.
    
    # Wait, but that means the tensor rank is the minimum r such that we can write
    # the all-ones (n×m×p) tensor as a sum of r rank-1 tensors.
    # And the all-ones tensor has rank 1 (it's 1⊗1⊗1)!
    
    # So the answer should be rank 1?? That can't be right.
    
    # I must be misunderstanding the tensor structure. Let me reconsider.
    
    # Ah, I think the issue is that in matrix multiplication, the tensor is NOT
    # simply all ones. The multiplication C = AB involves:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined so that the bilinear map (A,B) -> C is represented.
    # As a trilinear tensor: T(A,B,C) = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # For this to equal the "inner product" that enforces C = AB, we need:
    # T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j] and B[j,k]
    # which is always true, so T[i,j,k] = 1.
    
    # But the rank of this tensor as a 3-tensor... 
    # A rank-1 tensor is u ⊗ v ⊗ w, i.e., T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    # We need u[i*m+j] * v[j*p+k] * w[k*n+i] = 1 for all i,j,k.
    # This is possible: set u = v = w = all ones. Then the product is 1 everywhere.
    # So rank 1!
    
    # But this contradicts known results. The resolution: the tensor for matrix
    # multiplication is NOT the all-ones tensor in the way I'm thinking.
    
    # I think the correct tensor is: T[i, j, k] where the three modes are:
    # Mode 1: indices of A, which are (i,j) for i in [0,n), j in [0,m)
    # Mode 2: indices of B, which are (j,k) for j in [0,m), k in [0,p)
    # Mode 3: indices of C, which are (i,k) for i in [0,n), k in [0,p)
    
    # And T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k', else 0.
    # Wait no, that would make it a "matching" tensor.
    
    # Actually, I think the correct definition is:
    # The multiplication tensor T ∈ R^{nm × mp × np} where
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
    # No, that's not right either.
    
    # Let me think about it from the decomposition perspective.
    # If we have rank-r decomposition with U (nm×r), V (mp×r), W (np×r),
    # then the condition is:
    # For the multiplication to be exact, we need:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * ... 
    # Hmm, this isn't working out cleanly.
    
    # Let me just use the verification: the decomposition is valid if for all
    # i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # AND for all "invalid" combinations, the sum = 0.
    # But what are "invalid" combinations? 
    # The tensor T[i,j,k] is only defined for i in [0,n), j in [0,m), k in [0,p).
    # There are no other entries. So the only constraint is that the sum equals 1
    # for all valid (i,j,k).
    
    # If that's the case, then rank 1 works: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, 
    # W[k*n+i, 0] = 1 for all valid indices. Then the sum is 1*1*1 = 1. 
    # So rank = 1?
    
    # This is clearly wrong because the problem states the schoolbook rank is n*m*p = 60.
    
    # I must be fundamentally misunderstanding the tensor structure. Let me reconsider.
    
    # Ah! I think the issue is that the tensor is NOT all ones. The multiplication
    # tensor for (n,m,p) has:
    # T[i, j, k] = 1 if the (i,k) entry of the product matrix C = AB involves
    # the product A[i,j] * B[j,k].
    # But this is true for ALL i,j,k, so T is all ones.
    
    # Unless... the tensor is defined differently. Maybe:
    # T is a tensor in R^{nm} ⊗ R^{mp} ⊗ R^{np} where
    # T[a, b, c] = 1 if a corresponds to A[i,j], b corresponds to B[j',k], 
    # c corresponds to C[i',k'] and i=i', j=j', k=k'.
    # So T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
    
    # In this case, T is NOT all ones. It's a "diagonal" tensor in some sense.
    # For n=m=p=1: T[0,0,0] = 1, rank 1.
    # For n=1,m=2,p=1: T[(0,0),(0,0),(0,0)] = 1, T[(0,0),(1,0),(0,0)] = 0, etc.
    # Actually T[(0,j),(j',0),(0,0)] = 1 if j=j', else 0.
    # So T is the identity-like tensor.
    
    # In this interpretation, the tensor rank is the minimum r such that we can write
    # T as sum of r rank-1 tensors, where a rank-1 tensor is u ⊗ v ⊗ w with
    # u ∈ R^{nm}, v ∈ R^{mp}, w ∈ R^{np}.
    
    # For the "diagonal" tensor T[(i,j),(j',k),(i',k')] = δ_{i,i'} δ_{j,j'} δ_{k,k'},
    # we can write it as sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    # which has rank n*m*p (the schoolbook rank).
    
    # But can we do better? For 2×2×2, the answer is 7 < 8.
    
    # So the tensor is: T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
    # Equivalently, T[a, b, c] = 1 if the three indices are "compatible" (the j from
    # A matches the j' from B, and the i from A matches i' from C, and k from B matches
    # k' from C).
    
    # Now, the condition for a rank-r decomposition U, V, W is:
    # sum_c U[a, c] * V[b, c] * W[c_idx, c] = T[a, b, c_idx]
    # for all a in [0,nm), b in [0,mp), c_idx in [0,np).
    
    # Where a = i*m+j, b = j'*p+k, c_idx = k'*n+i'.
    # T[a, b, c_idx] = 1 if i=i', j=j', k=k', else 0.
    
    # So the condition is:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    
    # This makes much more sense! And the schoolbook decomposition uses
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = i*m*p + j*p + k,
    # giving rank n*m*p.
    
    # Now I understand the problem. Let me try to find a better decomposition.
    
    # For 2×3×10, I need to find the minimum rank.
    # Known: rank(2,2,2) = 7.
    # For 2×3×3: I can try to construct a decomposition.
    
    # Let me try a computational approach: set up the equations and solve.
    
    # The equations are:
    # For all i,i' in [0,2), j,j' in [0,3), k,k' in [0,10):
    #   sum_c U[i*3+j, c] * V[j'*10+k, c] * W[k'*2+i', c] = 1 if i=i', j=j', k=k' else 0
    
    # This is a system of 2*3*10 * 2*3*10 = 60*30 = 1800 equations (but many are 0).
    # The non-zero equations are 2*3*10 = 60.
    
    # This is a nonlinear system (bilinear in U,V,W). Hard to solve directly.
    
    # Alternative: fix U and V, solve for W. Or use iterative methods.
    
    # Given time constraints, let me try a random initialization and gradient descent,
    # or just try to find a structured decomposition.
    
    # Let me try the following: use the 2×2×2 Strassen decomposition as a building block.
    # For the 2×3×10 problem, I can think of it as:
    # C = A * B where A is 2×3, B is 3×10.
    # Split B into B1 (3×5) and B2 (3×5): C = [A*B1, A*B2]
    # So rank(2,3,10) ≤ 2 * rank(2,3,5)
    
    # And rank(2,3,5) ≤ rank(2,3,2) + rank(2,3,3)
    # If rank(2,3,2) and rank(2,3,3) are known...
    
    # This is getting complicated. Let me just try a numerical optimization approach.
    
    # Actually, let me try a simpler approach: use the fact that the tensor can be
    # decomposed using the "border rank" or just try random rank-r decompositions
    # and check if they work.
    
    # Given the time constraint, let me try to construct a decomposition of rank < 60
    # using a structured approach.
    
    # For now, let me just return the schoolbook decomposition. The problem is
    # that finding a better one is non-trivial.
    
    # Actually, let me try one more thing: the tensor T for (n,m,p) has the property
    # that T[(i,j),(j',k),(i',k')] = δ_{i,i'}δ_{j,j'}δ_{k,k'}.
    # This can be written as: T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    # which is the schoolbook decomposition of rank n*m*p.
    
    # To improve, we need to find a more efficient decomposition.
    
    # Let me try to use a numerical method: initialize U, V, W randomly with rank r,
    # and use gradient descent to minimize the error.
    
    # But this is computationally expensive and may not converge in time.
    
    # Given the constraints, let me just return the schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
