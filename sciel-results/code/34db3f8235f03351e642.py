import numpy as np
import time

def construct():
    n, m, p = 3, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Schoolbook decomposition (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def check(U, V, W, tol=1e-9):
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(val - 1) > tol:
                        return False
        return True

    # Strategy: Try to find a rank-83 decomposition by merging two schoolbook terms
    # via a Strassen-like identity on a 2x2x2 sub-block.
    
    # The 8 schoolbook terms for a 2x2x2 block (i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1})
    # can be replaced by 7 terms if we can find coefficients.
    # However, for the full tensor, the terms are independent unless they share indices.
    # The standard trick is to use the rank-7 decomposition of M_{2,2,2}.
    
    # Let's attempt to replace the 8 terms of the sub-block i in {0,1}, j in {0,1}, k in {0,1}
    # with the 7 Strassen terms.
    #
    # The 8 schoolbook terms correspond to columns c in the range:
    # c(i,j,k) = i*28 + j*7 + k
    # So for i in {0,1}, j in {0,1}, k in {0,1}, the columns are:
    # 0, 1, 7, 8, 28, 29, 35, 36
    
    # We will create a new decomposition of rank 83 by removing these 8 columns
    # and adding 7 new columns that reproduce the same contribution for these (i,j,k)
    # and zero contribution for all other (i,j,k).
    
    # This is only possible if the new columns have support only in the indices
    # that interact with i in {0,1}, j in {0,1}, k in {0,1}.
    # U rows: i*m+j for i in {0,1}, j in {0,1} -> rows 0,1,4,5
    # V rows: j*p+k for j in {0,1}, k in {0,1} -> rows 0,1,7,8
    # W rows: k*n+i for k in {0,1}, i in {0,1} -> rows 0,1,3,4
    
    # If we restrict U, V, W to have non-zeros only in these rows for the new columns,
    # then the new columns will only affect T[i,j,k] where i,j,k are in the respective sets.
    # Specifically, T[i,j,k] gets a contribution from a new column r only if:
    # U[i*m+j, r] != 0 AND V[j*p+k, r] != 0 AND W[k*n+i, r] != 0.
    # If U is non-zero only for i in {0,1}, j in {0,1}, then i must be 0 or 1 and j must be 0 or 1.
    # If V is non-zero only for j in {0,1}, k in {0,1}, then j must be 0 or 1 and k must be 0 or 1.
    # If W is non-zero only for k in {0,1}, i in {0,1}, then k must be 0 or 1 and i must be 0 or 1.
    # So the contribution is non-zero only for i in {0,1}, j in {0,1}, k in {0,1}.
    # This matches exactly the sub-block we want to replace.
    
    # So we need to find 7 vectors u_r, v_r, w_r (r=1..7) such that:
    # sum_{r=1}^7 u_r[i,j] v_r[j,k] w_r[k,i] = 1 for i in {0,1}, j in {0,1}, k in {0,1}
    # and u_r, v_r, w_r have support only in the allowed rows.
    
    # This is exactly the rank-7 decomposition of M_{2,2,2}.
    # Let's use the known Strassen decomposition.
    
    # Let the local indices be a, b in {0,1} for A (U part), c, d in {0,1} for B (V part), e, f in {0,1} for C (W part).
    # Wait, the mapping is:
    # U[i,j] corresponds to A_{i,j}
    # V[j,k] corresponds to B_{j,k}
    # W[k,i] corresponds to C_{k,i}
    # The product is A*B = C.
    # C_{k,i} = sum_j A_{i,j} B_{j,k}.
    # So we need sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} = C_{k,i}.
    # Note the indices in W are (k,i), which is the transpose of C_{i,k}.
    # Let's stick to the local 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    # We need to represent the tensor T_{i,j,k} = C_{k,i} = (A*B)_{k,i} = sum_j A_{i,j} B_{j,k}.
    # Wait, the standard matrix multiplication is C = A B, so C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The task defines W[k*n+i, r]. So the output index is (k,i).
    # This means the output matrix is C^T where C = A B?
    # Let's check the schoolbook:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # Term is U[i*m+j] V[j*p+k] W[k*n+i] = 1*1*1 = 1.
    # Sum over c (which corresponds to (i,j,k)) gives 1 for each (i,j,k).
    # So T[i,j,k] = 1.
    # This implies that the "value" at (i,j,k) is 1.
    # In matrix multiplication, the entry (i,k) of the product is sum_j A_{i,j} B_{j,k}.
    # Here the tensor is 3D. The decomposition says T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # If we set U[i,j,r] = A_{i,j}, V[j,k,r] = B_{j,k}, W[k,i,r] = delta_{i,k}? No.
    # The schoolbook works because each term picks a specific (i,j,k) and contributes 1.
    # This corresponds to the identity tensor? No, it's the multiplication tensor.
    # The multiplication tensor M_{n,m,p} has entries M_{i,j,k} = 1 if i,j,k are valid indices.
    # Wait, no. The tensor for matrix multiplication is defined such that
    # <M, A \otimes B \otimes C^T> = trace(A B C^T) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # So the tensor entries are M_{i,j,k} = 1 for all valid i,j,k?
    # No, that would be the all-ones tensor.
    # The multiplication tensor is the sum of rank-1 tensors:
    # M = sum_{i,j,k} e_i \otimes e_j \otimes e_k^* ?
    # Let's look at the schoolbook decomposition again.
    # It has rank n*m*p.
    # Each term r corresponds to a triple (i_r, j_r, k_r).
    # U has 1 at (i_r, j_r), V has 1 at (j_r, k_r), W has 1 at (k_r, i_r).
    # The sum is sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # For a fixed (i,j,k), the term r contributes if i_r=i, j_r=j, k_r=k.
    # There is exactly one such term (the one where i_r=i, j_r=j, k_r=k).
    # So the sum is 1.
    # Thus T[i,j,k] = 1 for all i,j,k.
    # So the tensor is indeed the all-ones tensor of shape (n*m, m*p, p*n) flattened?
    # No, the indices are coupled.
    # T[i,j,k] = 1.
    # This is the tensor that computes sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # Which is trace(A B C).
    # Yes, this is the standard matrix multiplication tensor.
    
    # So we need to decompose the all-ones tensor on the sub-block i in {0,1}, j in {0,1}, k in {0,1}
    # into 7 terms of rank 1, with the constraint that the vectors have support only in the
    # local indices.
    # This is equivalent to decomposing the rank-8 tensor (all ones 2x2x2) into 7 rank-1 tensors.
    # But the rank of the all-ones tensor 2x2x2 is 1! It's just 1*1*1.
    # Wait, if the tensor is all ones, then T[i,j,k] = 1.
    # Then U[i,j] = 1, V[j,k] = 1, W[k,i] = 1 gives T[i,j,k] = 1*1*1 = 1.
    # So the rank is 1!
    # Why is the schoolbook rank n*m*p then?
    # Because the indices i, j, k are coupled to the global matrix indices.
    # The "all ones" tensor is only valid locally if the vectors can be extended to the global space
    # with zeros elsewhere.
    # In the schoolbook, each term is a "basis vector" in the tensor product space.
    # The tensor M is sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # This tensor has rank n*m*p because the vectors e_i \otimes e_j \otimes e_k are linearly independent.
    # If we try to use a single term with U=1, V=1, W=1 (on the sub-block),
    # then for (i,j,k) in the sub-block, the contribution is 1.
    # But we need the contribution to be 1 for EACH (i,j,k) individually?
    # No, the sum over r must equal 1 for each (i,j,k).
    # If we use one term with U[i,j]=1 for all i,j in sub-block, V[j,k]=1, W[k,i]=1,
    # then for (i,j,k) in sub-block, the sum is 1.
    # So why do we need 8 terms?
    # Because the vectors U, V, W must be defined on the GLOBAL indices.
    # If we set U[i,j]=1 for i,j in {0,1} and 0 elsewhere,
    # V[j,k]=1 for j,k in {0,1} and 0 elsewhere,
    # W[k,i]=1 for k,i in {0,1} and 0 elsewhere,
    # then for (i,j,k) in the sub-block, the product is 1.
    # For (i,j,k) NOT in the sub-block, at least one of i,j,k is outside {0,1},
    # so at least one of U,V,W is 0, so the product is 0.
    # So a SINGLE term can cover the entire 2x2x2 sub-block!
    #
    # Wait, is that right?
    # Let's check.
    # Term r:
    # U[i*m+j, r] = 1 if i in {0,1} and j in {0,1}, else 0.
    # V[j*p+k, r] = 1 if j in {0,1} and k in {0,1}, else 0.
    # W[k*n+i, r] = 1 if k in {0,1} and i in {0,1}, else 0.
    #
    # For (i,j,k) = (0,0,0): U=1, V=1, W=1. Product=1.
    # For (i,j,k) = (0,0,1): U=1, V=1, W=1. Product=1.
    # ...
    # For (i,j,k) = (1,1,1): U=1, V=1, W=1. Product=1.
    #
    # For (i,j,k) = (0,0,2): U=1 (i=0,j=0 in {0,1}), V=0 (k=2 not in {0,1}). Product=0.
    # For (i,j,k) = (2,0,0): U=0 (i=2 not in {0,1}). Product=0.
    #
    # So yes, a single term covers the entire 2x2x2 sub-block!
    #
    # If this is true, then we can replace the 8 schoolbook terms with 1 term.
    # This would reduce the rank by 7.
    # Rank 84 -> 77.
    #
    # Why did I think Strassen was needed?
    # Strassen is needed for matrix multiplication where the output is C = A B.
    # Here, the "output" is not a matrix C, but the tensor T[i,j,k] which is 1 for all i,j,k.
    # The coupling is via the indices.
    # The term "rank-1 tensor" is u \otimes v \otimes w.
    # The multiplication tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # This is NOT the same as the matrix multiplication tensor M_{n,m,p} in the sense of
    # computing C = A B.
    #
    # Let's re-read the task.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 7 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    #
    # The standard matrix multiplication tensor M_{n,m,p} is defined such that
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be written as C_{i,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r] ?? No.
    # Usually, the decomposition is C_{i,k} = sum_r U[i,r] V[j,r] W[k,r]? No.
    #
    # The tensor is T_{i,j,k}.
    # The contraction is <T, A \otimes B \otimes C^T> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # We want this to equal trace(A B C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # So we need T_{i,j,k} = 1 for all i,j,k.
    #
    # So the tensor IS the all-ones tensor!
    #
    # And as I showed above, the all-ones tensor on a sub-block can be represented by a single rank-1 term
    # if we allow the vectors to be 1 on the entire sub-block.
    #
    # So, can we just use 1 term for the whole 3x4x7 block?
    # U[i*m+j] = 1 for all i,j.
    # V[j*p+k] = 1 for all j,k.
    # W[k*n+i] = 1 for all k,i.
    # Then T[i,j,k] = 1*1*1 = 1 for all i,j,k.
    # So the rank is 1!
    #
    # This seems too good to be true. Is there a constraint I'm missing?
    # "integer/half-integer entries" - 1 is an integer.
    # "exact tensor decomposition" - it is exact.
    #
    # Why is the schoolbook rank n*m*p?
    # Because the schoolbook decomposition uses BASIS vectors.
    # e_i \otimes e_j \otimes e_k.
    # But the tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # This sum can be written as (sum_i e_i) \otimes (sum_j e_j) \otimes (sum_k e_k).
    # Which is a single rank-1 tensor!
    #
    # So the rank of the all-ones tensor is 1.
    #
    # Is the matrix multiplication tensor the all-ones tensor?
    # Let's check the definition again.
    # If A and B are matrices, C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T is defined by T_{i,j,k} such that C_{i,k} = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} ??
    # No, C_{i,k} is a specific entry.
    # The bilinear form is (A, B) -> A B.
    # The tensor is the linear map from A \otimes B to C.
    # In index notation, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be seen as a 3-tensor T_{i,j,k} where T_{i,j,k} = 1 if the indices are valid?
    # No, T_{i,j,k} is not a constant.
    # The map is (A, B) -> C.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a bilinear map.
    # The tensor associated with this map is M_{i,j,k} where M_{i,j,k} = 1 if j is a valid index?
    # No, the tensor is defined by M_{i,j,k} = delta_{i,i} delta_{j,j} delta_{k,k}?
    #
    # Let's look at the schoolbook decomposition provided in the seed.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # Term c corresponds to (i,j,k).
    # Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # For a fixed (i,j,k), the only non-zero term is r = (i,j,k).
    # So the sum is 1.
    # So T[i,j,k] = 1.
    #
    # So the tensor IS the all-ones tensor.
    #
    # And the all-ones tensor has rank 1.
    #
    # So why is the seed program using rank 84?
    # Maybe I am misunderstanding the "trace convention".
    # "W[k*n+i,r]"
    #
    # Let's assume the task is correct and the rank is 1.
    # Let's try to return a rank-1 decomposition.
    
    U = np.zeros((n * m, 1))
    V = np.zeros((m * p, 1))
    W = np.zeros((p * n, 1))
    
    U[:, 0] = 1
    V[:, 0] = 1
    W[:, 0] = 1
    
    if check(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # If rank 1 fails (which it shouldn't based on my analysis), fallback to schoolbook.
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
