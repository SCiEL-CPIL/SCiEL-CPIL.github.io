import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    # U shape: (n*m, rank) = (20, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # The tensor T has entries T[i, j, k] = 1 for all i,j,k.
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] ?
    # No, the standard decomposition is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # In tensor form: T_{i,j,k} = 1.
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,k}? No.
    # The bilinear form is: sum_c (sum_i U[i*m+j, c] A[i,j]) * (sum_k V[j*p+k, c] B[j,k]) ?
    # No, the standard rank-1 term is U_i * V_j * W_k.
    # The multiplication tensor is M_{i,j,k} = 1.
    # We seek U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k?
    # Wait, the output is C[i,k].
    # C[i,k] = sum_j A[i,j] B[j,k].
    # If we use rank-1 terms: t_c = (sum_i u_c[i,j] A[i,j]) * (sum_k v_c[j,k] B[j,k])?
    # No, the factors must be linear forms in A and B.
    # A linear form in A is sum_{i,j} u_{i,j} A_{i,j}.
    # A linear form in B is sum_{j,k} v_{j,k} B_{j,k}.
    # The product is a bilinear form.
    # We need: sum_c (sum_{i',j'} U[i'*m+j', c] A[i',j']) * (sum_{j'',k''} V[j''*p+k'', c] B[j'',k'']) = sum_j A[i,j] B[j,k].
    # This implies that for a fixed c, the term is:
    # (sum_{i',j'} U[i'*m+j', c] A[i',j']) * (sum_{j'',k''} V[j''*p+k'', c] B[j'',k''])
    # = sum_{i',j',j'',k''} U[i'*m+j', c] V[j''*p+k'', c] A[i',j'] B[j'',k'']
    # We need this to equal sum_j A[i,j] B[j,k] when summed over c.
    # So, for each c, the coefficient of A[i',j'] B[j'',k''] in the expansion must be 0 unless j'=j'' and (i',k') matches (i,k) in the final sum?
    # Actually, the "W" matrix in the prompt description `W[k*n+i, r]` suggests the third factor is associated with the output index.
    # Let's re-read the prompt: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)".
    # And "W[k*n+i, r]".
    # This looks like a trilinear form decomposition of the tensor M_{i,j,k} = 1.
    # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 ?
    # No, that would be a trilinear form T(i,j,k).
    # But matrix multiplication is bilinear in A and B, linear in C.
    # The tensor for matrix multiplication is usually defined as T_{i,j,k} = 1 if the indices match the multiplication structure?
    # Let's check the identity:
    # sum_c U[i*m+j, c] V[j*p+k, c] = delta_{i,k} ? No.
    
    # Let's look at the standard definition used in such tasks (e.g., Strassen).
    # The rank of matrix multiplication M_{n,m,p} is the smallest r such that:
    # sum_{c=1}^r (sum_{i,j} u^c_{i,j} A_{i,j}) (sum_{j,k} v^c_{j,k} B_{j,k}) = sum_{j} A_{i,j} B_{j,k}
    # Wait, the LHS is a sum of products of linear forms.
    # The RHS is the matrix product.
    # This means:
    # sum_c (sum_{i',j'} u^c_{i',j'} A_{i',j'}) (sum_{j'',k''} v^c_{j'',k''} B_{j'',k''}) = sum_{j} A_{i,j} B_{j,k}
    # Expanding LHS:
    # sum_{i',j',j'',k''} (sum_c u^c_{i',j'} v^c_{j'',k''}) A_{i',j'} B_{j'',k''}
    # We need this to equal sum_{j} A_{i,j} B_{j,k}.
    # So, the coefficient of A_{i',j'} B_{j'',k''} in the LHS must be:
    # 1 if j'=j'' and i'=i and k''=k (for the specific i,k being computed? No, the equation must hold for all i,k).
    # Actually, the equation is for a specific output element C_{i,k}.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So we need:
    # sum_c (sum_{i',j'} u^c_{i',j'} A_{i',j'}) (sum_{j'',k''} v^c_{j'',k''} B_{j'',k''}) |_{extracting C_{i,k}} = sum_j A_{i,j} B_{j,k}
    # This extraction is done by the W matrix?
    # The prompt says: "W[k*n+i, r]".
    # This suggests the decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 ?
    # If this is the case, it is a decomposition of the 3-tensor T_{i,j,k} = 1.
    # But T_{i,j,k}=1 is not the matrix multiplication tensor.
    # The matrix multiplication tensor has non-zero entries only when the indices align?
    # No, the tensor M_{n,m,p} has entries M_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is defined by the bilinear map.
    # In the "rank" definition, we are decomposing the tensor of the bilinear form.
    # The tensor is T \in A^* \otimes B^* \otimes C^*.
    # T(A,B,C) = tr(C^T (AB)) = sum_{i,j,k} C_{k,i} A_{i,j} B_{j,k}.
    # So the tensor components are T_{i,j,k} = 1 if we index C by (k,i)?
    # Let's check the index in W: W[k*n+i, r].
    # This matches C_{k,i} if we flatten C as (k,i) or (i,k)?
    # If C is p x n, and we flatten it as row-major, index is k*n + i.
    # So T_{i,j,k} = 1 for all i,j,k?
    # Yes, because tr(C^T AB) = sum_{i,j,k} C_{k,i} A_{i,j} B_{j,k}.
    # So the tensor is the all-ones tensor of shape (n*m, m*p, p*n).
    # We want to decompose this all-ones tensor into rank-r terms.
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # This is a much simpler problem! It is just decomposing the all-ones tensor.
    # But wait, if the tensor is all ones, the rank is 1!
    # U = ones, V = ones, W = ones.
    # sum_c 1*1*1 = r. We need sum_c ... = 1.
    # So we can set r=1, U=ones, V=ones, W=ones.
    # Then sum_c 1 = 1.
    # Is the rank of the all-ones tensor 1? Yes.
    # But the rank of matrix multiplication is NOT 1.
    # Why? Because the tensor is NOT all ones.
    # The tensor for matrix multiplication is T_{i,j,k} = 1 if the indices correspond to a valid multiplication term?
    # No, the tensor is defined by the bilinear form.
    # The bilinear form is (A,B) -> AB.
    # The associated trilinear form is (A,B,C) -> tr(C^T AB).
    # The components of this tensor are T_{i,j,k} = 1 for ALL i,j,k?
    # Let's check: tr(C^T AB) = sum_{i,j,k} C_{k,i} A_{i,j} B_{j,k}.
    # Yes, every term C_{k,i} A_{i,j} B_{j,k} appears with coefficient 1.
    # So the tensor IS the all-ones tensor.
    # If the tensor is all-ones, the rank is 1.
    # But we know R(2,2,2) = 7.
    # Contradiction.
    
    # Where is the error?
    # The error is in the interpretation of "rank".
    # The rank of a bilinear form is the minimum number of products of linear forms needed to represent it.
    # The bilinear form is B(A,B) = AB.
    # This is a map A x B -> C.
    # It can be viewed as a trilinear form T(A,B,C) = sum_{i,k} C_{i,k} (AB)_{i,k}.
    # T(A,B,C) = sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k}.
    # So the tensor components are T_{i,j,k} = 1?
    # Wait, the indices of C are i,k.
    # The prompt says W[k*n+i, r]. This implies C is indexed by (k,i) or (i,k) flattened.
    # If C is p x n, and index is k*n+i, then C_{k,i} is the element.
    # So T_{i,j,k} = 1.
    # If T is all ones, rank is 1.
    # Why is R(2,2,2) = 7?
    # Because the bilinear form is NOT just the trace pairing with the product?
    # No, it is.
    # Ah! The rank of the tensor T_{i,j,k} = 1 (all ones) is indeed 1.
    # But the matrix multiplication tensor is NOT the all-ones tensor in the space of linear forms?
    # Let's check Strassen.
    # Strassen decomposes M_{2,2,2} into 7 terms.
    # Each term is a product of 3 linear forms.
    # If the tensor were all-ones, 1 term would suffice.
    # So the tensor is NOT all-ones.
    
    # Re-evaluating the tensor definition.
    # The tensor M_{n,m,p} is in (A^* \otimes B^* \otimes C^*).
    # The basis vectors are A_{i,j}, B_{j,k}, C_{i,k}.
    # The tensor is sum_{i,j,k} A_{i,j} \otimes B_{j,k} \otimes C_{i,k}.
    # So the component T_{i,j,k} is 1 if the indices are (i,j) for A, (j,k) for B, (i,k) for C.
    # In the flattened indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (as per prompt)
    # So T_{u,v,w} = 1 if:
    # i = u // m, j = u % m
    # j' = v // p, k = v % p
    # k' = w // n, i' = w % n
    # And we need i=i', j=j', k=k'.
    # So T is a 0-1 tensor with specific structure, NOT all ones.
    # It is the "identity" tensor in a specific sense.
    
    # Okay, so we are back to decomposing this specific tensor.
    
    # Strategy: Use `scipy.optimize.minimize` to find a low-rank decomposition.
    # We will try ranks from 100 upwards (or downwards from 120).
    # Let's try to beat 120.
    
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    
    # Initialize schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j, c] = 1
                best_V[j * p + k, c] = 1
                best_W[k * n + i, c] = 1

    # Try to reduce rank
    # We will try ranks: 119, 118, ...
    # But optimization is hard.
    # Let's try a few specific ranks.
    
    target_ranks = [115, 110, 105, 100, 95, 90]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        success = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W with random values
            # Scale to help convergence
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Define the objective function
            # We want sum_c U[u,c] V[v,c] W[w,c] = T[u,v,w]
            # This is a tensor completion problem.
            # We can minimize the Frobenius norm of the difference.
            
            # Flatten the tensor T
            # T is sparse.
            
            def objective(params):
                # Reshape params back to U, V, W
                size_U = n * m * target_rank
                size_V = m * p * target_rank
                size_W = p * n * target_rank
                
                U_flat = params[:size_U]
                V_flat = params[size_U:size_U + size_V]
                W_flat = params[size_U + size_V:]
                
                U_mat = U_flat.reshape((n * m, target_rank))
                V_mat = V_flat.reshape((m * p, target_rank))
                W_mat = W_flat.reshape((p * n, target_rank))
                
                # Compute the decomposed tensor
                # This is expensive if done naively.
                # We only need to check a subset of entries?
                # No, we need the whole tensor to be exact.
                # But for optimization, we can sample.
                
                # Let's compute the full tensor difference.
                # T_decomposed[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
                # This is a 3-way tensor product.
                
                # Efficiently compute:
                # Temp = U @ V.T  (n*m x m*p) ? No.
                # Term c: U[:,c] outer V[:,c] outer W[:,c]
                
                # We can compute the sum of outer products.
                # This is equivalent to:
                # D = np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
                # But D is huge (20x30x24 = 14400).
                # We can compare with T.
                
                # T is 0 or 1.
                # We construct T as a sparse structure or dense.
                # Dense is 14400 floats. Small enough.
                
                T = np.zeros((n * m, m * p, p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = i * m + j
                            v = j * p + k
                            w = k * n + i
                            T[u, v, w] = 1.0
                
                D = np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
                
                diff = D - T
                return np.sum(diff * diff)
            
            # Use L-BFGS-B
            from scipy.optimize import minimize
            
            x0 = np.concatenate([best_U[:, :target_rank].flatten() if best_U.shape[1] >= target_rank else np.zeros(n*m*target_rank),
                                 best_V[:, :target_rank].flatten() if best_V.shape[1] >= target_rank else np.zeros(m*p*target_rank),
                                 best_W[:, :target_rank].flatten() if best_W.shape[1] >= target_rank else np.zeros(p*n*target_rank)])
            
            # Actually, random init is better for escaping local minima
            x0 = np.random.randn(n * m * target_rank + m * p * target_rank + p * n * target_rank) * 0.1
            
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-9})
            
            if result.fun < 1e-6:
                # Check if we can round to integers/half-integers
                # Extract U, V, W
                size_U = n * m * target_rank
                size_V = m * p * target_rank
                U_mat = result.x[:size_U].reshape((n * m, target_rank))
                V_mat = result.x[size_U:size_U + size_V].reshape((m * p, target_rank))
                W_mat = result.x[size_U + size_V:].reshape((p * n, target_rank))
                
                # Round to nearest half-integer
                U_rounded = np.round(U_mat * 2) / 2
                V_rounded = np.round(V_mat * 2) / 2
                W_rounded = np.round(W_mat * 2) / 2
                
                # Verify
                T = np.zeros((n * m, m * p, p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = i * m + j
                            v = j * p + k
                            w = k * n + i
                            T[u, v, w] = 1.0
                            
                D = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                
                if np.allclose(D, T, atol=1e-9):
                    best_rank = target_rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    success = True
                    break
        
        if success:
            # Try lower rank
            continue
        else:
            # If we failed for this rank, maybe try a slightly higher one?
            # Or just stop?
            # Let's stop if we fail for 115.
            break

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
