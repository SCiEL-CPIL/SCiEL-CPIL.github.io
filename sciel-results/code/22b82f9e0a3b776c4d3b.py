import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # U, V, W are 4x7 matrices
    # Indices: rows 0..3 correspond to (i,j) in 0..1,0..1 order
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Construct rank-49 decomposition for 4x4 via Kronecker product
    # U4 will be 16x49
    # Row index in U4: i*4 + j, where i,j in 0..3
    # i = 2*i_hi + i_lo, j = 2*j_hi + j_lo
    # Col index in U4: r*7 + s, where r,s in 0..6
    # U4[i*4+j, r*7+s] = U2[i_hi*2+j_hi, r] * U2[i_lo*2+j_lo, s]
    
    def build_kronecker_factor(F2):
        # F2 is 4x7
        # Output is 16x49
        F4 = np.zeros((16, 49), dtype=int)
        for i in range(4):
            for j in range(4):
                i_hi = i // 2
                i_lo = i % 2
                j_hi = j // 2
                j_lo = j % 2
                
                # Get the corresponding rows from F2
                row_hi = F2[i_hi * 2 + j_hi] # shape (7,)
                row_lo = F2[i_lo * 2 + j_lo] # shape (7,)
                
                # Outer product
                outer = np.outer(row_hi, row_lo) # shape (7, 7)
                
                # Flatten and place into F4
                F4[i * 4 + j, :] = outer.flatten()
        return F4

    U4 = build_kronecker_factor(U2)
    V4 = build_kronecker_factor(V2)
    W4 = build_kronecker_factor(W2)

    # Verify correctness
    # Check sum_r U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] == 1
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                prod = np.sum(U4[i * 4 + j] * V4[j * 4 + k] * W4[k * 4 + i])
                if prod != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to trivial rank 64 if construction fails (should not happen)
        rank = 64
        U_f = np.zeros((16, 64), dtype=int)
        V_f = np.zeros((16, 64), dtype=int)
        W_f = np.zeros((16, 64), dtype=int)
        col = 0
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    U_f[ii * 4 + jj, col] = 1
                    V_f[jj * 4 + kk, col] = 1
                    W_f[kk * 4 + ii, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    # Attempt to find a lower rank decomposition
    # Check if the 49 terms are linearly dependent.
    # If so, we can potentially reduce the rank to 48.
    
    if time.time() - start_time < deadline - 5:
        # Build the matrix of terms
        # T is a vector of 64 ones.
        # Terms are 49 vectors of length 64.
        terms = np.zeros((49, 64), dtype=float)
        for c in range(49):
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        terms[c, i * 16 + j * 4 + k] = U4[i * 4 + j, c] * V4[j * 4 + k, c] * W4[k * 4 + i, c]
        
        # Check rank of terms matrix
        rank_terms = np.linalg.matrix_rank(terms, tol=1e-9)
        
        if rank_terms < 49:
            # Find null space
            # We want vector lambda such that terms.T @ lambda = 0
            # i.e., sum_c lambda_c * terms[c] = 0
            # This implies T = sum_c T_c = sum_c T_c - sum_c lambda_c T_c = sum_c (1 - lambda_c) T_c ?
            # No. T = sum_c 1 * T_c.
            # If sum_c lambda_c T_c = 0, then T = sum_c 1 * T_c = sum_c (1 + lambda_c) T_c ?
            # Let's check: sum_c (1 + lambda_c) T_c = sum_c T_c + sum_c lambda_c T_c = T + 0 = T.
            # So the new coefficients are 1 + lambda_c.
            # If any 1 + lambda_c is 0, we can drop that term.
            
            # Use SVD to find null space
            U_svd, s, Vt = np.linalg.svd(terms, full_matrices=True)
            # Null space is spanned by rows of Vt corresponding to small singular values
            # The rank is rank_terms. The null space dimension is 49 - rank_terms.
            # Let's take the last (49 - rank_terms) rows of Vt.
            null_space = Vt[rank_terms:, :].T # shape (49, 49-rank_terms)
            
            # Try to find a null vector that has a component close to -1 for some index
            # We want lambda such that 1 + lambda_c = 0 => lambda_c = -1.
            # Let's iterate through the null space basis vectors.
            
            found_reduction = False
            for vec in null_space.T:
                # vec is a vector in null space
                # Check if 1 + vec_c is close to 0 for any c
                # We can scale vec. Let vec = alpha * v.
                # We want 1 + alpha * v_c = 0 => alpha = -1/v_c.
                # This is possible for any c where v_c != 0.
                # But we need the resulting coefficients to be "nice" (integers/half-integers) or at least valid.
                # The problem allows integer/half-integer entries.
                # The original entries are integers.
                # If we use a linear combination, the new U, V, W matrices will be linear combinations of the old columns.
                # U_new[:, c'] = U_old[:, c'] + ...
                # This might result in non-integer entries if the coefficients are not integers.
                # However, the task says "integer/half-integer entries".
                # If we find a rational null vector, we can scale it to integers.
                
                # Let's just check if we can eliminate a term.
                # If we can eliminate term k, we set coefficient of k to 0.
                # Coeff_k = 1 + lambda_k = 0 => lambda_k = -1.
                # We need a null vector lambda such that lambda_k = -1.
                # This is a linear system.
                
                # Let's try to solve for lambda such that lambda_k = -1 for some k.
                # This is equivalent to finding a vector in the null space that has a specific value at k.
                # We can project the basis onto the condition.
                
                # Simpler: Just check if the null space contains a vector with a -1 entry?
                # Or rather, can we form a combination?
                
                # If rank_terms < 49, the vectors are dependent.
                # We can always express one vector as a combination of others.
                # Let's find the dependency explicitly.
                # Solve terms.T @ lambda = 0.
                
                # Let's pick the last column as the one to eliminate.
                # We want lambda_48 = -1 (so 1 + lambda_48 = 0).
                # Then lambda_0..47 are determined by the null space condition?
                # No, the null space is a subspace. We can choose any vector in it.
                # We want a vector lambda in Null(terms) such that lambda_48 = -1.
                # If the projection of the null space onto the 48-th coordinate is non-zero, we can do this.
                
                # Let's check if the 48-th column is in the span of the others.
                # This is equivalent to rank(terms[:, :48]) == rank(terms).
                
                # If we can eliminate term 48, we get a 48-term decomposition.
                # The new U, V, W will be:
                # U_new[:, c] = U_old[:, c] for c < 48
                # V_new[:, c] = V_old[:, c] for c < 48
                # W_new[:, c] = W_old[:, c] for c < 48
                # And we need to add the contribution of term 48 to the other terms.
                # T = sum_{c=0}^{47} T_c + T_48.
                # T_48 = sum_{c=0}^{47} mu_c T_c.
                # T = sum_{c=0}^{47} (1 + mu_c) T_c.
                # This means we can just modify the coefficients?
                # No, the decomposition is a sum of rank-1 tensors $U_c \otimes V_c \otimes W_c$.
                # We cannot just add coefficients to the same $U_c, V_c, W_c$.
                # We need to form new rank-1 tensors.
                # $T_48 = U_4 \otimes V_4 \otimes W_4$.
                # We want to write $U_4 \otimes V_4 \otimes W_4$ as a sum of terms involving $U_c, V_c, W_c$ for $c < 48$.
                # This is generally NOT possible. The span of $\{ U_c \otimes V_c \otimes W_c \}$ is not closed under addition in a way that allows one tensor to be a linear combination of others using the SAME factors.
                # The linear dependency $\sum \lambda_c T_c = 0$ is a dependency in the vector space of tensors.
                # It does NOT imply that one rank-1 tensor can be replaced by a linear combination of others using the same rank-1 factors.
                # It implies that the SUM is invariant under adding the dependency.
                # But to reduce the number of terms, we need to merge them.
                # Merging two rank-1 tensors $(A \otimes B \otimes C) + (A' \otimes B' \otimes C')$ into a single rank-1 tensor is only possible if they are proportional.
                # So, linear dependence of the term vectors does NOT necessarily allow reducing the rank of the decomposition.
                
                # Therefore, the rank 49 is likely minimal or at least hard to reduce by this method.
                
                pass

    # Return the rank-49 solution
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
