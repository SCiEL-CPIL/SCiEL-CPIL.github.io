"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U, V, W, rank = schoolbook()
    
    # Verify correctness
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    if not verify(U, V, W):
        # Should not happen for schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Greedy rank reduction
    # We represent the decomposition as a list of columns (u_c, v_c, w_c)
    # Initially, each column is a standard basis vector.
    # We try to replace pairs of columns with a single column.
    # This is equivalent to checking if the tensor formed by 2 terms can be represented by 1 term.
    # T_2 = u1 v1 w1^T + u2 v2 w2^T
    # We want to find u, v, w such that T_2 = u v w^T.
    # This is possible if and only if the rank of the matrix formed by flattening T_2 is 1? No.
    # The tensor T is rank 1 if and only if all its matricizations have rank 1.
    
    def try_merge(c1, c2):
        """
        Try to replace columns c1 and c2 with a single column.
        Returns the new column (u, v, w) if successful, else None.
        """
        u1 = U[:, c1]
        v1 = V[:, c1]
        w1 = W[:, c1]
        u2 = U[:, c2]
        v2 = V[:, c2]
        w2 = W[:, c2]
        
        # Tensor T = u1 v1 w1^T + u2 v2 w2^T
        # We want T = u v w^T
        # This implies that the support of u, v, w must be contained in the support of u1+u2, v1+v2, w1+w2.
        # More strongly, the tensor must have rank 1.
        
        # Let's check the rank of the tensor T.
        # We can check the rank of the matrix M_ij = sum_k T[i,j,k] ... no.
        # The tensor is rank 1 if and only if the matrix (u1 v1^T + u2 v2^T) has rank 1? No.
        
        # A simpler check:
        # If T = u v w^T, then for any i, j, k, T[i,j,k] = u[i] v[j] w[k].
        # This implies that the set of non-zero entries of T must form a "grid" structure.
        
        # Let's try to solve for u, v, w directly.
        # Assume u and v are non-zero.
        # T[i,j,k] = u[i] v[j] w[k]
        # T[i,j,0] = u[i] v[j] w[0]
        # T[i,j,1] = u[i] v[j] w[1]
        # So T[i,j,0] / T[i,j,1] = w[0] / w[1] for all i,j where T[i,j,1] != 0.
        # This ratio must be constant.
        
        # Let's construct the tensor T explicitly for the two terms.
        # Since the dimensions are small (15, 35, 21), we can just build the full tensor.
        # But that's expensive if done for every pair.
        
        # Optimization:
        # The support of T is the union of supports of term 1 and term 2.
        # Support of term 1: (i1, j1, k1) where u1[i1]!=0, v1[j1]!=0, w1[k1]!=0.
        # Since schoolbook terms are basis vectors, u1, v1, w1 have a single non-zero entry.
        # Wait, if we start with schoolbook, all terms are basis vectors.
        # Merging two basis vectors e_{a} e_{b} e_{c} + e_{a'} e_{b'} e_{c'} into a rank 1 tensor.
        # The sum of two basis tensors is rank 1 if and only if they are the same tensor?
        # No. e_1 e_1 e_1 + e_2 e_2 e_2 is rank 2.
        # e_1 e_1 e_1 + e_1 e_1 e_2 is rank 2?
        # Matrix: [1 1; 0 0]. Rank 1.
        # Tensor: T[0,0,0]=1, T[0,0,1]=1.
        # Can we write this as u v w^T?
        # u=[1,0], v=[1,0], w=[1,1].
        # u v w^T = [1,0]^T [1,0] [1,1] = [1,0]^T [1,0] [1,1] = [[1,1],[0,0]].
        # Yes!
        # So we CAN merge two schoolbook terms if they share 2 indices.
        # Term (i,j,k1) and (i,j,k2) can be merged into u=e_i, v=e_j, w=e_k1+e_k2.
        
        # General condition for merging two rank-1 tensors t1, t2 into rank 1:
        # The tensor t1+t2 must have rank 1.
        # This happens if and only if the supports are "aligned" in a specific way.
        
        # Let's just check the rank of the tensor t1+t2.
        # We can compute the matricization and check its rank.
        
        # Construct T
        T = np.zeros((n, m, p))
        T += np.einsum('i,j,k->ijk', u1, v1, w1)
        T += np.einsum('i,j,k->ijk', u2, v2, w2)
        
        # Check if T is rank 1
        # Flatten T to a matrix (n*m, p) and check rank
        M = T.reshape(n*m, p)
        # If rank(M) <= 1, then T is rank 1?
        # Yes, if T is rank 1, all matricizations have rank 1.
        # Conversely, if one matricization has rank 1, is T rank 1?
        # Not necessarily. But for the specific structure of sums of 2 basis tensors, it should work.
        # Actually, for a general tensor, rank 1 in all matricizations is necessary but not sufficient?
        # No, a tensor is rank 1 if and only if it can be written as a product of 3 vectors.
        # This is equivalent to the matrix M_ij = sum_k T_ijk ... no.
        # Let's just try to recover u, v, w.
        
        # Find the first non-zero entry in T
        idx = np.unravel_index(np.argmax(np.abs(T) > 1e-9), T.shape)
        if np.abs(T[idx]) < 1e-9:
            return None # Zero tensor
            
        i0, j0, k0 = idx
        val0 = T[i0, j0, k0]
        
        # Assume u[i0] = 1. Then v[j0] = val0 / w[k0].
        # This is underdetermined.
        # But we can check if the tensor fits the form u v w^T.
        # T[i,j,k] / T[i0,j0,k0] = (u[i]/u[i0]) * (v[j]/v[j0]) * (w[k]/w[k0])
        
        # Let's just check if the matrix M has rank 1.
        # If rank(M) == 1, then M = a b^T.
        # Then T[i,j,k] = a[i*m+j] b[k].
        # This implies u[i] v[j] = a[i*m+j] and w[k] = b[k].
        # We need to factor a into u and v.
        # a is a vector of length n*m.
        # We need a[i*m+j] = u[i] v[j].
        # This is possible if and only if the matrix A_ij = a[i*m+j] has rank 1.
        
        # So:
        # 1. Check if M (matricization of T) has rank 1.
        # 2. If yes, extract a, b.
        # 3. Check if A (reshaped a) has rank 1.
        # 4. If yes, extract u, v.
        
        # Step 1: Rank of M
        # M is (n*m) x p.
        # We can check if all 2x2 minors are 0.
        # Or just use SVD.
        U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
        if len(s) > 1 and s[1] > 1e-6:
            return None
        if s[0] < 1e-6:
            return None
            
        a = U_svd[:, 0] * s[0]
        b = Vt_svd[0, :]
        
        # Step 2: Reshape a to (n, m) and check rank
        A = a.reshape(n, m)
        U_svd2, s2, Vt_svd2 = np.linalg.svd(A, full_matrices=False)
        if len(s2) > 1 and s2[1] > 1e-6:
            return None
        if s2[0] < 1e-6:
            return None
            
        u = U_svd2[:, 0] * s2[0]
        v = Vt_svd2[0, :]
        
        # We have u, v, b.
        # But we need to fix the scaling.
        # T = u v b^T.
        # We scaled a = u v^T.
        # So T = (u v^T) b^T = u (v b^T)? No.
        # T[i,j,k] = a[i*m+j] b[k] = u[i] v[j] b[k].
        # So w = b.
        
        # Let's verify
        T_check = np.einsum('i,j,k->ijk', u, v, b)
        if not np.allclose(T, T_check, atol=1e-6):
            return None
            
        return u, v, b

    # Main loop
    # We maintain lists of columns.
    # To allow speed, we keep U, V, W as lists of vectors.
    U_list = [U[:, c] for c in range(rank)]
    V_list = [V[:, c] for c in range(rank)]
    W_list = [W[:, c] for c in range(rank)]
    
    current_rank = rank
    
    # Greedy search
    # Try to merge any pair.
    # To be efficient, we can prioritize pairs that are "close" (share indices).
    # But let's just iterate.
    
    # Optimization:
    # If we merge c1 and c2, we remove them and add the new one.
    # We need to be careful with indices.
    
    # We will use a "set" of active indices.
    # Actually, since we remove and add, it's easier to just use lists and swap-remove.
    
    merged_any = True
    while merged_any and time.time() < deadline:
        merged_any = False
        # Iterate over all pairs
        # To avoid modifying list while iterating, we collect merges to apply later?
        # Or apply immediately and restart?
        # Applying immediately is faster but might miss opportunities if order matters.
        # Let's apply immediately and restart the scan.
        
        for c1 in range(len(U_list)):
            for c2 in range(c1 + 1, len(U_list)):
                if time.time() > deadline:
                    break
                res = try_merge(c1, c2)
                if res is not None:
                    u_new, v_new, w_new = res
                    # Remove c1, c2
                    # Swap remove c2
                    U_list[c2] = U_list[-1]
                    V_list[c2] = V_list[-1]
                    W_list[c2] = W_list[-1]
                    U_list.pop()
                    V_list.pop()
                    W_list.pop()
                    
                    # Swap remove c1 (which is now at c1)
                    U_list[c1] = U_list[-1]
                    V_list[c1] = V_list[-1]
                    W_list[c1] = W_list[-1]
                    U_list.pop()
                    V_list.pop()
                    W_list.pop()
                    
                    # Add new
                    U_list.append(u_new)
                    V_list.append(v_new)
                    W_list.append(w_new)
                    
                    merged_any = True
                    current_rank -= 1
                    break
            if merged_any:
                break
            if time.time() > deadline:
                break

    # Convert back to matrices
    U_final = np.array(U_list).T
    V_final = np.array(V_list).T
    W_final = np.array(W_list).T
    
    # Verify
    if verify(U_final, V_final, W_final):
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        # Fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
