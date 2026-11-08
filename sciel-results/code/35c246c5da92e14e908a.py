import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build the rank-56 Strassen-based decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 patterns (rank-7)
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # U: (i,j) pairs, V: (j,k) pairs, W: (k,i) pairs
    strassen_U = [
        [(0, 1), (3, 1)],    # M1: (A00+A11)
        [(2, 1), (3, 1)],    # M2: (A10+A11)
        [(0, 1)],            # M3: A00
        [(3, 1)],            # M4: A11
        [(0, 1), (1, 1)],    # M5: (A00+A01)
        [(2, 1), (0, -1)],   # M6: (A10-A00)
        [(1, 1), (3, -1)],   # M7: (A01-A11)
    ]
    strassen_V = [
        [(0, 1), (3, 1)],    # M1: (B00+B11)
        [(0, 1)],            # M2: B00
        [(1, 1), (3, -1)],   # M3: (B01-B11)
        [(2, 1), (0, -1)],   # M4: (B10-B00)
        [(3, 1)],            # M5: B11
        [(0, 1), (1, 1)],    # M6: (B00+B01)
        [(2, 1), (3, 1)],    # M7: (B10+B11)
    ]
    strassen_W = [
        [(0, 1), (3, 1)],    # M1: C00+C11
        [(2, 1), (3, -1)],   # M2: C10-C11
        [(1, 1), (3, 1)],    # M3: C01+C11
        [(0, 1), (2, 1)],    # M4: C00+C10
        [(0, -1), (1, 1)],   # M5: -C00+C01
        [(3, 1)],            # M6: C11
        [(0, 1)],            # M7: C00
    ]
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val
    
    # 8 blocks: 2 (left/right A) x 4 (k-blocks of size 2)
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to reduce rank by finding linear dependencies
    # The tensor decomposition is valid if for all r:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    # 
    # We can try to find a subset of rank terms that still spans the tensor.
    # Use a greedy approach: try removing one term at a time and check validity.
    
    def verify_decomposition(U_arr, V_arr, W_arr, n=2, m=4, p=8):
        """Check if the decomposition is exact."""
        # For each output (i, k), check that sum_j A[i,j]*B[j,k] is reproduced
        # The tensor T[i,j,k; i',k'] should satisfy:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        # Wait, let me re-derive. The multiplication tensor for (n x m) * (m x p) -> (n x p):
        # T[i, j, k, i', k'] = 1 if i=i' and k=k' and j=j... no.
        # 
        # Actually the standard formulation: the bilinear map mult: R^(n*m) x R^(m*p) -> R^(n*p)
        # is represented by a tensor T in (R^(n*m))^* (x (R^(m*p))^* (x R^(n*p)
        # T(a, b) = a^T (T) b where T is a 3-index tensor.
        # 
        # For matrix mult: (AB)[i,k] = sum_j A[i,j] B[j,k]
        # So T[i, j, k; i', k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ... 
        # No wait. Let me think of it as:
        # T is a tensor in R^(n*m) (x R^(m*p) (x R^(n*p) such that
        # <T, A (x B> = sum_{i,k} (AB)[i,k] * C[i,k] for all C
        # = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
        # So T[i*m+j, j'*p+k', i''*p+k''] = 1 if j=j' and i=i'' and k=k''
        # i.e., T[a, b, c] = 1 if the (i,j) index of a, (j',k') index of b, and (i'',k'') index of c
        # satisfy j=j', i=i'', k=k''.
        #
        # In our factorization: T = sum_r u_r (x v_r (x w_r
        # where u_r is in R^(n*m), v_r in R^(m*p), w_r in R^(n*p)
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        #
        # So we need: sum_r U[i*m+j, r] V[j'*p+k', r] W[i''*p+k'', r] = 1 iff j=j' and i=i'' and k=k''
        
        # Efficient check: for random inputs
        np.random.seed(42)
        for trial in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            # Compute via decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j*p+k',r]*B[j,k']) * W[k*n+i,r]
            # Actually: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
            # Wait, that's not right either. Let me re-derive.
            
            # C[i,k] = sum_j A[i,j] B[j,k]
            # In the tensor decomposition: C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] summed over j?
            # No. The factorization is: the bilinear form is sum_r <u_r, A> <v_r, B> <w_r, C>
            # where <u_r, A> = sum_{i,j} U[i*m+j, r] A[i,j]
            # <v_r, B> = sum_{j,k} V[j*p+k, r] B[j,k]  
            # <w_r, C> = sum_{i,k} W[i*p+k, r] C[i,k]... 
            # 
            # Hmm, but our W has shape (p*n, rank) with index k*n+i.
            # So <w_r, C> = sum_{k,i} W[k*n+i, r] C[i,k]
            
            # The tensor equation is:
            # sum_r (<u_r, A>) (<v_r, B>) W[k*n+i, r] = C[i,k] for each (i,k)
            # But that's not how it works. The correct interpretation:
            # T is a 3-tensor, and T(a,b,c) = sum_r U[a,r] V[b,r] W[c,r]
            # For the matrix mult tensor: T(a,b,c) = sum_{i,j,k} A[i,j] B[j,k] C[i,k] * delta_{conditions}
            # 
            # Actually, I think the correct check is:
            # For each r, define:
            #   A_r = sum_j U[i*m+j, r] A[i,j] for each i  (this is a vector in R^n)
            #   B_r = sum_k V[j*p+k, r] B[j,k] for each j  (this is a vector in R^m)
            #   C_r = W[k*n+i, r] for each (i,k)  (this is a matrix in R^(n*p))
            # Then: C = sum_r outer(A_r, ?) ... 
            #
            # I'm overcomplicating this. Let me just verify numerically.
            # The decomposition gives: for each r, we have vectors u_r, v_r, w_r
            # such that the multiplication tensor = sum_r u_r (x v_r (x w_r
            # 
            # To check: pick random A, B, compute C = A@B
            # Then check: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
            # NO. That's not right. The tensor factorization means:
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # where a indexes (i,j) in A, b indexes (j',k') in B, c indexes (i'',k'') in C
            # And T[a,b,c] = 1 if j=j' and i=i'' and k=k'', else 0.
            #
            # So: sum_r U[i*m+j, r] V[j'*p+k', r] W[i''*p+k'', r] = 1 iff j=j' and i=i'' and k=k''
            #
            # To verify with actual matrices:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_j A[i,j] B[j,k] * 1
            # = sum_j A[i,j] B[j,k] * sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]  ... NO
            #
            # I think the issue is that W is indexed differently. Let me re-read the task.
            # "W[k*n+i,r]" - so W has p*n rows, indexed by (k,i)
            # 
            # The standard matrix multiplication tensor factorization:
            # mult(A,B) = sum_r (sum_{i,j} U[i,j,r] A[i,j]) * (sum_{j,k} V[j,k,r] B[j,k]) * ... 
            # No, that gives a scalar, not a matrix.
            #
            # Correct: The bilinear map is: (A,B) -> C where C[i,k] = sum_j A[i,j]B[j,k]
            # This can be written as: C = sum_r A_r (x B_r (x C_r where
            # A_r[i,j] = U[i*m+j, r], B_r[j,k] = V[j*p+k, r], C_r[i,k] = W[k*n+i, r]
            # and C = sum_r (A * A_r) (o) (B * B_r) (o) C_r ... no.
            #
            # Actually: C[i,k] = sum_j A[i,j] B[j,k]
            # In tensor form: C = sum_r (sum_j U[i*m+j,r] A[i,j]) * (sum_j V[j*p+k,r] B[j,k]) * W[k*n+i,r]
            # Wait, that would be: C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) * (sum_j V[j*p+k,r] B[j,k]) * W[k*n+i,r]
            # But the two sums over j are independent, so this doesn't simplify to sum_j A[i,j]B[j,k].
            #
            # I think the correct interpretation is:
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j' V[j'*p+k, r] B[j',k])
            # But that's a product of two separate sums, which doesn't match matrix multiplication.
            #
            # Let me look at this differently. The tensor T is in R^(n*m) (x R^(m*p) (x R^(n*p)
            # T[a, b, c] = 1 if the indices match appropriately.
            # a = i*m + j, b = j'*p + k', c = i''*p + k''
            # T[a,b,c] = 1 iff j=j' and i=i'' and k=k''
            #
            # Factorization: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            #
            # Now, the bilinear map: given A (flattened to a) and B (flattened to b),
            # C[c] = sum_{a,b} T[a,b,c] A[a] B[b] = sum_{a,b} sum_r U[a,r] V[b,r] W[c,r] A[a] B[b]
            # = sum_r W[c,r] (sum_a U[a,r] A[a]) (sum_b V[b,r] B[b])
            #
            # So: C[c] = sum_r W[c,r] * (U[:,r]^T A_flat) * (V[:,r]^T B_flat)
            # where c = i*p + k (or k*n+i per the convention)
            #
            # Let's define:
            #   u_r_A = sum_{i,j} U[i*m+j, r] A[i,j]
            #   v_r_B = sum_{j,k} V[j*p+k, r] B[j,k]
            # Then C[i,k] = sum_r W[k*n+i, r] * u_r_A * v_r_B
            #
            # Let's verify this with the Strassen construction!
            
            C_decomp = np.zeros((n, p))
            for r in range(len(U_arr[0])):
                u_val = 0.0
                for i in range(n):
                    for j in range(m):
                        u_val += U_arr[i*m+j, r] * A[i,j]
                v_val = 0.0
                for j in range(m):
                    for k in range(p):
                        v_val += V_arr[j*p+k, r] * B[j,k]
                for i in range(n):
                    for k in range(p):
                        C_decomp[i,k] += W_arr[k*n+i, r] * u_val * v_val
            
            if not np.allclose(C_decomp, C_true, atol=1e-9):
                return False
        return True
    
    # Verify the initial decomposition
    if not verify_decomposition(U_np, V_np, W_np):
        # Shouldn't happen, but return what we have
        return best_result
    
    # Try greedy rank reduction
    # For each rank index, check if removing it still gives a valid decomposition
    # (approximately, since exact removal might break things)
    
    # Actually, a better approach: find the rank of the tensor via numerical methods
    # Or try to find linear dependencies among the rank-1 terms.
    
    # Let's try a different approach: use the fact that we have 8 independent 2x2x2 blocks
    # and see if we can share rank terms between blocks.
    
    # For now, let's try to find if any single rank term can be removed
    for r_remove in range(rank):
        if time.time() > deadline:
            break
        
        # Create modified matrices without term r_remove
        U_mod = U_np.copy()
        V_mod = V_np.copy()
        W_mod = W_np.copy()
        U_mod[:, r_remove] = 0
        V_mod[:, r_remove] = 0
        W_mod[:, r_remove] = 0
        
        # Quick check: is the removed term zero?
        if np.allclose(U_mod[:, r_remove], 0) and np.allclose(V_mod[:, r_remove], 0) and np.allclose(W_mod[:, r_remove], 0):
            continue
        
        # Verify
        if verify_decomposition(U_mod, V_mod, W_mod):
            # We can remove this term!
            new_rank = rank - 1
            # Remove the column
            mask = [i for i in range(rank) if i != r_remove]
            U_new = U_np[:, mask]
            V_new = V_np[:, mask]
            W_new = W_np[:, mask]
            best_rank = new_rank
            best_result = {
                "U": U_new.tolist(),
                "V": V_new.tolist(),
                "W": W_new.tolist()
            }
            U_np = U_new
            V_np = V_new
            W_np = W_new
            rank = new_rank
            # Restart the search with the new rank
            r_remove = -1  # Reset loop index
            continue
    
    # Try removing pairs of terms
    if time.time() < deadline - 30:
        for r1 in range(rank):
            if time.time() > deadline:
                break
            for r2 in range(r1+1, rank):
                if time.time() > deadline:
                    break
                
                U_mod = U_np.copy()
                V_mod = V_np.copy()
                W_mod = W_np.copy()
                U_mod[:, r1] = 0
                U_mod[:, r2] = 0
                V_mod[:, r1] = 0
                V_mod[:, r2] = 0
                W_mod[:, r1] = 0
                W_mod[:, r2] = 0
                
                if verify_decomposition(U_mod, V_mod, W_mod):
                    mask = [i for i in range(rank) if i != r1 and i != r2]
                    U_new = U_np[:, mask]
                    V_new = V_np[:, mask]
                    W_new = W_np[:, mask]
                    best_rank = rank - 2
                    best_result = {
                        "U": U_new.tolist(),
                        "V": V_new.tolist(),
                        "W": W_new.tolist()
                    }
                    U_np = U_new
                    V_np = V_new
                    W_np = W_new
                    rank = best_rank
                    break
    
    # Ensure we return valid format
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
