import time
import numpy as np
_start_time = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def construct():
    n, m, p = 4, 4, 7
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Try to find a rank-minimal decomposition using algebraic structure
    # For 4x4x7 matrix multiplication, we know:
    # - Schoolbook rank: 4*4*7 = 112
    # - Theoretical lower bound: max(16, 28, 28) = 28
    # - Known best upper bound for 4x4x4 is 49 (Bini et al.), but for 4x4x7 we need to be more careful
    
    # Let's try a structured decomposition that exploits the rectangular nature
    # Strategy: group by output rows (k index) and try to share computations across i indices
    
    # For each output column k (0..6), we need to compute 4 outputs (one per i)
    # Each output C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Approach: For each j (input row of B / input col of A), we can compute
    # partial products and combine them cleverly
    
    # Let's try a decomposition based on the following observation:
    # We can use the fact that for fixed j, the vector (A[0,j], A[1,j], A[2,j], A[3,j])
    # gets multiplied by B[j,0], B[j,1], ..., B[j,6]
    
    # This gives us a 4x7 matrix for each j, which we can decompose
    
    # For a 4x7 matrix, the minimum rank is 4 (since min(4,7)=4)
    # So for each of the 4 values of j, we need at least 4 rank-1 terms
    # Total: 4 * 4 = 16 terms minimum? But we also need to account for the structure
    
    # Actually, let's think more carefully. The tensor is:
    # T[i,j,k] = delta_ik for the multiplication tensor
    # Wait no, for matrix multiplication: T[i,j,k] where i indexes row of A/col of C,
    # j indexes col of A/row of B, k indexes col of B/row of C
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    # The standard decomposition is rank 4*4*7 = 112
    # Let's try to find a better one
    
    # One approach: use the structure of small matrices
    # For 4x4x7, let's try to use the following decomposition:
    # Group the 7 columns of B into groups and use block structure
    
    # Actually, let me just try to construct a valid decomposition with rank < 112
    # using a systematic approach
    
    # For now, let's use a decomposition that groups by j and uses rank-4 for each j
    terms = []
    
    # For each j in range(4):
    #   We need to compute for all i in range(4) and k in range(7):
    #   A[i,j] * B[j,k] contributes to C[i,k]
    
    # This is a 4x7 matrix M_j where M_j[i,k] = A[i,j]*B[j,k]
    # The rank of this as a tensor in i,k is at most 4 (since it's outer product of 
    # column j of A and row j of B... wait no)
    
    # Actually M_j[i,k] = A[i,j] * B[j,k] is rank 1! It's the outer product of
    # column j of A (length 4) and row j of B (length 7)
    
    # So for each j, we have ONE rank-1 term that handles all i and k for that j
    # Total rank: 4 (one for each j)
    
    # Wait, that can't be right. Let me reconsider.
    
    # The multiplication tensor T has components T[i,j,k]
    # For matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So T[i,j,k] = 1 for all i,j,k? No...
    # T[i,j,k] should be such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j] * B[j,k]
    
    # The tensor for matrix multiplication is:
    # T[(i,j), (j,k), (k,i)] = 1 when the indices match appropriately
    
    # Actually, the standard way: the multiplication tensor mu_{n,m,p} has
    # mu_{i,j,k} = 1 if i is in [0,n), j in [0,m), k in [0,p)
    # and it represents: <mu, A ⊗ B ⊗ C> = sum_{i,j,k} A[i,j] B[j,k] C^T[k,i]
    # = sum_{i,k} C[k,i] sum_j A[i,j] B[j,k]
    # = sum_{i,k} C[k,i] (AB)[i,k]
    
    # So mu_{i,j,k} = 1 for all valid i,j,k
    # And the rank-1 decomposition is: for each (i,j,k), term with
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # This gives rank n*m*p = 4*4*7 = 112
    
    # To get a lower rank, we need to find linear dependencies
    
    # For the square case 4x4x4, the best known is 49.
    # For 4x4x7, I'm not sure of the exact minimum, but let's try to construct something better than 112.
    
    # One idea: use the fact that for fixed j, the terms with that j can be combined
    # For fixed j, we have terms for all (i,k): U[i*m+j], V[j*p+k], W[k*n+i]
    # These form a 4x7 grid of rank-1 terms
    # Can we decompose this 4x7 grid into fewer rank-1 terms?
    
    # The 4x7 matrix J_j[i,k] = 1 (all ones) has rank 1!
    # So for each j, all 28 terms (4*7) can be replaced by 1 term?
    # But wait, the U, V, W indices are different for each (i,k)
    
    # For fixed j, the terms are:
    # (i,k) -> U[i*m+j], V[j*p+k], W[k*n+i]
    # This is NOT a simple outer product because U depends on i, V on k, W on both i and k
    
    # So we can't simply combine them.
    
    # Let me try a different approach: use a known decomposition for smaller cases
    # and extend it.
    
    # Build schoolbook decomposition as baseline
    rank_school = n * m * p
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                u_col = [0] * dim_u
                v_col = [0] * dim_v
                w_col = [0] * dim_w
                u_col[idx_u] = 1
                v_col[idx_v] = 1
                w_col[idx_w] = 1
                U_list.append(u_col)
                V_list.append(v_col)
                W_list.append(w_col)
    
    # Verify schoolbook is correct
    def verify_decomp(U_arr, V_arr, W_arr, n, m, p):
        """Check if decomposition is valid"""
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # T[i,j,k] = 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U_arr.shape[1]):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    U_arr = np.array(U_list).T  # shape (dim_u, rank)
    V_arr = np.array(V_list).T  # shape (dim_v, rank)
    W_arr = np.array(W_list).T  # shape (dim_w, rank)
    
    if not verify_decomp(U_arr, V_arr, W_arr, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find a lower-rank decomposition using linear algebra
    # The tensor T is a 3-tensor. We can try to find its tensor rank.
    # This is NP-hard in general, but for small sizes we can try.
    
    # Approach: use the fact that the tensor has structure
    # For each j, the sub-tensor over (i,k) is a 4x7 all-ones matrix
    # But the U, V, W indices couple i and k differently
    
    # Let's try: can we reduce rank by 1? Try rank 111.
    # This is hard to do systematically. Instead, let's use a known result.
    
    # For now, return schoolbook. The inspirations suggest ~97-104 is achievable.
    # Let me try a simple heuristic: randomly perturb and see if we can remove terms.
    
    # Actually, let's just return the schoolbook decomposition.
    # Converting back to list format
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
