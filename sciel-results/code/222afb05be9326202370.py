import time
import numpy as np

def construct():
    _t0 = time.time()
    deadline = _t0 + 125.0

    n, m, p = 2, 4, 4

    def build_tensor():
        T = np.zeros((n*m, m*p, n*p), dtype=np.int64)
        for k in range(n):
            for j in range(m):
                for i in range(p):
                    T[k*m+j, j*p+i, k*p+i] = 1
        return T

    T = build_tensor()

    def verify(U, V, W):
        result = np.zeros((n*m, m*p, n*p), dtype=np.int64)
        for r in range(U.shape[1]):
            result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.array_equal(result, T)

    # Known rank-28 Strassen solution
    def strassen_28():
        rank = 28
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((n*p, rank), dtype=np.int64)
        
        def add_strassen_block(col_off, A_rows, B_rows, C_rows):
            U[A_rows[0], col_off+0] += 1
            U[A_rows[3], col_off+0] += 1
            V[B_rows[0], col_off+0] += 1
            V[B_rows[3], col_off+0] += 1
            W[C_rows[0], col_off+0] += 1
            W[C_rows[3], col_off+0] += 1
            
            U[A_rows[2], col_off+1] += 1
            U[A_rows[3], col_off+1] += 1
            V[B_rows[0], col_off+1] += 1
            W[C_rows[2], col_off+1] += 1
            W[C_rows[3], col_off+1] -= 1
            
            U[A_rows[0], col_off+2] += 1
            V[B_rows[1], col_off+2] += 1
            V[B_rows[3], col_off+2] -= 1
            W[C_rows[1], col_off+2] += 1
            W[C_rows[3], col_off+2] += 1
            
            U[A_rows[3], col_off+3] += 1
            V[B_rows[2], col_off+3] += 1
            V[B_rows[0], col_off+3] -= 1
            W[C_rows[0], col_off+3] += 1
            W[C_rows[2], col_off+3] += 1
            
            U[A_rows[0], col_off+4] += 1
            U[A_rows[1], col_off+4] += 1
            V[B_rows[3], col_off+4] += 1
            W[C_rows[0], col_off+4] -= 1
            W[C_rows[1], col_off+4] += 1
            
            U[A_rows[2], col_off+5] += 1
            U[A_rows[0], col_off+5] -= 1
            V[B_rows[0], col_off+5] += 1
            V[B_rows[1], col_off+5] += 1
            W[C_rows[3], col_off+5] += 1
            
            U[A_rows[1], col_off+6] += 1
            U[A_rows[3], col_off+6] -= 1
            V[B_rows[2], col_off+6] += 1
            V[B_rows[3], col_off+6] += 1
            W[C_rows[0], col_off+6] += 1
        
        add_strassen_block(0, [0,1,4,5], [0,1,4,5], [0,1,2,3])
        add_strassen_block(7, [2,3,6,7], [8,9,12,13], [4,5,6,7])
        add_strassen_block(14, [0,1,4,5], [2,3,6,7], [4,5,6,7])
        add_strassen_block(21, [2,3,6,7], [10,11,14,15], [4,5,6,7])
        
        return U, V, W

    U28, V28, W28 = strassen_28()
    assert verify(U28, V28, W28)

    # Now try to find lower rank.
    # Key insight: The 2x4x4 tensor has a specific structure that might allow
    # rank below 28. Let me try a direct search for small rank decompositions.
    
    # The tensor T has 32 nonzero entries: T[k*4+j, j*4+i, k*4+i] = 1
    # for k in {0,1}, j in {0,1,2,3}, i in {0,1,2,3}
    
    # For a rank-r decomposition with entries in {-1,0,1}, each rank-1 term
    # u⊗v⊗w contributes to at most (nnz_u)(nnz_v)(nnz_w) positions.
    
    # Let me try to construct a rank-24 solution by merging Strassen terms.
    # The idea: some of the 28 Strassen terms might be combinable.
    
    # Actually, let me try a completely different approach:
    # Use the "winograd" style for the 1x4x4 sub-problems.
    
    # For 1x4 x 4x4: C[i] = sum_j A[j] B[j,i]
    # This can be done in rank 4 (schoolbook) but maybe less?
    # The tensor is T[j, j*4+i, i] = 1 for j,i in 0..3
    # This is a 4x16x4 tensor. Its rank is at least 4 (slice rank argument).
    # Actually, it's exactly 4: use u=e_j, v=e_{j*4+i}, w=e_i doesn't work directly.
    # Better: for each j, u=e_j, v=e_{j*4+0}+...+e_{j*4+3}, w=... no.
    # Schoolbook: 4 terms: u=e_j, v=e_{j*4+i}, w=e_i for each (j,i) -> 16 terms
    # But we can do: u=e_j, v=sum_i e_{j*4+i}, w=... no that doesn't give separate outputs.
    # Actually for 1xn x nxn, the rank is n: use u=e_j, v=e_{j*0}, w=e_0; 
    # u=e_j, v=e_{j*1}, w=e_1; etc. That's n^2 terms for schoolbook.
    # Wait no. Let me reconsider.
    
    # For 1x4 x 4x4: C[i] = sum_j A[j] B[j,i]
    # Tensor: T[j, j*4+i, i] = 1
    # Rank-1 term: u ⊗ v ⊗ w where u has 4 entries, v has 16, w has 4
    # If u=e_j, v=e_{j*4+i}, w=e_i, that gives T[j, j*4+i, i] += 1. 
    # We need one term for each (j,i) pair, so 16 terms. That's schoolbook.
    # Can we do better? The rank of this tensor... 
    # It's related to the "matrix multiplication" tensor M_{1,4,4}.
    # Known: rank(M_{1,n,n}) = n^2 for n>=2? No, that's schoolbook.
    # Actually I think rank(M_{1,n,n}) can be less than n^2.
    
    # For our 2x4x4 problem, let me just try to verify if rank 28 is optimal
    # or if we can do better with a clever construction.
    
    # Given time constraints, let me try one more thing: a rank-24 construction
    # by exploiting the 2-row structure more aggressively.
    
    # Construction attempt for rank 24:
    # Think of it as: C = [A1 B; A2 B] where A1, A2 are 1x4
    # = [A1 B1 + A1 B2 + A1 B3 + A1 B4, A2 B1 + A2 B2 + A2 B3 + A2 B4]
    # where Bj is the j-th column of B (4x1)
    
    # Each Aj Bj is a 1x4 vector. We need to compute 8 such products
    # (2 rows of A, 4 columns of B) but they share structure.
    
    # Actually, let me just go with rank 28 which is verified.
    # A proper search for lower rank would require the Björck-Perne algorithm
    # or similar, which is complex to implement in the time budget.
    
    # But wait - let me try one more idea. What if I use a different 
    # block structure? Instead of 4 blocks of 2x2x2, what about
    # treating it as a 2x4x4 problem directly?
    
    # The lower bound for 2x4x4 matrix multiplication rank:
    # - Slice rank lower bound: max over slices
    # - The (0,0) slice: T[0, :, :] gives B's contribution to C[0,:]
    #   This is a 16x8 tensor with rank at least 4
    # - Similar for other slices
    # - Combined lower bound might be around 16-20
    
    # Upper bound is 28 (Strassen). The gap suggests room for improvement.
    
    # Let me try a direct small search for rank 20-27.
    # For rank r, I need to find U (8xr), V (16xr), W (8xr) with entries in 
    # a small set such that sum_r U[:,r]⊗V[:,r]⊗W[:,r] = T.
    
    # This is a system of 8*16*8 = 1024 equations in 32r unknowns.
    # For r=24: 768 unknowns, 1024 equations (overdetermined)
    # For r=28: 896 unknowns, 1024 equations (still overdetermined but feasible)
    
    # The search space is huge. Let me use a heuristic: start from rank-28
    # and try to merge columns.
    
    # Check: can any two columns of (U,V,W) be merged?
    # If U[:,r1] = U[:,r2] and V[:,r1] = V[:,r2] and W[:,r1] = W[:,r2],
    # then we can combine them into one column with doubled weight.
    
    # More generally, if U[:,r1] = a*U[:,r2], V[:,r1] = b*V[:,r2], 
    # W[:,r1] = c*W[:,r2] with a*b*c = 1, we might merge.
    
    # Let me check for proportional columns in the rank-28 solution:
    U_check = U28.astype(float)
    V_check = V28.astype(float)
    W_check = W28.astype(float)
    
    # Check for duplicate or proportional columns
    can_merge = False
    for r1 in range(28):
        if time.time() > deadline - 10:
            break
        for r2 in range(r1+1, 28):
            # Check if columns are proportional
            u1, u2 = U_check[:,r1], U_check[:,r2]
            v1, v2 = V_check[:,r1], V_check[:,r2]
            w1, w2 = W_check[:,r1], W_check[:,r2]
            
            # Skip if any is zero
            if np.all(u1==0) or np.all(u2==0): continue
            if np.all(v1==0) or np.all(v2==0): continue
            if np.all(w1==0) or np.all(w2==0): continue
            
            # Check proportionality
            try:
                # Find ratio from first nonzero
                mask_u = u1 != 0
                if np.any(mask_u):
                    ratio_u = u2[mask_u] / u1[mask_u]
                    if not np.allclose(ratio_u, ratio_u[0]): continue
                    ru = ratio_u[0]
                else: continue
                
                mask_v = v1 != 0
                if np.any(mask_v):
                    ratio_v = v2[mask_v] / v1[mask_v]
                    if not np.allclose(ratio_v, ratio_v[0]): continue
                    rv = ratio_v[0]
                else: continue
                
                mask_w = w1 != 0
                if np.any(mask_w):
                    ratio_w = w2[mask_w] / w1[mask_w]
                    if not np.allclose(ratio_w, ratio_w[0]): continue
                    rw = ratio_w[0]
                else: continue
                
                # Check if ru*rv*rw = 1 (for exact merging)
                if abs(ru*rv*rw - 1) < 1e-9:
                    can_merge = True
                    break
            except:
                continue
        if can_merge:
            break
    
    if not can_merge:
        # No simple merges. Return rank 28.
        pass
    
    # Return the verified rank-28 solution
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
