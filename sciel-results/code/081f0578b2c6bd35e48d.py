"""Matrix multiplication 3x4x6 with optimized block decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    DEADLINE = 130  # Safety margin before 150s kill
    
    # Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1
        
        # M3 = A00(B01-B11)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1
        
        # M4 = A11(B10-B00)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1
        W_s[2, 3] = 1
        
        # M5 = (A00+A10)B11
        U_s[0, 4] = 1; U_s[2, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1
        W_s[3, 4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[2, 5] = 1
        W_s[3, 5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[1, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1
        
        return U_s, V_s, W_s
    
    # Schoolbook decomposition for 1x2x2
    def schoolbook_1x2x2():
        U_s = np.zeros((2, 4))
        V_s = np.zeros((4, 4))
        W_s = np.zeros((2, 4))
        
        col = 0
        for d in range(2):
            for c in range(2):
                U_s[d, col] = 1
                V_s[d*2+c, col] = 1
                W_s[c, col] = 1
                col += 1
        
        return U_s, V_s, W_s

    # Rank of 1x1x2 (single element times 1x2): rank 2
    def schoolbook_1x1x2():
        U_s = np.zeros((1, 2))
        V_s = np.zeros((2, 2))
        W_s = np.zeros((2, 2))
        U_s[0, 0] = 1; U_s[0, 1] = 1
        V_s[0, 0] = 1; V_s[1, 1] = 1
        W_s[0, 0] = 1; W_s[1, 1] = 1
        return U_s, V_s, W_s

    # Rank of 2x1x2: rank 4 (schoolbook)
    def schoolbook_2x1x2():
        U_s = np.zeros((2, 4))
        V_s = np.zeros((2, 4))
        W_s = np.zeros((2, 4))
        col = 0
        for a in range(2):
            for c in range(2):
                U_s[a, col] = 1
                V_s[c, col] = 1
                W_s[c, col] = 1
                col += 1
        return U_s, V_s, W_s

    # Build decomposition given a partition of A rows and B cols
    def build_decomp(a_splits, b_splits):
        """
        a_splits: list of sizes summing to n=3
        b_splits: list of sizes summing to p=6
        For each (a_block, b_block), we have a_block x 4 x b_block subproblem.
        We split the middle dim (4) into 2+2 and use:
          - if a_block==2 and b_block==2: Strassen 2x2x2 (rank 7)
          - if a_block==1 and b_block==2: schoolbook 1x2x2 (rank 4)
          - if a_block==2 and b_block==1: schoolbook 2x1x2 (rank 4)
          - if a_block==1 and b_block==1: schoolbook 1x1x2 (rank 2)
          - if a_block==3: split into 2+1
          - if b_block==3: split into 2+1
          - if b_block==4: split into 2+2
          - if b_block==6: split into 2+2+2
        """
        all_U = []
        all_V = []
        all_W = []
        
        def add_block(U_block, V_block, W_block):
            all_U.append(U_block)
            all_V.append(V_block)
            all_W.append(W_block)
        
        # Process each combination of A row block and B col block
        a_offset = 0
        for a_size in a_splits:
            b_offset = 0
            for b_size in b_splits:
                # Subproblem: a_size x 4 x b_size
                # Split middle into 2+2
                for j_start in [0, 2]:
                    # Sub-subproblem: a_size x 2 x b_size
                    if a_size == 2 and b_size == 2:
                        U_s, V_s, W_s = strassen_2x2x2()
                        U_g = np.zeros((12, 7))
                        V_g = np.zeros((24, 7))
                        W_g = np.zeros((18, 7))
                        for a in range(2):
                            for d in range(2):
                                local_row = a * 2 + d
                                global_i = a_offset + a
                                global_j = j_start + d
                                U_g[global_i * 4 + global_j, :] = U_s[local_row, :]
                        for d in range(2):
                            for c in range(2):
                                local_row = d * 2 + c
                                global_j = j_start + d
                                global_k = b_offset + c
                                V_g[global_j * 6 + global_k, :] = V_s[local_row, :]
                        for c in range(2):
                            for a in range(2):
                                local_row = c * 2 + a
                                global_k = b_offset + c
                                global_i = a_offset + a
                                W_g[global_k * 3 + global_i, :] = W_s[local_row, :]
                        add_block(U_g, V_g, W_g)
                    elif a_size == 1 and b_size == 2:
                        U_s, V_s, W_s = schoolbook_1x2x2()
                        U_g = np.zeros((12, 4))
                        V_g = np.zeros((24, 4))
                        W_g = np.zeros((18, 4))
                        for d in range(2):
                            global_i = a_offset
                            global_j = j_start + d
                            U_g[global_i * 4 + global_j, :] = U_s[d, :]
                        for d in range(2):
                            for c in range(2):
                                local_row = d * 2 + c
                                global_j = j_start + d
                                global_k = b_offset + c
                                V_g[global_j * 6 + global_k, :] = V_s[local_row, :]
                        for c in range(2):
                            local_row = c
                            global_k = b_offset + c
                            global_i = a_offset
                            W_g[global_k * 3 + global_i, :] = W_s[local_row, :]
                        add_block(U_g, V_g, W_g)
                    elif a_size == 2 and b_size == 1:
                        U_s, V_s, W_s = schoolbook_2x1x2()
                        U_g = np.zeros((12, 4))
                        V_g = np.zeros((24, 4))
                        W_g = np.zeros((18, 4))
                        for a in range(2):
                            for c in range(2):
                                global_i = a_offset + a
                                global_j = j_start
                                U_g[global_i * 4 + global_j, c] = U_s[a, c]
                        for c in range(2):
                            global_j = j_start
                            global_k = b_offset
                            V_g[global_j * 6 + global_k, :] = V_s[:, c]
                        for a in range(2):
                            for c in range(2):
                                global_k = b_offset
                                global_i = a_offset + a
                                W_g[global_k * 3 + global_i, c] = W_s[a, c]
                        add_block(U_g, V_g, W_g)
                    elif a_size == 1 and b_size == 1:
                        U_s, V_s, W_s = schoolbook_1x1x2()
                        U_g = np.zeros((12, 2))
                        V_g = np.zeros((24, 2))
                        W_g = np.zeros((18, 2))
                        for c in range(2):
                            global_i = a_offset
                            global_j = j_start
                            U_g[global_i * 4 + global_j, c] = U_s[0, c]
                            global_k = b_offset
                            V_g[global_j * 6 + global_k, c] = V_s[c, c]
                            W_g[global_k * 3 + global_i, c] = W_s[c, c]
                        add_block(U_g, V_g, W_g)
                    else:
                        # Fallback: schoolbook for this sub-subproblem
                        rank = a_size * 2 * b_size
                        U_g = np.zeros((12, rank))
                        V_g = np.zeros((24, rank))
                        W_g = np.zeros((18, rank))
                        col = 0
                        for a in range(a_size):
                            for d in range(2):
                                for c in range(b_size):
                                    U_g[(a_offset+a)*4 + (j_start+d), col] = 1
                                    V_g[(j_start+d)*6 + (b_offset+c), col] = 1
                                    W_g[(b_offset+c)*3 + (a_offset+a), col] = 1
                                    col += 1
                        add_block(U_g, V_g, W_g)
            b_offset += b_size
        a_offset += a_size
        
        total_rank = sum(U.shape[1] for U in all_U)
        U_final = np.zeros((12, total_rank))
        V_final = np.zeros((24, total_rank))
        W_final = np.zeros((18, total_rank))
        col_offset = 0
        for U_b, V_b, W_b in zip(all_U, all_V, all_W):
            r = U_b.shape[1]
            U_final[:, col_offset:col_offset+r] = U_b
            V_final[:, col_offset:col_offset+r] = V_b
            W_final[:, col_offset:col_offset+r] = W_b
            col_offset += r
        return U_final, V_final, W_final, total_rank

    def check_decomposition(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def reduce_rank_via_nullspace(U, V, W):
        """
        Given a valid decomposition of rank R, build the flattening matrix M where
        M[:, r] is the vectorized tensor entry for rank-1 component r.
        If components are linearly dependent, we can find a smaller set of
        independent components that span the same subspace.
        
        The tensor T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
        We form M of shape (n*m*p, R) where M[(i*m+j)*p+k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        Then the tensor is the sum over r of M[:,r] * (scalar 1), so T = M @ ones(R).
        Actually T_{i,j,k} = sum_r M[(i*m+j)*p+k, r]. So T = M @ 1.
        If M has rank r' < R, we can find a basis and reduce.
        But we need to express the same T with fewer rank-1 terms in the U,V,W structure.
        
        Alternative approach: find nullspace of M. If sum_r c_r * M[:,r] = 0 for some c,
        then we can't simply drop components. But if we can find a change of basis in the
        rank-1 components that zeros out some, we can reduce.
        
        Actually, the correct approach: The tensor T is fixed. Any decomposition T = sum_r u_r ⊗ v_r ⊗ w_r
        gives a point in the variety. The rank is the minimum number of terms.
        
        Given a specific decomposition with R terms, we can try to find if there's a
        linear combination of the R terms that gives the same tensor with some terms zeroed.
        This is equivalent to: find c in R^R such that sum_r c_r * T_r = T (where T_r is the
        rank-1 tensor for component r) and c has fewer non-zero entries than R.
        
        But sum_r T_r = T, so sum_r (c_r - 1) * T_r = 0. Let d_r = c_r - 1. We need
        sum_r d_r * T_r = 0 and c has fewer nonzeros.
        
        Equivalently: find a vector d in nullspace of M (flattening) such that 1 - d has
        fewer nonzeros than 1.
        
        Let's build M and find its nullspace, then search for a sparse 1-d.
        """
        R = U.shape[1]
        N = n * m * p  # 72
        
        # Build flattening matrix M: M[idx, r] where idx = (i*m+j)*p + k
        M = np.zeros((N, R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
        
        # Find rank of M
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = max(M.shape) * np.finfo(float).eps * s_vals[0]
        rank_M = np.sum(s_vals > tol)
        
        if rank_M >= R:
            return U, V, W, R  # No reduction possible
        
        # M has rank rank_M < R. Find nullspace.
        # Nullspace of M: vectors d such that M @ d = 0
        # From SVD: nullspace is spanned by last R - rank_M columns of V
        if rank_M < R:
            null_vecs = Vt_svd[rank_M:, :].T  # shape (R, R - rank_M)
            null_dim = R - rank_M
        else:
            null_vecs = np.zeros((R, 0))
            null_dim = 0
        
        if null_dim == 0:
            return U, V, W, R
        
        # We want to find d in nullspace(M) such that c = 1 - d has fewer nonzeros.
        # Equivalently, find d in nullspace(M) that is close to 1 (the all-ones vector)
        # in the sense that 1-d is sparse.
        # 
        # Project 1 onto nullspace: d_proj = null_vecs @ (null_vecs.T @ ones)
        # Then c = 1 - d_proj should have fewer nonzeros if the nullspace contains
        # directions close to 1.
        
        ones = np.ones(R)
        # Project ones onto nullspace
        if null_dim > 0:
            coeffs = null_vecs.T @ ones
            d_proj = null_vecs @ coeffs
            c = ones - d_proj
            
            # Try to make c sparse: find the closest sparse vector in the affine space
            # {ones + d : d in nullspace(M)}
            # This is: minimize ||c||_0 subject to M@c = M@ones = T (the tensor)
            # Actually M@c = T for any valid c.
            
            # Greedy: try zeroing out entries of c one at a time
            # Start with c, try to set each entry to 0 and re-project
            c_best = c.copy()
            for attempt in range(50):  # limited attempts
                # Find the smallest entry in c_best
                abs_c = np.abs(c_best)
                # Try zeroing the smallest entry
                min_idx = np.argmin(abs_c)
                if abs_c[min_idx] > 1e-8:
                    break
                
                # Set c[min_idx] = 0 and re-solve: find c' in affine space with c'[min_idx] = 0
                # Affine space: c = ones + null_vecs @ alpha for some alpha
                # Constraint: c[min_idx] = 0 => ones[min_idx] + null_vecs[min_idx, :] @ alpha = 0
                # This is one linear equation in alpha (null_dim unknowns)
                # If null_dim >= 1, we can satisfy this.
                
                # Actually, let's do it differently. We have c = ones + N @ alpha where N = null_vecs.
                # We want to find alpha such that c[min_idx] = 0:
                # ones[min_idx] + N[min_idx, :] @ alpha = 0
                # N[min_idx, :] @ alpha = -ones[min_idx]
                
                row = null_vecs[min_idx, :]
                if np.linalg.norm(row) < 1e-12:
                    continue
                # Least squares solution for alpha
                alpha_sol, _, _, _ = np.linalg.lstsq(row.reshape(1, -1), [-1.0], rcond=None)
                c_new = ones + null_vecs @ alpha_sol
                # Check if this is still valid (M @ c_new should equal T)
                # By construction it should be, but verify
                if np.max(np.abs(M @ c_new - M @ ones)) < 1e-6:
                    if np.sum(np.abs(c_new) > 1e-8) < np.sum(np.abs(c_best) > 1e-8):
                        c_best = c_new
            
            # Count nonzeros in c_best
            nonzeros = np.sum(np.abs(c_best) > 1e-8)
            if nonzeros < R:
                # Reconstruct U, V, W with scaled columns
                U_new = U * c_best[np.newaxis, :]
                V_new = V * c_best[np.newaxis, :]
                W_new = W * c_best[np.newaxis, :]
                # Remove zero columns
                mask = np.abs(c_best) > 1e-8
                U_final = U_new[:, mask]
                V_final = V_new[:, mask]
                W_final = W_new[:, mask]
                new_rank = int(mask.sum())
                return U_final, V_final, W_final, new_rank
        
        return U, V, W, R

    # Search over different partitions
    best_rank = float('inf')
    best_result = None
    
    # Possible A row splits (must sum to 3)
    a_split_options = [[2,1], [1,2], [1,1,1], [3]]
    # Possible B col splits (must sum to 6)
    b_split_options = [[2,2,2], [3,3], [2,4], [4,2], [1,2,3], [3,2,1], [1,1,4], [1,4,1], [4,1,1], [2,1,3], [3,1,2], [1,3,2], [2,3,1]]
    
    for a_splits in a_split_options:
        if sum(a_splits) != n:
            continue
        for b_splits in b_split_options:
            if sum(b_splits) != p:
                continue
            if time.time() - start_time > DEADLINE - 15:
                break
            try:
                U, V, W, rank = build_decomp(a_splits, b_splits)
                if check_decomposition(U, V, W):
                    # Try to reduce rank via nullspace
                    U_r, V_r, W_r, rank_r = reduce_rank_via_nullspace(U, V, W)
                    if rank_r < best_rank:
                        best_rank = rank_r
                        best_result = (U_r, V_r, W_r)
                    elif rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
            except Exception:
                pass
    
    if best_result is not None:
        U_final, V_final, W_final = best_result
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
