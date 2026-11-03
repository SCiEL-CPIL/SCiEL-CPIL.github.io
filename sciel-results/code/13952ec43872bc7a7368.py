"""Matrix multiplication 3x4x6 with bilinear circuit optimization."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
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

    # Bilinear circuit optimization: find linear dependencies among rank-1 terms
    def optimize_circuit(U, V, W, max_iter=50):
        """
        Given a decomposition with rank R, try to reduce rank by finding
        linear dependencies. For each pair of terms, check if one can be
        expressed as a linear combination of others.
        
        Key insight: if column r of U can be written as sum_r' alpha_r' * U[:,r'],
        then we need corresponding adjustments in V and W.
        """
        R = U.shape[1]
        if R < 2:
            return U, V, W, R
        
        # Build the "bilinear form" matrix: for each rank-1 term, we have
        # U_col, V_col, W_col. The decomposition is valid if:
        # sum_r U[:,r] * V[:,r] * W[:,r] (broadcast) == tensor
        #
        # To reduce rank, we look for a linear relation:
        # U[:,r0] = sum_{r != r0} c_r * U[:,r]
        # AND simultaneously V[:,r0] and W[:,r0] must satisfy
        # V[:,r0]*W[:,r0] = sum_{r != r0} c_r * V[:,r]*W[:,r]
        # This is hard to satisfy simultaneously, so we use a different approach.
        
        # Alternative: use the fact that the tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # We can try to find a change of basis in the rank space that zeros out
        # some columns.
        
        # Method: For each candidate column to eliminate, solve for coefficients
        # that make the contribution of that column cancel with others.
        
        # First, compute the tensor
        T = np.zeros((n, m, p))
        for r in range(R):
            T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), R
        
        # Try eliminating columns one at a time
        for r0 in range(R - 1, -1, -1):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            # Try to express term r0 as combination of other terms
            # We need: U[:,r0]*V[:,r0]*W[:,r0] = sum_{r!=r0} c_r * U[:,r]*V[:,r]*W[:,r]
            # This is a trilinear constraint - hard to solve directly.
            
            # Simpler approach: check if the column r0 is redundant by checking
            # if removing it and adjusting others still gives the correct tensor.
            
            # Use a greedy approach: try to find small perturbation that eliminates r0
            # by adjusting other columns.
            
            # Actually, let's try a different strategy: use SVD-like decomposition
            # of the "coefficient matrix" to find low-rank structure.
            
            # For each output element (i,k), the contribution is:
            # sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
            # This is equivalent to: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
            
            # Let's try: for each r0, see if we can find c_r such that
            # for all (i,j,k): U[i,j,r0]*V[j,k,r0]*W[k,i,r0] = sum_{r!=r0} c_r * U[i,j,r]*V[j,k,r]*W[k,i,r]
            # This is a system of n*m*p equations in (R-1) unknowns.
            
            # Build the linear system
            A_mat = np.zeros((n*m*p, R-1))
            b_vec = np.zeros(n*m*p)
            
            idx = 0
            other_cols = [r for r in range(R) if r != r0]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        b_vec[idx] = U[i*m+j, r0] * V[j*p+k, r0] * W[k*n+i, r0]
                        for ci, r in enumerate(other_cols):
                            A_mat[idx, ci] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        idx += 1
            
            # Solve least squares
            try:
                c, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
                # Check if the solution is good
                recon_error = np.linalg.norm(A_mat @ c - b_vec) / max(1.0, np.linalg.norm(b_vec))
                if recon_error < 1e-8:
                    # We can eliminate column r0
                    # New decomposition: for r != r0, new_V[:,r] = V[:,r] - c_r * V[:,r0]
                    # Wait, that's not quite right. Let me think again.
                    #
                    # Original: sum_r U_r * V_r * W_r = T
                    # We want: sum_{r!=r0} U_r * V'_r * W'_r = T
                    # where V'_r = V_r - c_r * V_r0 (or similar)
                    #
                    # Actually, the cleanest way: 
                    # T = U_r0*V_r0*W_r0 + sum_{r!=r0} U_r*V_r*W_r
                    # If U_r0*V_r0*W_r0 = sum_{r!=r0} c_r * U_r*V_r*W_r
                    # Then T = sum_{r!=r0} (1 + c_r) * U_r*V_r*W_r
                    # So we can just scale: V'_r = (1+c_r) * V_r for r != r0, and drop r0.
                    # But wait, that changes the product. Let me verify:
                    # sum_{r!=r0} U_r * V'_r * W_r = sum_{r!=r0} (1+c_r) U_r V_r W_r
                    # = sum_{r!=r0} U_r V_r W_r + sum_{r!=r0} c_r U_r V_r W_r
                    # = (T - U_r0 V_r0 W_r0) + U_r0 V_r0 W_r0 = T. Yes!
                    
                    new_R = R - 1
                    new_U = np.zeros((U.shape[0], new_R))
                    new_V = np.zeros((V.shape[0], new_R))
                    new_W = np.zeros((W.shape[0], new_R))
                    
                    ci = 0
                    for r in other_cols:
                        new_U[:, ci] = U[:, r]
                        new_V[:, ci] = (1.0 + c[ci]) * V[:, r]
                        new_W[:, ci] = W[:, r]
                        ci += 1
                    
                    # Verify
                    if check_decomposition(new_U, new_V, new_W):
                        best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_R
                        U, V, W, R = new_U, new_V, new_W, new_R
                        # Continue trying to eliminate more
                        r0 = max(0, r0 - 1)  # adjust index
            except Exception:
                pass
        
        return best_U, best_V, best_W, best_rank

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
            if time.time() - start_time > DEADLINE - 10:
                break
            try:
                U, V, W, rank = build_decomp(a_splits, b_splits)
                if check_decomposition(U, V, W):
                    # Try to optimize the circuit
                    U_opt, V_opt, W_opt, rank_opt = optimize_circuit(U, V, W)
                    if rank_opt < best_rank:
                        best_rank = rank_opt
                        best_result = (U_opt, V_opt, W_opt)
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
