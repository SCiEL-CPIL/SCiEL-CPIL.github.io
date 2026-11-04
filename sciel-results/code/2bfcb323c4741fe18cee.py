import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0

    # Build rank-98 Strassen block decomposition
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0: (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    # c=1: (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    # c=2: A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    # c=3: A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    # c=4: (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    # c=5: (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    # c=6: (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]

    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]

    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]

    # Build the full rank-98 decomposition
    rank_initial = 98
    U_full = np.zeros((16, rank_initial))
    V_full = np.zeros((32, rank_initial))
    W_full = np.zeros((32, rank_initial))

    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s

                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            a_idx = gr * 4 + gc
                            eff = r * 2 + c
                            U_full[a_idx, col] += coeff_a * U_2[eff, m_s]

                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            gk = gc + half * 4
                            b_idx = gr * 8 + gk
                            eff = r * 2 + c
                            V_full[b_idx, col] += coeff_b * V_2[eff, m_s]

                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bi * 2 + r
                            gk = bj * 2 + c + half * 4
                            w_idx = gk * 4 + gi
                            eff = r * 2 + c
                            W_full[w_idx, col] += coeff_c * W_2[eff, m_s]

    if time.time() > deadline - 20:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Find linear dependencies: a term k is redundant if its (U,V,W) column
    # is in the span of other terms' columns
    # But we must be careful: removing a term changes the decomposition unless
    # we can express it as a linear combination of others
    
    # Actually, for tensor rank, we can only remove a term if there's a linear
    # dependence among the triples (U_k, V_k, W_k)
    
    # Let's try a different approach: use the fact that the two halves might
    # have shared structure. Check if any term in half 0 is a scalar multiple
    # of a term in half 1
    
    best_rank = rank_initial
    best_U, best_V, best_W = U_full, V_full, W_full
    
    # Try to find columns that are linearly dependent
    # Build a matrix where each row is a "feature" and each column is a term
    # Feature: for each (i,j) in U, (j,k) in V, (k,i) in W, the product
    # Actually, simpler: just check pairwise linear dependence
    
    # For each pair of columns, check if one is a scalar multiple of the other
    remove_set = set()
    for i in range(rank_initial):
        if i in remove_set:
            continue
        for j in range(i + 1, rank_initial):
            if j in remove_set:
                continue
            # Check if column j is a scalar multiple of column i
            col_i = np.concatenate([U_full[:, i], V_full[:, i], W_full[:, i]])
            col_j = np.concatenate([U_full[:, j], V_full[:, j], W_full[:, j]])
            
            # Find the ratio from the first non-zero element
            nonzero_i = np.nonzero(col_i)[0]
            if len(nonzero_i) == 0:
                continue
            idx = nonzero_i[0]
            if col_i[idx] == 0:
                continue
            ratio = col_j[idx] / col_i[idx]
            
            # Check if col_j == ratio * col_i
            if np.allclose(col_j, ratio * col_i, atol=1e-10):
                remove_set.add(j)
                break
    
    if len(remove_set) > 0:
        keep_cols = [i for i in range(rank_initial) if i not in remove_set]
        U_reduced = U_full[:, keep_cols]
        V_reduced = V_full[:, keep_cols]
        W_reduced = W_full[:, keep_cols]
        
        # Verify correctness
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 8)
        C_expected = A_test @ B_test
        
        # The decomposition is: C = sum_k (U_k outer V_k) * W_k
        # Where U_k is 4x4, V_k is 4x8, W_k is 8x4
        # Actually, the convention is:
        # C[i,k] = sum_j sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # Wait, let me re-read the task...
        # "W[k*n+i,r]" - so W is indexed by (k,i) where k is column of C, i is row of C
        # U is (n*m, rank) = (16, rank), so U[i*m+j, r]
        # V is (m*p, rank) = (32, rank), so V[j*p+k, r]
        
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_computed = np.zeros((8, 4))
        for r in range(len(keep_cols)):
            u_vec = U_reduced[:, r]
            v_vec = V_reduced[:, r]
            w_vec = W_reduced[:, r]
            
            u_mat = u_vec.reshape(4, 4)  # [i, j]
            v_mat = v_vec.reshape(4, 8)  # [j, k]
            w_mat = w_vec.reshape(8, 4)  # [k, i]
            
            # C[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is: for each i, k: sum_j u[i,j]*v[j,k]*w[k,i]
            # = (u * v) elementwise... no
            # Let's compute it as: sum_j u[i,j] * v[j,k] * w[k,i]
            # = sum_j (u * v)[i,j,k]... 
            # Actually: for fixed i,k: sum_j u[i,j]*v[j,k]*w[k,i]
            # w[k,i] is scalar for fixed k,i
            # So: w[k,i] * sum_j u[i,j]*v[j,k]
            # = w[k,i] * (u @ v)[i,k]
            
            uv = u_mat @ v_mat  # (4, 8), uv[i,k] = sum_j u[i,j]*v[j,k]
            C_computed += w_mat * uv  # elementwise: C[i,k] += w[k,i] * uv[i,k]
        
        if np.allclose(C_computed, C_expected, atol=1e-10):
            best_rank = len(keep_cols)
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
    
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
