import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 recursive Strassen decomposition for 4x4 matrix multiplication.
    Attempts to find rank-48 if time permits via linear dependency search.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_U_index(row, col):
        return row * m + col
    
    def get_V_index(row, col):
        return row * p + col
    
    def get_W_index(row, col):
        return col * n + row
    
    def a_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def b_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def c_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    L_terms = []
    R_terms = []
    contrib_terms = []
    
    # Term 0: P1 = (A11 + A22)(B11 + B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L0[r][c] = {(ar, ac): 1, (dr, dc): 1}
            er, ec = b_elem(0, 0, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R0[r][c] = {(er, ec): 1, (hr, hc): 1}
    contrib0 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib0[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib0[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L0); R_terms.append(R0); contrib_terms.append(contrib0)
    
    # Term 1: P2 = (A21 + A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(1, 0, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L1[r][c] = {(ar, ac): 1, (dr, dc): 1}
            er, ec = b_elem(0, 0, r, c)
            R1[r][c] = {(er, ec): 1}
    contrib1 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(1, 0, r, c)
            contrib1[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib1[(r, c)].append((cr2, cc2, -1))
    L_terms.append(L1); R_terms.append(R1); contrib_terms.append(contrib1)
    
    # Term 2: P3 = A11(B12 - B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            L2[r][c] = {(ar, ac): 1}
            fr, fc = b_elem(0, 1, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R2[r][c] = {(fr, fc): 1, (hr, hc): -1}
    contrib2 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 1, r, c)
            contrib2[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib2[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L2); R_terms.append(R2); contrib_terms.append(contrib2)
    
    # Term 3: P4 = A22(B21 - B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            dr, dc = a_elem(1, 1, r, c)
            L3[r][c] = {(dr, dc): 1}
            gr, gc = b_elem(1, 0, r, c)
            er, ec = b_elem(0, 0, r, c)
            R3[r][c] = {(gr, gc): 1, (er, ec): -1}
    contrib3 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib3[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 0, r, c)
            contrib3[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L3); R_terms.append(R3); contrib_terms.append(contrib3)
    
    # Term 4: P5 = (A11 + A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            br, bc = a_elem(0, 1, r, c)
            L4[r][c] = {(ar, ac): 1, (br, bc): 1}
            hr, hc = b_elem(1, 1, r, c)
            R4[r][c] = {(hr, hc): 1}
    contrib4 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib4[(r, c)] = [(cr, cc, -1)]
            cr2, cc2 = c_elem(0, 1, r, c)
            contrib4[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L4); R_terms.append(R4); contrib_terms.append(contrib4)
    
    # Term 5: P6 = (A21 - A11)(B11 + B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            cr, cc = a_elem(1, 0, r, c)
            ar, ac = a_elem(0, 0, r, c)
            L5[r][c] = {(cr, cc): 1, (ar, ac): -1}
            er, ec = b_elem(0, 0, r, c)
            fr, fc = b_elem(0, 1, r, c)
            R5[r][c] = {(er, ec): 1, (fr, fc): 1}
    contrib5 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(1, 1, r, c)
            contrib5[(r, c)] = [(cr, cc, 1)]
    L_terms.append(L5); R_terms.append(R5); contrib_terms.append(contrib5)
    
    # Term 6: P7 = (A12 - A22)(B21 + B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            br, bc = a_elem(0, 1, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L6[r][c] = {(br, bc): 1, (dr, dc): -1}
            gr, gc = b_elem(1, 0, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R6[r][c] = {(gr, gc): 1, (hr, hc): 1}
    contrib6 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib6[(r, c)] = [(cr, cc, 1)]
    L_terms.append(L6); R_terms.append(R6); contrib_terms.append(contrib6)
    
    sub_contrib = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    for t in range(7):
        L_t = L_terms[t]
        R_t = R_terms[t]
        contrib_t = contrib_terms[t]
        
        a = L_t[0][0]; b = L_t[0][1]; c = L_t[1][0]; d = L_t[1][1]
        e = R_t[0][0]; f = R_t[0][1]; g = R_t[1][0]; h = R_t[1][1]
        
        sub_products = []
        x1 = {}; y1 = {}
        for key, coeff in a.items(): x1[key] = x1.get(key, 0) + coeff
        for key, coeff in d.items(): x1[key] = x1.get(key, 0) + coeff
        for key, coeff in e.items(): y1[key] = y1.get(key, 0) + coeff
        for key, coeff in h.items(): y1[key] = y1.get(key, 0) + coeff
        sub_products.append((x1, y1))
        
        x2 = {}; y2 = dict(e)
        for key, coeff in c.items(): x2[key] = x2.get(key, 0) + coeff
        for key, coeff in d.items(): x2[key] = x2.get(key, 0) + coeff
        sub_products.append((x2, y2))
        
        x3 = dict(a); y3 = {}
        for key, coeff in f.items(): y3[key] = y3.get(key, 0) + coeff
        for key, coeff in h.items(): y3[key] = y3.get(key, 0) - coeff
        sub_products.append((x3, y3))
        
        x4 = dict(d); y4 = {}
        for key, coeff in g.items(): y4[key] = y4.get(key, 0) + coeff
        for key, coeff in e.items(): y4[key] = y4.get(key, 0) - coeff
        sub_products.append((x4, y4))
        
        x5 = {}; y5 = dict(h)
        for key, coeff in a.items(): x5[key] = x5.get(key, 0) + coeff
        for key, coeff in b.items(): x5[key] = x5.get(key, 0) + coeff
        sub_products.append((x5, y5))
        
        x6 = {}; y6 = {}
        for key, coeff in c.items(): x6[key] = x6.get(key, 0) + coeff
        for key, coeff in a.items(): x6[key] = x6.get(key, 0) - coeff
        for key, coeff in e.items(): y6[key] = y6.get(key, 0) + coeff
        for key, coeff in f.items(): y6[key] = y6.get(key, 0) + coeff
        sub_products.append((x6, y6))
        
        x7 = {}; y7 = {}
        for key, coeff in b.items(): x7[key] = x7.get(key, 0) + coeff
        for key, coeff in d.items(): x7[key] = x7.get(key, 0) - coeff
        for key, coeff in g.items(): y7[key] = y7.get(key, 0) + coeff
        for key, coeff in h.items(): y7[key] = y7.get(key, 0) + coeff
        sub_products.append((x7, y7))
        
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k
            for (row, col), coeff in x_k.items():
                u_idx = get_U_index(row, col)
                U[u_idx][rank_idx] += coeff
            for (row, col), coeff in y_k.items():
                v_idx = get_V_index(row, col)
                V[v_idx][rank_idx] += coeff
            for (pr, pc, sub_coeff) in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (out_row, out_col, block_coeff) in contrib_t[(pr, pc)]:
                        w_idx = get_W_index(out_row, out_col)
                        W[w_idx][rank_idx] += sub_coeff * block_coeff
    
    # Verify rank-49 decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    idx_i = np.repeat(np.arange(n), m*p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n*m)
    
    u_idx = idx_i * m + idx_j
    v_idx = idx_j * p + idx_k
    w_idx = idx_k * n + idx_i
    
    tensor_vals = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx], axis=1)
    
    if not np.allclose(tensor_vals, 1.0, atol=1e-10):
        # Fallback: standard rank-64
        U_std = np.zeros((16, 64))
        V_std = np.zeros((16, 64))
        W_std = np.zeros((16, 64))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r = i * 16 + j * 4 + k
                    U_std[i*4+j, r] = 1
                    V_std[j*4+k, r] = 1
                    W_std[k*4+i, r] = 1
        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U, "V": V, "W": W}
    
    # Search for rank-48: check if any of the 49 rank-1 tensors is a linear combination
    # of the others. If so, we can eliminate it.
    # Each rank-1 tensor T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] flattened to 4096-dim vector
    
    tensors = []
    for r in range(49):
        t = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r]).flatten()
        tensors.append(t)
    
    tensors_arr = np.array(tensors)  # shape (49, 4096)
    
    # Compute the rank of this matrix. If rank < 49, there's a dependency.
    try:
        # Use SVD for numerical stability
        _, s, vh = np.linalg.svd(tensors_arr, full_matrices=False)
        rank_found = np.sum(s > 1e-8)
        
        if rank_found < 49:
            # Find a dependency: there exists a non-trivial vector alpha such that
            # sum_r alpha_r * T_r = 0
            # The left null space of tensors_arr gives us this
            # Since tensors_arr is 49x4096, we need the right null space of its transpose
            # Or equivalently, find vectors in the null space of tensors_arr.T
            
            # Actually, we want: find alpha in R^49 such that tensors_arr.T @ alpha = 0
            # This means alpha is in the null space of tensors_arr.T
            # tensors_arr.T is 4096x49, so null space is in R^49
            
            # Use SVD: tensors_arr = U_svd @ diag(s) @ Vh
            # Null space of tensors_arr.T = null space of Vh.T @ diag(s) @ U_svd.T
            # = null space of (Vh.T @ diag(s)) since U_svd is orthonormal
            # The last (49 - rank_found) columns of Vh correspond to zero singular values
            
            # The right singular vectors corresponding to zero singular values span the null space
            # of tensors_arr.T, i.e., they give us the dependencies
            
            # Vh is 49x49 (since we used full_matrices=False and min(49,4096)=49)
            # The last (49-rank_found) rows of Vh span the null space
            
            null_vecs = Vh[rank_found:]  # shape (49-rank_found, 49)
            
            # For each null vector, we can potentially eliminate one term
            # If null_vec has a non-zero component at index j, we can express T_j as a combo of others
            
            best_reduction = None
            for nv in null_vecs:
                # Find the index with the largest absolute coefficient
                max_idx = np.argmax(np.abs(nv))
                max_val = nv[max_idx]
                if abs(max_val) > 1e-10:
                    # Try to eliminate term max_idx
                    # The dependency is: sum_r nv[r] * T_r = 0
                    # So T_max_idx = -sum_{r != max_idx} (nv[r]/nv[max_idx]) * T_r
                    # This means we can drop term max_idx and adjust the others
                    
                    # But we need to verify that the resulting 48-term decomposition is valid
                    # The new decomposition would have:
                    # U_new = U[:, [0..max_idx-1, max_idx+1..48]]  (48 columns)
                    # V_new = V[:, [0..max_idx-1, max_idx+1..48]]
                    # W_new = W[:, [0..max_idx-1, max_idx+1..48]]
                    # But wait - this doesn't work directly because the dependency is on the
                    # TENSORS, not on the individual U, V, W vectors.
                    
                    # Actually, if sum_r alpha_r * (U_r ⊗ V_r ⊗ W_r) = 0, this does NOT
                    # necessarily mean we can drop one term. The tensor decomposition is not
                    # simply a linear combination of rank-1 tensors in a way that allows
                    # elimination unless the rank-1 tensors are linearly dependent in a
                    # specific way.
                    
                    # For a valid tensor decomposition, we need:
                    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                    # If the 49 rank-1 tensors are linearly dependent, it doesn't automatically
                    # mean we can reduce the rank. We need to find a different decomposition.
                    
                    # However, if there's a linear dependency among the rank-1 tensors,
                    # it suggests the rank might be reducible. Let's try a direct approach:
                    # Check if we can find a 48-dimensional subspace that contains all
                    # the necessary information.
                    
                    # For now, let's just note that a dependency exists and try to
                    # construct a reduced decomposition.
                    
                    if best_reduction is None or abs(max_val) > best_reduction[1]:
                        best_reduction = (max_idx, abs(max_val), nv)
            
            if best_reduction is not None:
                # We found a potential dependency. Let's try to actually reduce the rank.
                # This is complex, so for now we'll just return the rank-49 result.
                # A full rank reduction would require solving a system of equations
                # to find new U, V, W matrices of size 48.
                pass
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
