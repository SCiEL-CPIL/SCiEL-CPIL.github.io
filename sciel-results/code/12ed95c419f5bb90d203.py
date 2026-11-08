import time
import numpy as np

DIMENSIONS = [4, 4, 5]
TIME_LIMIT_SECONDS = 110
START_TIME = time.time()

def _elapsed() -> float:
    return time.time() - START_TIME

def construct():
    n, m, p = DIMENSIONS
    
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
        
    # Strassen 4x4x4
    super_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1])
    ]
    
    c_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])
    ]
    
    s_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    def get_p_global(p_elem_idx, p_blocks):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0: continue
            if b_idx == 0: gi, gj = r, c
            elif b_idx == 1: gi, gj = r, 2+c
            elif b_idx == 2: gi, gj = 2+r, c
            elif b_idx == 3: gi, gj = 2+r, 2+c
            res.append((gi, gj, coeff))
        return res

    def get_q_global(q_elem_idx, q_blocks):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0: continue
            if b_idx == 0: gj, gk = r, c
            elif b_idx == 1: gj, gk = r, 2+c
            elif b_idx == 2: gj, gk = 2+r, c
            elif b_idx == 3: gj, gk = 2+r, 2+c
            res.append((gj, gk, coeff))
        return res

    for s_idx in range(7):
        p_blocks, q_blocks = super_terms[s_idx]
        for t_idx in range(7):
            p_elems, q_elems = inner_terms[t_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            for p_idx, p_coeff in p_elems:
                elems = get_p_global(p_idx, p_blocks)
                for i, j, c in elems:
                    u_list.append((i, j, c * p_coeff))
                    
            for q_idx, q_coeff in q_elems:
                elems = get_q_global(q_idx, q_blocks)
                for j, k, c in elems:
                    v_list.append((j, k, c * q_coeff))
            
            for s_pos in range(4):
                sc = s_contrib[t_idx][s_pos]
                if sc == 0: continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                
                for c_pos in range(4):
                    cc = c_contrib[s_idx][c_pos]
                    if cc == 0: continue
                    
                    total_coeff = sc * cc
                    if total_coeff == 0: continue
                    
                    if c_pos == 0: gi, gk = s_r, s_c
                    elif c_pos == 1: gi, gk = s_r, 2+s_c
                    elif c_pos == 2: gi, gk = 2+s_r, s_c
                    elif c_pos == 3: gi, gk = 2+s_r, 2+s_c
                    
                    w_list.append((gk, gi, total_coeff))
            
            add_term(u_list, v_list, w_list)

    # Schoolbook 4x4x1
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(u_list, v_list, w_list)

    # Self-check (vectorized)
    if _elapsed() < TIME_LIMIT_SECONDS - 10:
        try:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            
            if not np.allclose(result, 1.0, atol=1e-10):
                return {}
        except Exception:
            return {}

    # Attempt rank reduction via linear dependence detection
    if _elapsed() < TIME_LIMIT_SECONDS - 15:
        try:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            num_terms = rank
            
            # Build the tensor contribution matrix: each column r contributes
            # a vector in R^(n*m*p). If columns are linearly dependent, we can remove some.
            # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            U_r = U_np.reshape(n, m, num_terms)
            V_r = V_np.reshape(m, p, num_terms)
            W_r = W_np.reshape(p, n, num_terms)
            
            # For each r, compute the flattened tensor contribution
            # Use einsum to get all contributions at once:
            # contrib[i,j,k,r] = U[i,j,r] * V[j,k,r] * W[k,i,r]
            contrib = np.einsum('ijr,jkr,kir->ijk r', U_r, V_r, W_r)
            # contrib shape: (n, m, p, num_terms)
            
            # Flatten to get a matrix of shape (n*m*p, num_terms)
            contrib_mat = contrib.reshape(n * m * p, num_terms)
            
            # Find the rank of this matrix
            # If rank < num_terms, there are linear dependencies
            # Use SVD to find the null space
            try:
                U_svd, s_vals, Vt = np.linalg.svd(contrib_mat, full_matrices=False)
                # The null space dimension is num_terms - rank(contrib_mat)
                # Find the rank by counting singular values above tolerance
                tol = max(contrib_mat.shape) * s_vals[0] * np.finfo(float).eps * 10
                matrix_rank = np.sum(s_vals > tol)
                
                if matrix_rank < num_terms:
                    # There are linear dependencies. Find null space vectors.
                    # Null space is spanned by last (num_terms - matrix_rank) rows of Vt
                    null_space = Vt[matrix_rank:]  # shape: (num_terms - matrix_rank, num_terms)
                    
                    # Try to remove columns greedily based on null space structure
                    # A column r can be removed if there's a null vector with nonzero entry at r
                    # and the remaining columns still span the same tensor space.
                    
                    # Greedy: try removing each column, verify with full einsum
                    # But use the null space info to prioritize which columns to try first
                    
                    # For each null space vector, the columns with largest |component| are candidates
                    # for removal (they can be expressed in terms of others)
                    
                    # Collect candidate columns to try removing
                    candidates = list(range(num_terms))
                    
                    # Sort candidates by how "redundant" they are
                    # A column is more redundant if it has small norm in the quotient space
                    # Simple heuristic: try all, but order by column norm (smaller = more likely redundant)
                    col_norms = np.linalg.norm(contrib_mat, axis=0)
                    candidates.sort(key=lambda r: col_norms[r])
                    
                    best_rank = num_terms
                    best_mask = list(range(num_terms))
                    
                    # Greedy removal: try removing columns one at a time
                    for r_remove in candidates:
                        if _elapsed() > TIME_LIMIT_SECONDS - 5:
                            break
                        
                        new_mask = [x for x in best_mask if x != r_remove]
                        if len(new_mask) == 0:
                            break
                        
                        U_new = U_np[:, new_mask]
                        V_new = V_np[:, new_mask]
                        W_new = W_np[:, new_mask]
                        
                        U_r_new = U_new.reshape(n, m, -1)
                        V_r_new = V_new.reshape(m, p, -1)
                        W_r_new = W_new.reshape(p, n, -1)
                        
                        result_new = np.einsum('ijr,jkr,kir->ijk', U_r_new, V_r_new, W_r_new)
                        
                        if np.allclose(result_new, 1.0, atol=1e-10):
                            best_rank = len(new_mask)
                            best_mask = new_mask
            
                    if best_rank < num_terms:
                        rank = best_rank
                        U = U_np[:, best_mask].tolist()
                        V = V_np[:, best_mask].tolist()
                        W = W_np[:, best_mask].tolist()
            except Exception:
                pass
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
