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
    if _elapsed() < TIME_LIMIT_SECONDS - 5:
        try:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Contract over r: result[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
            # We can do this via: for each r, outer product contribution
            # Or more efficiently: reshape and use einsum
            # result[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            
            if not np.allclose(result, 1.0, atol=1e-10):
                return {}
        except Exception:
            return {}

    # Attempt rank reduction via linear dependence detection
    if _elapsed() < TIME_LIMIT_SECONDS - 30:
        try:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            # Each rank-1 term is (U[:,r], V[:,r], W[:,r])
            # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # If we can find a linear combination of columns that sums to zero,
            # we can remove that column and reduce rank.
            # 
            # More precisely: if there exist coefficients c_r such that
            # sum_r c_r * U[:,r] = 0, sum_r c_r * V[:,r] = 0, sum_r c_r * W[:,r] = 0
            # then we can subtract that combination. But that's trivial (all zeros).
            #
            # What we want: a subset of columns whose tensor contributions sum to zero.
            # i.e., sum_{r in S} U[i,j,r]*V[j,k,r]*W[k,i,r] = 0 for all i,j,k
            #
            # This is equivalent to: the vectors in the space of tensors spanned by
            # rank-1 terms are linearly dependent.
            #
            # Let's flatten each rank-1 term into a vector of size n*m*p
            # and check for linear dependencies.
            
            def flatten_term(r):
                """Flatten the r-th rank-1 term into a vector."""
                u = U_np[:, r].reshape(n, m)
                v = V_np[:, r].reshape(m, p)
                w = W_np[:, r].reshape(p, n)
                # Tensor T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
                t = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            t[i, j, k] = u[i, j] * v[j, k] * w[k, i]
                return t.flatten()
            
            # Build matrix where each column is a flattened rank-1 term
            # This is expensive for rank=65, but we can try with a time limit
            num_terms = rank
            tensor_dim = n * m * p
            
            # Instead of building the full matrix (65 x 80), let's use a smarter approach
            # Check if any single term can be removed (i.e., is it a linear combination of others?)
            # That's too expensive. Instead, let's try a greedy approach:
            # Try removing each column and check if the result is still valid.
            
            # Actually, a better approach: use SVD/null space on a random projection
            # But the simplest: try each column removal and verify.
            
            # With 65 columns, trying all 65 removals and verifying each takes 65 * check_time
            # Check time is fast with einsum. Let's try.
            
            U_orig = np.array(U, copy=True)
            V_orig = np.array(V, copy=True)
            W_orig = np.array(W, copy=True)
            
            best_rank = rank
            best_U, best_V, best_W = U_orig, V_orig, W_orig
            
            for r_remove in range(num_terms):
                if _elapsed() > TIME_LIMIT_SECONDS - 10:
                    break
                    
                # Remove column r_remove
                mask = [x for x in range(num_terms) if x != r_remove]
                U_new = U_orig[:, mask]
                V_new = V_orig[:, mask]
                W_new = W_orig[:, mask]
                
                # Check validity
                U_r_new = U_new.reshape(n, m, -1)
                V_r_new = V_new.reshape(m, p, -1)
                W_r_new = W_new.reshape(p, n, -1)
                
                result_new = np.einsum('ijr,jkr,kir->ijk', U_r_new, V_r_new, W_r_new)
                
                if np.allclose(result_new, 1.0, atol=1e-10):
                    new_rank = num_terms - 1
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U, best_V, best_W = U_new, V_new, W_new
            
            if best_rank < rank:
                rank = best_rank
                U = best_U.tolist()
                V = best_V.tolist()
                W = best_W.tolist()
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
