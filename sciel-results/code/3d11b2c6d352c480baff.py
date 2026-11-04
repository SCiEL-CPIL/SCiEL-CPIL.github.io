# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.polynomial import polynomial as P

def construct():
    t0 = time.time()
    n, m, p = 2, 3, 10

    def deadline_ok():
        return (time.time() - t0) < 140.0

    # Start with the known rank-55 Strassen+schoolbook construction
    rank0 = 55
    U_mat = [[0] * rank0 for _ in range(n * m)]
    V_mat = [[0] * rank0 for _ in range(m * p)]
    W_mat = [[0] * rank0 for _ in range(p * n)]

    terms = []

    # Strassen for 5 pairs of columns in B
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        a00, a01, a10, a11 = 0, 1, 3, 4
        b0k0, b0k1 = 0 * 10 + k0, 0 * 10 + k1
        b1k0, b1k1 = 1 * 10 + k0, 1 * 10 + k1
        c0k0, c0k1 = k0 * 2 + 0, k1 * 2 + 0
        c1k0, c1k1 = k0 * 2 + 1, k1 * 2 + 1

        terms.append(([(a00, 1), (a11, 1)], [(b0k0, 1), (b1k1, 1)], [(c0k0, 1), (c1k1, 1)]))
        terms.append(([(a10, 1), (a11, 1)], [(b0k0, 1)], [(c1k0, 1), (c1k1, -1)]))
        terms.append(([(a00, 1)], [(b0k1, 1), (b1k1, -1)], [(c0k1, 1), (c1k1, 1)]))
        terms.append(([(a11, 1)], [(b1k0, 1), (b0k0, -1)], [(c0k0, 1), (c1k0, 1)]))
        terms.append(([(a00, 1), (a01, 1)], [(b1k1, 1)], [(c0k0, -1), (c0k1, 1)]))
        terms.append(([(a10, 1), (a00, -1)], [(b0k0, 1), (b0k1, 1)], [(c1k1, 1)]))
        terms.append(([(a01, 1), (a11, -1)], [(b1k0, 1), (b1k1, 1)], [(c0k0, 1)]))

    # Schoolbook for third row of A
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))

    for t_idx, (u_terms, v_terms, w_terms) in enumerate(terms):
        for idx, coeff in u_terms:
            U_mat[idx][t_idx] = coeff
        for idx, coeff in v_terms:
            V_mat[idx][t_idx] = coeff
        for idx, coeff in w_terms:
            W_mat[idx][t_idx] = coeff

    best_U, best_V, best_W = U_mat, V_mat, W_mat
    best_rank = rank0

    # Try to reduce rank by removing linearly dependent columns
    def try_reduce(U, V, W):
        """Attempt to reduce rank by finding and removing redundant tensor terms."""
        r = len(U[0])
        # Build the tensor in 3D form
        T = np.zeros((n*m, m*p, p*n))
        for t in range(r):
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        T[i,j,k] += U[i][t] * V[j][t] * W[k][t]
        return T

    def reduce_rank(U, V, W, max_iters=3):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for _ in range(max_iters):
            if not deadline_ok():
                break
            # Check if any column is linearly dependent on others in the "fiber" sense
            # Simpler: check if removing a term and re-solving keeps tensor intact
            # For now, try random pair eliminations
            for t_remove in range(r):
                if not deadline_ok():
                    break
                # Try to express term t_remove as combination of others
                # This is a linear system: sum_{t != t_remove} c_t * term_t = term_{t_remove}
                # Each term is a vector in R^{nm * mp * pn}
                # Too expensive for large dimensions, so use a simpler heuristic:
                # Check if U column is in span of other U columns, AND same for V, AND same for W
                U_other = np.delete(U_arr, t_remove, axis=1)
                V_other = np.delete(V_arr, t_remove, axis=1)
                W_other = np.delete(W_arr, t_remove, axis=1)
                
                u_col = U_arr[:, t_remove]
                v_col = V_arr[:, t_remove]
                w_col = W_arr[:, t_remove]
                
                # Check if u_col is in column space of U_other
                try:
                    coeff_u, _, _, _ = np.linalg.lstsq(U_other, u_col, rcond=None)
                    residual_u = np.linalg.norm(U_other @ coeff_u - u_col)
                except:
                    continue
                
                if residual_u > 1e-8:
                    continue
                    
                try:
                    coeff_v, _, _, _ = np.linalg.lstsq(V_other, v_col, rcond=None)
                    residual_v = np.linalg.norm(V_other @ coeff_v - v_col)
                except:
                    continue
                    
                if residual_v > 1e-8:
                    continue
                    
                try:
                    coeff_w, _, _, _ = np.linalg.lstsq(W_other, w_col, rcond=None)
                    residual_w = np.linalg.norm(W_other @ coeff_w - w_col)
                except:
                    continue
                    
                if residual_w > 1e-8:
                    continue
                
                # Check if the combination actually reproduces the tensor term
                # The term must be expressible, not just each factor individually
                # Verify by checking the outer product sum
                # This is a sufficient condition check
                # If all three are in span, the tensor term might be removable
                # but we need to verify the actual tensor equality
                pass  # For safety, skip actual removal without full verification
        
        return U_arr, V_arr, W_arr

    # Alternative: use rank revelation via Gaussian elimination on the tensor
    # Reshape tensor to matrix and find null space
    def find_reductions(U, V, W, max_attempts=50):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build full tensor
        T = np.einsum('it,jt,kt->ijk', U_arr, V_arr, W_arr)
        
        # Try to find a lower-rank representation
        # For each potential removal, check if tensor is preserved
        for attempt in range(max_attempts):
            if not deadline_ok():
                break
            # Randomly pick a term to try removing
            t_remove = np.random.randint(0, r)
            
            # Try to find coefficients c such that term_t_remove = sum_{t != t_remove} c_t * term_t
            # In tensor form: T[:, :, :] term t_remove
            term_removed = np.einsum('i,j,k->ijk', U_arr[:, t_remove], V_arr[:, t_remove], W_arr[:, t_remove])
            
            # Build matrix of other terms
            other_indices = [i for i in range(r) if i != t_remove]
            if len(other_indices) == 0:
                break
                
            # Flatten other terms
            A_mat = np.zeros((n*m*m*p*p*n, len(other_indices)))
            for idx, t in enumerate(other_indices):
                term_t = np.einsum('i,j,k->ijk', U_arr[:, t], V_arr[:, t], W_arr[:, t])
                A_mat[:, idx] = term_t.flatten()
            
            b_vec = term_removed.flatten()
            
            try:
                coeff, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                residual = np.linalg.norm(A_mat @ coeff - b_vec)
            except:
                continue
            
            if residual < 1e-7:
                # Remove this term
                new_U = np.delete(U_arr, t_remove, axis=1)
                new_V = np.delete(V_arr, t_remove, axis=1)
                new_W = np.delete(W_arr, t_remove, axis=1)
                return new_U, new_V, new_W, True
        
        return U_arr, V_arr, W_arr, False

    # Apply rank reduction
    U_arr = np.array(U_mat, dtype=float)
    V_arr = np.array(V_mat, dtype=float)
    W_arr = np.array(W_mat, dtype=float)
    
    for round in range(3):
        if not deadline_ok():
            break
        U_new, V_new, W_new, success = find_reductions(U_arr, V_arr, W_arr, max_attempts=100)
        if success:
            U_arr, V_arr, W_arr = U_new, V_new, W_new
            best_rank = len(U_arr[0])
            best_U, best_V, best_W = U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
        else:
            break

    return {'U': best_U, 'V': best_V, 'W': best_W}
# EVOLVE-BLOCK-END
