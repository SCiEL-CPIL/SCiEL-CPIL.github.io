import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 135.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def check_time():
        return time.time() - t0 < DEADLINE
    
    def validate(U, V, W, r):
        """Vectorized validation using NumPy."""
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                u_row = Ua[i*m + j, :]
                for k in range(p):
                    v_row = Va[j*p + k, :]
                    w_row = Wa[k*n + i, :]
                    tot = np.dot(u_row, v_row * w_row)
                    if tot != 1:
                        return False
        return True
    
    def validate_fast(U, V, W, r):
        """Even faster: precompute products."""
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        # Build the full tensor check
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape: U is (nm, r), V is (mp, r), W is (pn, r)
        # For efficiency, iterate over c and accumulate
        T = np.zeros((n, m, p), dtype=np.int64)
        for c in range(r):
            uc = Ua[:, c]  # (nm,)
            vc = Va[:, c]  # (mp,)
            wc = Wa[:, c]  # (pn,)
            # uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
            # Reshape uc to (n,m), vc to (m,p), wc to (p,n)
            ur = uc.reshape(n, m)
            vr = vc.reshape(m, p)
            wr = wc.reshape(p, n)
            # T[i,j,k] += ur[i,j] * vr[j,k] * wr[k,i]
            # This is like a tensor contraction
            # Loop over i,j,k would be slow; use einsum
            T += np.einsum('ij,jk,ki->ijk', ur, vr, wr)
        return np.all(T == 1)
    
    # Start with known 129-rank solution
    STRASSEN_TERMS = [
        {'a': [1, 0, 0, 1], 'b': [1, 0, 0, 1]},
        {'a': [0, 0, 0, 1], 'b': [-1, 0, 1, 0]},
        {'a': [0, 0, 1, 1], 'b': [0, 1, 0, 0]},
        {'a': [1, 0, 0, 0], 'b': [0, 1, 0, -1]},
        {'a': [1, 1, 0, 0], 'b': [1, 0, 0, 1]},
        {'a': [0, 1, 0, -1], 'b': [0, 0, 1, 1]},
        {'a': [-1, 0, 1, 0], 'b': [1, 1, 0, 0]},
    ]
    C_BLOCKS = [
        [1, 1, -1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, -1, 0],
    ]
    
    def generate_4x4x4_strassen_terms():
        terms = []
        for outer_s in range(7):
            a_coeff = STRASSEN_TERMS[outer_s]['a']
            b_coeff = STRASSEN_TERMS[outer_s]['b']
            for cbi in range(4):
                outer_sign = C_BLOCKS[cbi][outer_s]
                if outer_sign == 0:
                    continue
                ci_block, cj_block = divmod(cbi, 2)
                for inner_s in range(7):
                    ia_coeff = STRASSEN_TERMS[inner_s]['a']
                    ib_coeff = STRASSEN_TERMS[inner_s]['b']
                    for inner_cbi in range(4):
                        inner_sign = C_BLOCKS[inner_cbi][inner_s]
                        if inner_sign == 0:
                            continue
                        inner_ci, inner_cj = divmod(inner_cbi, 2)
                        c_row = ci_block * 2 + inner_ci
                        c_col = cj_block * 2 + inner_cj
                        c_global = c_row * 4 + c_col
                        for li in range(2):
                            for lj in range(2):
                                ia_c = ia_coeff[li*2+lj]
                                if ia_c == 0:
                                    continue
                                for ri in range(2):
                                    for rj in range(2):
                                        ib_c = ib_coeff[ri*2+rj]
                                        if ib_c == 0:
                                            continue
                                        for ab in range(4):
                                            ac = a_coeff[ab]
                                            if ac == 0:
                                                continue
                                            abr, abc = divmod(ab, 2)
                                            a_row = 2*abr + li
                                            a_col = 2*abc + lj
                                            a_global = a_row * 4 + a_col
                                            for bb in range(4):
                                                bc = b_coeff[bb]
                                                if bc == 0:
                                                    continue
                                                bbr, bbc = divmod(bb, 2)
                                                b_row = 2*bbr + ri
                                                b_col = 2*bbc + rj
                                                b_global = b_row * 4 + b_col
                                                coeff = outer_sign * inner_sign * ac * ia_c * bc * ib_c
                                                if coeff != 0:
                                                    terms.append((a_global, b_global, c_global, coeff))
        term_dict = {}
        for t in terms:
            key = (t[0], t[1], t[2])
            if key in term_dict:
                term_dict[key] += t[3]
            else:
                term_dict[key] = t[3]
        return [(a, b, c, coeff) for (a, b, c), coeff in term_dict.items() if coeff != 0]
    
    terms_4x4 = generate_4x4x4_strassen_terms()
    terms_left = []
    for (a, b, c, coeff) in terms_4x4:
        j = b // 4
        k_old = b % 4
        b_new = j * 9 + k_old
        terms_left.append((a, b_new, c, coeff))
    
    terms_right = []
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                a_global = i * 4 + j
                b_global = j * 9 + k
                c_global = k * 4 + i
                terms_right.append((a_global, b_global, c_global, 1))
    
    all_terms = terms_left + terms_right
    r_best = len(all_terms)
    
    U_best = [[0]*r_best for _ in range(nm)]
    V_best = [[0]*r_best for _ in range(mp)]
    W_best = [[0]*r_best for _ in range(pn)]
    
    for c, (a, b, cg, coeff) in enumerate(all_terms):
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = cg // n
        i2 = cg % n
        U_best[i*m + j][c] += coeff
        V_best[j*p + k][c] += 1
        W_best[k*n + i][c] += 1
    
    if not validate(U_best, V_best, W_best, r_best):
        r_fb = n * m * p
        U = [[0]*r_fb for _ in range(nm)]
        V = [[0]*r_fb for _ in range(mp)]
        W = [[0]*r_fb for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy for faster operations
    U_np = np.array(U_best, dtype=np.int64)
    V_np = np.array(V_best, dtype=np.int64)
    W_np = np.array(W_best, dtype=np.int64)
    
    # Local search: try to eliminate individual terms
    # For each term c, check if we can absorb it into other terms
    # Strategy: for each column c, try to set it to zero and see if we can
    # adjust other columns to compensate
    
    def try_eliminate(Ua, Va, Wa, r, col):
        """Try to eliminate column col by adjusting other columns.
        Returns (new_U, new_V, new_W, new_r) or None if failed."""
        # The contribution of column col to T[i,j,k] is:
        # Ua[i*m+j, col] * Va[j*p+k, col] * Wa[k*n+i, col]
        # We need to find adjustments to other columns that cancel this.
        # This is a system of linear equations over the integers.
        # For simplicity, try: can we add a multiple of some other column to cancel?
        # Actually, let's try a simpler approach: just remove the column and check
        # if the remaining still works (unlikely but cheap to try)
        
        # Better: try to find integer combinations
        # For now, just try removing and see
        if r <= 1:
            return None
        
        # Try: remove column col
        U_new = np.delete(Ua, col, axis=1)
        V_new = np.delete(Va, col, axis=1)
        W_new = np.delete(Wa, col, axis=1)
        
        # Check if still valid
        T = np.zeros((n, m, p), dtype=np.int64)
        for c in range(r - 1):
            ur = U_new[:, c].reshape(n, m)
            vr = V_new[:, c].reshape(m, p)
            wr = W_new[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', ur, vr, wr)
        
        if np.all(T == 1):
            return U_new, V_new, W_new, r - 1
        return None
    
    # Try eliminating terms using linear algebra
    # For each pair of columns, check if one can be expressed as a linear combination
    # of the others (in the tensor space)
    
    def try_eliminate_pair(Ua, Va, Wa, r, col1, col2):
        """Try to eliminate col2 by expressing its contribution as a combo of col1.
        We need: U[:,col2]*V[:,col2]*W[:,col2] = alpha * U[:,col1]*V[:,col1]*W[:,col1]
        for all (i,j,k) entries, which requires the rank-1 tensors to be proportional."""
        # Check if the 3D tensors are proportional
        # T2[i,j,k] = Ua[i*m+j,col2]*Va[j*p+k,col2]*Wa[k*n+i,col2]
        # T1[i,j,k] = Ua[i*m+j,col1]*Va[j*p+k,col1]*Wa[k*n+i,col1]
        # We need T2 = alpha * T1 for some alpha
        T1 = np.einsum('i,j,k->ijk', 
                       Ua[:,col1].reshape(n,m), 
                       Va[:,col1].reshape(m,p), 
                       Wa[:,col1].reshape(p,n))
        T2 = np.einsum('i,j,k->ijk', 
                       Ua[:,col2].reshape(n,m), 
                       Va[:,col2].reshape(m,p), 
                       Wa[:,col2].reshape(p,n))
        
        # Check proportionality
        nonzero_mask = T1 != 0
        if not np.any(nonzero_mask):
            return None
        alphas = T2[nonzero_mask] / T1[nonzero_mask]
        # Check if all alphas are the same
        alpha = alphas[0]
        if not np.allclose(alphas, alpha, atol=1e-9):
            return None
        # Verify for all entries
        if not np.allclose(T2, alpha * T1, atol=1e-9):
            return None
        # alpha should be integer or half-integer for exact solution
        # For now, just try if alpha is close to an integer
        alpha_rounded = round(alpha)
        if abs(alpha - alpha_rounded) < 1e-9:
            # Subtract alpha_rounded * col1 from col2
            Ua[:,col2] -= alpha_rounded * Ua[:,col1]
            Va[:,col2] -= alpha_rounded * Va[:,col1]
            Wa[:,col2] -= alpha_rounded * Wa[:,col1]
            # Check if col2 is now zero
            if np.all(Ua[:,col2] == 0) and np.all(Va[:,col2] == 0) and np.all(Wa[:,col2] == 0):
                U_new = np.delete(Ua, col2, axis=1)
                V_new = np.delete(Va, col2, axis=1)
                W_new = np.delete(Wa, col2, axis=1)
                return U_new, V_new, W_new, r - 1
        return None
    
    # Try eliminating terms
    improved = True
    while improved and check_time():
        improved = False
        r_current = U_np.shape[1]
        for col1 in range(r_current):
            if not check_time():
                break
            for col2 in range(col1 + 1, r_current):
                if not check_time():
                    break
                U_copy = U_np.copy()
                V_copy = V_np.copy()
                W_copy = W_np.copy()
                result = try_eliminate_pair(U_copy, V_copy, W_copy, r_current, col1, col2)
                if result is not None:
                    U_np, V_np, W_np, r_new = result
                    r_best = r_new
                    improved = True
                    break
            if improved:
                break
    
    # Try more aggressive local search: use the structure of the problem
    # The 4x4x4 part uses Strassen (rank 7), and the 4x4x5 part is schoolbook (rank 20)
    # Total: 7 + 20 = 27, but we have overlapping terms
    # Let's try to find a better decomposition by exploiting the block structure
    
    # Alternative: try to use a 4x4x9 decomposition more directly
    # The tensor is T[i,j,k] = delta[i,j,k] for the multiplication
    # We can try to find rank-r decompositions using integer programming heuristics
    
    # For now, try a different approach: use the fact that the right 5 columns
    # are just identity-like terms, and see if we can merge them with Strassen terms
    
    # Try to find if any column can be eliminated by linear combination
    # using the full tensor constraint
    
    def try_linear_elimination(Ua, Va, Wa, r):
        """Try to find a non-trivial linear combination of columns that gives zero tensor.
        If found, we can use it to eliminate one column."""
        # The tensor contribution of column c is:
        # T_c[i,j,k] = Ua[i*m+j,c] * Va[j*p+k,c] * Wa[k*n+i,c]
        # We want sum_c alpha_c * T_c = 0 for some alpha not all zero
        # This is a system of n*m*p = 144 equations in r unknowns
        # If r > 144, there's always a solution, but we need integer solutions
        
        # For smaller r, try to find if one column is in the span of others
        # by checking if its tensor is a linear combination of others
        for target_col in range(r):
            if not check_time():
                return None
            # Build the tensor for target_col
            T_target = np.einsum('i,j,k->ijk', 
                                Ua[:,target_col].reshape(n,m), 
                                Va[:,target_col].reshape(m,p), 
                                Wa[:,target_col].reshape(p,n))
            T_flat = T_target.flatten()
            
            # Build the matrix of other tensors
            other_cols = [c for c in range(r) if c != target_col]
            if len(other_cols) == 0:
                continue
            
            A = np.zeros((n*m*p, len(other_cols)), dtype=np.int64)
            for idx, oc in enumerate(other_cols):
                T_oc = np.einsum('i,j,k->ijk', 
                                Ua[:,oc].reshape(n,m), 
                                Va[:,oc].reshape(m,p), 
                                Wa[:,oc].reshape(p,n))
                A[:, idx] = T_oc.flatten()
            
            # Solve A * x = T_flat
            # Use least squares with integer rounding
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A.astype(np.float64), T_flat.astype(np.float64), rcond=None)
                # Check if the solution is close to integer
                x_rounded = np.round(x).astype(np.int64)
                if np.allclose(A @ x_rounded, T_flat, atol=1e-9):
                    # We found an integer combination
                    # Subtract this combination from target_col
                    for idx, oc in enumerate(other_cols):
                        if x_rounded[idx] != 0:
                            Ua[:,target_col] -= x_rounded[idx] * Ua[:,oc]
                            Va[:,target_col] -= x_rounded[idx] * Va[:,oc]
                            Wa[:,target_col] -= x_rounded[idx] * Wa[:,oc]
                    # Check if target_col is now zero
                    if np.all(Ua[:,target_col] == 0) and np.all(Va[:,target_col] == 0) and np.all(Wa[:,target_col] == 0):
                        U_new = np.delete(Ua, target_col, axis=1)
                        V_new = np.delete(Va, target_col, axis=1)
                        W_new = np.delete(Wa, target_col, axis=1)
                        return U_new, V_new, W_new, r - 1
            except:
                pass
        return None
    
    # Try linear elimination
    improved = True
    while improved and check_time():
        improved = False
        result = try_linear_elimination(U_np.copy(), V_np.copy(), W_np.copy(), U_np.shape[1])
        if result is not None:
            U_np, V_np, W_np, r_new = result
            r_best = r_new
            improved = True
    
    # Convert back to lists
    U_best = U_np.tolist()
    V_best = V_np.tolist()
    W_best = W_np.tolist()
    
    return {"U": U_best, "V": V_best, "W": W_best}
