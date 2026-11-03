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
    
    # Try eliminating terms
    improved = True
    while improved and check_time():
        improved = False
        for col in range(U_np.shape[1]):
            if not check_time():
                break
            result = try_eliminate(U_np, V_np, W_np, U_np.shape[1], col)
            if result is not None:
                U_np, V_np, W_np, r_new = result
                r_best = r_new
                improved = True
                break
    
    # Now try more aggressive local search: perturb entries
    # Try to find lower-rank solutions by random search with the structure
    random.seed(42)
    
    # Try random search for lower ranks
    if check_time():
        for r_try in range(r_best - 1, max(36, r_best - 30), -1):
            if not check_time():
                break
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts and check_time():
                attempts += 1
                # Generate sparse random matrices
                U_try = np.zeros((nm, r_try), dtype=np.int64)
                V_try = np.zeros((mp, r_try), dtype=np.int64)
                W_try = np.zeros((pn, r_try), dtype=np.int64)
                
                for i in range(nm):
                    for c in range(r_try):
                        if random.random() < 0.6:
                            U_try[i, c] = random.choice([-1, 1])
                for i in range(mp):
                    for c in range(r_try):
                        if random.random() < 0.6:
                            V_try[i, c] = random.choice([-1, 1])
                for i in range(pn):
                    for c in range(r_try):
                        if random.random() < 0.6:
                            W_try[i, c] = random.choice([-1, 1])
                
                # Validate
                T = np.zeros((n, m, p), dtype=np.int64)
                for c in range(r_try):
                    ur = U_try[:, c].reshape(n, m)
                    vr = V_try[:, c].reshape(m, p)
                    wr = W_try[:, c].reshape(p, n)
                    T += np.einsum('ij,jk,ki->ijk', ur, vr, wr)
                
                if np.all(T == 1):
                    r_best = r_try
                    U_np = U_try
                    V_np = V_try
                    W_np = W_try
                    break
    
    # Convert back to lists
    U_best = U_np.tolist()
    V_best = V_np.tolist()
    W_best = W_np.tolist()
    
    return {"U": U_best, "V": V_best, "W": W_best}
