import time
import random
from fractions import Fraction as F

DIMENSIONS = [4, 4, 9]
DEADLINE = 140.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def check_time():
        return time.time() - t0 < DEADLINE
    
    def validate_int(U, V, W, r):
        """Validate using integer arithmetic. U, V, W are lists of lists of ints."""
        for i in range(n):
            for j in range(m):
                Uj = U[i*m + j]
                for k in range(p):
                    Vk = V[j*p + k]
                    Wk = W[k*n + i]
                    tot = 0
                    for c in range(r):
                        tot += Uj[c] * Vk[c] * Wk[c]
                    if tot != 1:
                        return False
        return True
    
    # Strassen 2x2x2 terms
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
    
    if not validate_int(U_best, V_best, W_best, r_best):
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
    
    # STRUCTURAL CHANGE: Iterative column elimination via tensor linear dependence
    # For each column c, check if the rank-1 tensor T_c = u_c ⊗ v_c ⊗ w_c
    # can be expressed as a linear combination of the other columns.
    # We do this by checking if T_c lies in the span of {T_d : d != c}
    # in the 144-dim space of tensors.
    
    def tensor_flatten(U, V, W, r, c):
        """Flatten column c into a 144-dim vector (one entry per (i,j,k))."""
        vec = [0] * (n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*m*p + j*p + k] = U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        return vec
    
    def try_eliminate_column(U, V, W, r, col):
        """Try to eliminate column 'col' by expressing it as combination of others.
        Returns (new_U, new_V, new_W, new_r) or None."""
        # Build the system: find coefficients alpha_d such that
        # T_col = sum_{d != col} alpha_d * T_d
        # This is a linear system in the 144-dim tensor space.
        # But we need the decomposition to remain valid, which is more constrained.
        # 
        # Alternative: directly try removing column 'col' and check if we can
        # adjust the remaining columns to compensate. This is harder.
        # 
        # Simpler approach: just remove and validate.
        r_new = r - 1
        U_new = [row[:] for row in U]
        V_new = [row[:] for row in V]
        W_new = [row[:] for row in W]
        
        # Remove column 'col'
        for i in range(nm):
            U_new[i] = U_new[i][:col] + U_new[i][col+1:]
        for i in range(mp):
            V_new[i] = V_new[i][:col] + V_new[i][col+1:]
        for i in range(pn):
            W_new[i] = W_new[i][:col] + W_new[i][col+1:]
        
        if validate_int(U_new, V_new, W_new, r_new):
            return U_new, V_new, W_new, r_new
        return None
    
    # Iteratively try to eliminate columns
    improved = True
    while improved and check_time():
        improved = False
        for col in range(r_best):
            if not check_time():
                break
            result = try_eliminate_column(U_best, V_best, W_best, r_best, col)
            if result is not None:
                U_best, V_best, W_best, r_best = result
                improved = True
                break  # Restart from beginning after a successful elimination
    
    # Try removing pairs of columns
    if check_time():
        for c1 in range(r_best):
            if not check_time():
                break
            for c2 in range(c1 + 1, r_best):
                if not check_time():
                    break
                U_test = [row[:] for row in U_best]
                V_test = [row[:] for row in V_best]
                W_test = [row[:] for row in W_best]
                
                # Remove columns c1 and c2 (c1 first, then adjust c2 index)
                U_test = [row[:c1] + row[c1+1:c2] + row[c2+1:] for row in U_test]
                V_test = [row[:c1] + row[c1+1:c2] + row[c2+1:] for row in V_test]
                W_test = [row[:c1] + row[c1+1:c2] + row[c2+1:] for row in W_test]
                
                if validate_int(U_test, V_test, W_test, r_best - 2):
                    U_best = U_test
                    V_best = V_test
                    W_best = W_test
                    r_best -= 2
                    break
    
    # Try removing triples
    if check_time() and r_best > 3:
        for c1 in range(r_best):
            if not check_time():
                break
            for c2 in range(c1 + 1, r_best):
                if not check_time():
                    break
                for c3 in range(c2 + 1, r_best):
                    if not check_time():
                        break
                    U_test = [row[:] for row in U_best]
                    V_test = [row[:] for row in V_best]
                    W_test = [row[:] for row in W_best]
                    
                    U_test = [row[:c1] + row[c1+1:c2] + row[c2+1:c3] + row[c3+1:] for row in U_test]
                    V_test = [row[:c1] + row[c1+1:c2] + row[c2+1:c3] + row[c3+1:] for row in V_test]
                    W_test = [row[:c1] + row[c1+1:c2] + row[c2+1:c3] + row[c3+1:] for row in W_test]
                    
                    if validate_int(U_test, V_test, W_test, r_best - 3):
                        U_best = U_test
                        V_best = V_test
                        W_best = W_test
                        r_best -= 3
                        break
    
    return {"U": U_best, "V": V_best, "W": W_best}
