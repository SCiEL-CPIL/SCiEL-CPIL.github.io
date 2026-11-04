import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]
DEADLINE = 125.0  # safety margin before 150s kill

# Strassen 2x2x2 decomposition
# For A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
# M0 = (a00+a11)(b00+b11)
# M1 = a11(b10-b00)
# M2 = (a10+a11)b01
# M3 = a00(b01-b11)
# M4 = (a00+a01)(b11+b00)
# M5 = (a01-a11)(b10+b11)
# M6 = (a10-a00)(b00+b01)
# C00 = M0 + M3 - M4 + M6
# C01 = M2 + M4
# C10 = M1 + M3
# C11 = M0 - M1 + M2 - M5

STRASSEN_TERMS = [
    {'a': [1, 0, 0, 1], 'b': [1, 0, 0, 1]},  # M0
    {'a': [0, 0, 0, 1], 'b': [-1, 0, 1, 0]},  # M1
    {'a': [0, 0, 1, 1], 'b': [0, 1, 0, 0]},   # M2
    {'a': [1, 0, 0, 0], 'b': [0, 1, 0, -1]},  # M3
    {'a': [1, 1, 0, 0], 'b': [1, 0, 0, 1]},   # M4
    {'a': [0, 1, 0, -1], 'b': [0, 0, 1, 1]},  # M5
    {'a': [-1, 0, 1, 0], 'b': [1, 1, 0, 0]},  # M6
]

C_BLOCKS = [
    [1, 1, -1, 0, 0, 0, 1],    # C00: M0, M3, -M4, M6
    [0, 0, 1, 0, 1, 0, 0],     # C01: M2, M4
    [0, 1, 0, 1, 0, 0, 0],     # C10: M1, M3
    [1, -1, 1, 0, 0, -1, 0],   # C11: M0, -M1, M2, -M5
]


def generate_4x4x4_strassen_terms():
    """Generate terms for 4x4 * 4x4 -> 4x4 using Strassen-of-Strassen.
    Returns dict mapping (a_global, b_global, c_global) -> coeff
    where a_global = i*4+j, b_global = j*4+k, c_global = k*4+i
    """
    term_dict = {}
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
                                                key = (a_global, b_global, c_global)
                                                term_dict[key] = term_dict.get(key, 0) + coeff
    return term_dict


def validate_decomposition(U, V, W, n, m, p, r):
    """Validate that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(r):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    def check_time():
        return time.time() - t0 < DEADLINE
    
    # Generate Strassen terms for the 4x4 * 4x4 -> 4x4 part
    terms_4x4 = generate_4x4x4_strassen_terms()
    
    # Build the full decomposition for 4x4 * 4x9 -> 4x9
    # Left part: columns 0-3 of B use Strassen
    # Right part: columns 4-8 of B use schoolbook
    
    all_terms = []
    
    # Left part: columns 0-3 of B (use Strassen)
    for (a, b, c), coeff in terms_4x4.items():
        # a = i*4 + j
        # b = j*4 + k_old (k_old in 0..3)
        # c = k_old*4 + i
        j = b // 4
        k_old = b % 4
        b_new = j * 9 + k_old
        all_terms.append((a, b_new, c, coeff))
    
    # Right part: columns 4-8 of B (schoolbook)
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                a_global = i * 4 + j
                b_global = j * 9 + k
                c_global = k * 4 + i
                all_terms.append((a_global, b_global, c_global, 1))
    
    r = len(all_terms)
    
    # Build U, V, W
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c_idx, (a, b, cg, coeff) in enumerate(all_terms):
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = cg // n
        i2 = cg % n
        
        U[i*m + j][c_idx] += coeff
        V[j*p + k][c_idx] += 1
        W[k*n + i][c_idx] += 1
    
    # Validate
    if not validate_decomposition(U, V, W, n, m, p, r):
        # Fallback to schoolbook
        r_fb = n * m * p
        U = [[0]*r_fb for _ in range(n*m)]
        V = [[0]*r_fb for _ in range(m*p)]
        W = [[0]*r_fb for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
