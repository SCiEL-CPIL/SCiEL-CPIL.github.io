import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]
DEADLINE = 135.0

# Strassen 2x2x2: 7 terms
# A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
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

# Each term: (a_vec, b_vec) where vectors are in [a00,a01,a10,a11] / [b00,b01,b10,b11]
S_TERMS = [
    ([1,0,0,1], [1,0,0,1]),   # M0
    ([0,0,0,1], [-1,0,1,0]),  # M1
    ([0,0,1,1], [0,1,0,0]),   # M2
    ([1,0,0,0], [0,1,0,-1]),  # M3
    ([1,1,0,0], [1,0,0,1]),   # M4
    ([0,1,0,-1], [0,0,1,1]),  # M5
    ([-1,0,1,0], [1,1,0,0]),  # M6
]

# C block contributions: C[cbi] = sum_s S_C[cbi][s] * M_s
# cbi: 0=C00, 1=C01, 2=C10, 3=C11
S_C = [
    [1, 0, 0, 1, -1, 0, 1],   # C00
    [0, 0, 1, 0, 1, 0, 0],    # C01
    [0, 1, 0, 1, 0, 0, 0],    # C10
    [1, -1, 1, 0, 0, -1, 0],  # C11
]


def strassen_2x2x2_scalar_terms():
    """Generate scalar terms for 2x2*2x2->2x2 Strassen.
    Returns list of (a_idx, b_idx, c_idx, coeff) where indices are 0..3 row-major."""
    terms = {}
    for s in range(7):
        a_vec, b_vec = S_TERMS[s]
        for cbi in range(4):
            sign = S_C[cbi][s]
            if sign == 0:
                continue
            ci, cj = divmod(cbi, 2)
            c_idx = ci * 2 + cj
            for ai in range(4):
                if a_vec[ai] == 0:
                    continue
                for bi in range(4):
                    if b_vec[bi] == 0:
                        continue
                    coeff = sign * a_vec[ai] * b_vec[bi]
                    key = (ai, bi, c_idx)
                    terms[key] = terms.get(key, 0) + coeff
    return [(k[0], k[1], k[2], v) for k, v in terms.items() if v != 0]


def strassen_4x4x4_scalar_terms():
    """Generate scalar terms for 4x4*4x4->4x4 using Strassen-of-Strassen.
    Returns list of (a_idx, b_idx, c_idx, coeff) where indices are 0..15 row-major (4x4)."""
    # Outer Strassen on 2x2 blocks of 2x2 matrices
    # A 4x4 = [[A00,A01],[A10,A11]], each 2x2
    # Block index bi: 0=A00(rows0-1,cols0-1), 1=A01(rows0-1,cols2-3), 2=A10(rows2-3,cols0-1), 3=A11(rows2-3,cols2-3)
    
    terms = {}
    for outer_s in range(7):
        a_vec, b_vec = S_TERMS[outer_s]
        for cbi in range(4):
            outer_sign = S_C[cbi][outer_s]
            if outer_sign == 0:
                continue
            ci_block, cj_block = divmod(cbi, 2)
            # Inner Strassen for the 2x2*2x2 product
            for inner_s in range(7):
                ia_vec, ib_vec = S_TERMS[inner_s]
                for inner_cbi in range(4):
                    inner_sign = S_C[inner_cbi][inner_s]
                    if inner_sign == 0:
                        continue
                    inner_ci, inner_cj = divmod(inner_cbi, 2)
                    # Global output position
                    c_row = ci_block * 2 + inner_ci
                    c_col = cj_block * 2 + inner_cj
                    c_idx = c_row * 4 + c_col
                    
                    # For each element in the inner 2x2 A and B matrices
                    for li in range(2):
                        for lj in range(2):
                            ia_c = ia_vec[li*2+lj]
                            if ia_c == 0:
                                continue
                            for ri in range(2):
                                for rj in range(2):
                                    ib_c = ib_vec[ri*2+rj]
                                    if ib_c == 0:
                                        continue
                                    # L[li][lj] = sum over outer A blocks: a_vec[ab] * A[2*abr+li][2*abc+lj]
                                    # R[ri][rj] = sum over outer B blocks: b_vec[bb] * B[2*bbr+ri][2*bbc+rj]
                                    for ab in range(4):
                                        ac = a_vec[ab]
                                        if ac == 0:
                                            continue
                                        abr, abc = divmod(ab, 2)
                                        a_row = 2*abr + li
                                        a_col = 2*abc + lj
                                        a_idx = a_row * 4 + a_col
                                        
                                        for bb in range(4):
                                            bc = b_vec[bb]
                                            if bc == 0:
                                                continue
                                            bbr, bbc = divmod(bb, 2)
                                            b_row = 2*bbr + ri
                                            b_col = 2*bbc + rj
                                            b_idx = b_row * 4 + b_col
                                            
                                            coeff = outer_sign * inner_sign * ac * ia_c * bc * ib_c
                                            key = (a_idx, b_idx, c_idx)
                                            terms[key] = terms.get(key, 0) + coeff
    return [(k[0], k[1], k[2], v) for k, v in terms.items() if v != 0]


def validate_decomp(U, V, W, n, m, p, r):
    """Validate that sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(r):
                    u_val = U[i*m+j][c]
                    v_val = V[j*p+k][c]
                    w_val = W[k*n+i][c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    def elapsed():
        return time.time() - t0
    
    # Strategy: Use Strassen-of-Strassen for 4x4*4x4 (first 4 cols of B),
    # and schoolbook for 4x4*4x5 (remaining 5 cols of B).
    # Total rank: 49 + 80 = 129 (if Strassen works) vs 144 schoolbook.
    
    # Generate 4x4x4 Strassen terms
    s4_terms = strassen_4x4x4_scalar_terms()
    
    # Verify Strassen 4x4x4 is correct
    def verify_4x4x4(terms):
        """Check that sum of terms reconstructs 4x4*4x4->4x4 identity tensor."""
        # Build a tensor T[i,j,k] and check it equals 1 for all valid (i,j,k)
        # In 4x4*4x4->4x4: T[i,j,k] = 1 if j is the "middle" index
        # Actually the tensor is: for each (i,j,k), coefficient is 1
        # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should = 1 for all i,j,k in 0..3
        tensor = {}
        for (a, b, c, coeff) in terms:
            i = a // 4
            j = a % 4
            j2 = b // 4
            k = b % 4
            k2 = c // 4
            i2 = c % 4
            if j != j2 or i != i2:
                return False
            key = (i, j, k)
            tensor[key] = tensor.get(key, 0) + coeff
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if tensor.get((i, j, k), 0) != 1:
                        return False
        return True
    
    if not verify_4x4x4(s4_terms):
        # Strassen is buggy, use schoolbook
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Remap Strassen terms for B 4x9 (first 4 columns)
    terms_left = []
    for (a, b, c, coeff) in s4_terms:
        # b = j*4 + k_old (k_old in 0..3) -> b_new = j*9 + k_old
        j = b // 4
        k_old = b % 4
        b_new = j * 9 + k_old
        # c = k_old*4 + i -> stays the same since W is indexed by (k,i) with k in 0..8
        terms_left.append((a, b_new, c, coeff))
    
    # Schoolbook for remaining 5 columns (k=4..8)
    terms_right = []
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                a_global = i * 4 + j
                b_global = j * 9 + k
                c_global = k * 4 + i
                terms_right.append((a_global, b_global, c_global, 1))
    
    all_terms = terms_left + terms_right
    r = len(all_terms)
    
    # Build U, V, W
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c_idx, (a, b, cg, coeff) in enumerate(all_terms):
        i = a // m
        j = a % m
        U[i*m + j][c_idx] += coeff
        j2 = b // p
        k = b % p
        V[j*p + k][c_idx] += 1
        k2 = cg // n
        i2 = cg % n
        W[k*n + i][c_idx] += 1
    
    # Final validation
    if not validate_decomp(U, V, W, n, m, p, r):
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
