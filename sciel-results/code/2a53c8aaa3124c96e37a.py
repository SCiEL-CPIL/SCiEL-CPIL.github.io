import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W

def _strassen_2x2_terms():
    return [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

def _strassen_4x4_terms():
    s2 = _strassen_2x2_terms()
    macro_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    terms = []
    for A_m, B_m, C_m in macro_terms:
        for u2, v2, w2 in s2:
            u_coeffs = {}
            v_coeffs = {}
            w_coeffs = {}
            for (li, lj), u_val in u2.items():
                for (si, sj), a_val in A_m.items():
                    gi = si * 2 + li
                    gj = sj * 2 + lj
                    u_coeffs[(gi, gj)] = u_coeffs.get((gi, gj), 0) + u_val * a_val
            for (lj, lk), v_val in v2.items():
                for (sj, sk), b_val in B_m.items():
                    gj = sj * 2 + lj
                    gk = sk * 2 + lk
                    v_coeffs[(gj, gk)] = v_coeffs.get((gj, gk), 0) + v_val * b_val
            for (li, lk), w_val in w2.items():
                for (si, sk), c_val in C_m.items():
                    gi = si * 2 + li
                    gk = sk * 2 + lk
                    w_coeffs[(gi, gk)] = w_coeffs.get((gi, gk), 0) + w_val * c_val
            u_coeffs = {k: v for k, v in u_coeffs.items() if v != 0}
            v_coeffs = {k: v for k, v in v_coeffs.items() if v != 0}
            w_coeffs = {k: v for k, v in w_coeffs.items() if v != 0}
            terms.append((u_coeffs, v_coeffs, w_coeffs))
    return terms

def _construct_4x1_decomposition():
    n, m, p = 5, 5, 5
    blocks = [[0, 1, 2, 3], [4]]
    s4_terms = _strassen_4x4_terms()
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    def add_schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    def add_strassen_block(i_list, j_list, k_list, terms):
        for u_coeffs, v_coeffs, w_coeffs in terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), val in u_coeffs.items():
                gi = i_list[li]
                gj = j_list[lj]
                u_entries.append((gi * m + gj, val))
            for (lj, lk), val in v_coeffs.items():
                gj = j_list[lj]
                gk = k_list[lk]
                v_entries.append((gj * p + gk, val))
            for (li, lk), val in w_coeffs.items():
                gi = i_list[li]
                gk = k_list[lk]
                w_entries.append((gk * n + gi, val))
            add_column(u_entries, v_entries, w_entries)
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                if len(i_list) == 4 and len(j_list) == 4 and len(k_list) == 4:
                    add_strassen_block(i_list, j_list, k_list, s4_terms)
                else:
                    add_schoolbook_block(i_list, j_list, k_list)
    
    total_rank = len(columns)
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx, c] = val
        for idx, val in v_d.items():
            V[idx, c] = val
        for idx, val in w_d.items():
            W[idx, c] = val
    return U, V, W

def _reduce_rank_by_elimination(U, V, W, n=5, m=None, p=5):
    """Try to reduce rank by finding linear dependencies among columns."""
    if _deadline_exceeded():
        return U, V, W
    
    num_cols = U.shape[1]
    if num_cols <= 1:
        return U, V, W
    
    # Convert to exact fractions for precision
    U_frac = [[Fraction(int(U[i,c]*1000000), 1000000) if U[i,c] != 0 else Fraction(0,1) for c in range(num_cols)] for i in range(25)]
    V_frac = [[Fraction(int(V[i,c]*1000000), 1000000) if V[i,c] != 0 else Fraction(0,1) for c in range(num_cols)] for i in range(25)]
    W_frac = [[Fraction(int(W[i,c]*1000000), 1000000) if W[i,c] != 0 else Fraction(0,1) for c in range(num_cols)] for i in range(25)]
    
    # Check for duplicate or proportional columns
    keep = list(range(num_cols))
    used = set()
    
    for c in range(num_cols):
        if c in used:
            continue
        for d in range(c+1, num_cols):
            if d in used:
                continue
            # Check if column d is proportional to column c
            found = False
            ratio = None
            for i in range(25):
                if U_frac[i][c] != 0:
                    if U_frac[i][d] == 0:
                        found = False
                        break
                    ratio = U_frac[i][d] / U_frac[i][c]
                    found = True
                    break
            if not found:
                continue
            proportional = True
            for i in range(25):
                if U_frac[i][c] != 0 and U_frac[i][d] != U_frac[i][c] * ratio:
                    proportional = False
                    break
                if U_frac[i][c] == 0 and U_frac[i][d] != 0:
                    proportional = False
                    break
            if not proportional:
                continue
            for i in range(25):
                if V_frac[i][c] != 0 and V_frac[i][d] != V_frac[i][c] * ratio:
                    proportional = False
                    break
                if V_frac[i][c] == 0 and V_frac[i][d] != 0:
                    proportional = False
                    break
            if not proportional:
                continue
            for i in range(25):
                if W_frac[i][c] != 0 and W_frac[i][d] != W_frac[i][c] * ratio:
                    proportional = False
                    break
                if W_frac[i][c] == 0 and W_frac[i][d] != 0:
                    proportional = False
                    break
            if proportional:
                used.add(d)
    
    if used:
        new_cols = [c for c in range(num_cols) if c not in used]
        U_new = U[:, new_cols]
        V_new = V[:, new_cols]
        W_new = W[:, new_cols]
        return U_new, V_new, W_new
    
    return U, V, W

def _verify_decomposition_fast(U, V, W, n=5, m=5, p=5):
    if _deadline_exceeded():
        return False
    # Check all valid (i,j,k) triples must give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-8:
                    return False
    # Check all invalid triples must give 0
    # Exhaustive check for 5x5x5: 25*25*25 = 15625 total, 125 valid
    # Sample a large number of invalid triples
    rng = np.random.RandomState(42)
    samples = 8000
    for _ in range(samples):
        a = rng.randint(0, 25)
        b = rng.randint(0, 25)
        c = rng.randint(0, 25)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if j == j2 and k == k2 and i == i2:
            continue
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > 1e-8:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 150.0
    n, m, p = DIMENSIONS
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)
    try:
        if time.time() - start_time < deadline - 80:
            U_block, V_block, W_block = _construct_4x1_decomposition()
            if time.time() - start_time < deadline - 40:
                U_red, V_red, W_red = _reduce_rank_by_elimination(U_block, V_block, W_block, n, m, p)
                if time.time() - start_time < deadline - 25:
                    valid = _verify_decomposition_fast(U_red, V_red, W_red, n, m, p)
                    if valid:
                        return {
                            "U": U_red.tolist(),
                            "V": V_red.tolist(),
                            "W": W_red.tolist()
                        }
            if time.time() - start_time < deadline - 20:
                valid = _verify_decomposition_fast(U_block, V_block, W_block, n, m, p)
                if valid:
                    return {
                        "U": U_block.tolist(),
                        "V": V_block.tolist(),
                        "W": W_block.tolist()
                    }
    except Exception:
        pass
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
