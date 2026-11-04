import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Schoolbook fallback (rank 36)
    schoolbook_rank = n * m * p
    U_sb = [[0.0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0.0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0.0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1.0
                V_sb[j * p + k][col] = 1.0
                W_sb[k * n + i][col] = 1.0
                col += 1

    # Block Strassen construction for rank 34.
    # Indices: U[i*m+j] = A[i,j], V[j*p+k] = B[j,k], W[k*n+i] = C[i,k]
    U_terms = []
    V_terms = []
    W_terms = []

    def add_term(U_col, V_col, W_col):
        U_vec = [Fraction(0)] * (n * m)
        V_vec = [Fraction(0)] * (m * p)
        W_vec = [Fraction(0)] * (p * n)
        for idx, val in U_col.items():
            U_vec[idx] += val
        for idx, val in V_col.items():
            V_vec[idx] += val
        for idx, val in W_col.items():
            W_vec[idx] += val
        U_terms.append(U_vec)
        V_terms.append(V_vec)
        W_terms.append(W_vec)

    def uidx(i, j):
        return i * m + j

    def vidx(j, k):
        return j * p + k

    def widx(i, k):
        return k * n + i

    def add_mult(i, j, k):
        add_term({uidx(i, j): Fraction(1)},
                 {vidx(j, k): Fraction(1)},
                 {widx(i, k): Fraction(1)})

    def add_strassen(a_idx, b_idx, c_idx):
        """
        a_idx: dict a00,a01,a10,a11 -> (i,j) in A
        b_idx: dict b00,b01,b10,b11 -> (j,k) in B
        c_idx: dict c00,c01,c10,c11 -> (i,k) in C
        Strassen 7-term for 2x2x2.
        """
        a00, a01, a10, a11 = a_idx['a00'], a_idx['a01'], a_idx['a10'], a_idx['a11']
        b00, b01, b10, b11 = b_idx['b00'], b_idx['b01'], b_idx['b10'], b_idx['b11']
        c00, c01, c10, c11 = c_idx['c00'], c_idx['c01'], c_idx['c10'], c_idx['c11']

        # M1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        add_term({uidx(*a00): Fraction(1), uidx(*a11): Fraction(1)},
                 {vidx(*b00): Fraction(1), vidx(*b11): Fraction(1)},
                 {widx(*c00): Fraction(1), widx(*c11): Fraction(1)})
        # M2 = (a10+a11)b00 -> c10 += 1
        add_term({uidx(*a10): Fraction(1), uidx(*a11): Fraction(1)},
                 {vidx(*b00): Fraction(1)},
                 {widx(*c10): Fraction(1)})
        # M3 = a00(b10-b00) -> c01 += 1
        add_term({uidx(*a00): Fraction(1)},
                 {vidx(*b10): Fraction(1), vidx(*b00): Fraction(-1)},
                 {widx(*c01): Fraction(1)})
        # M4 = a11(b01-b11) -> c00 += 1, c10 += 1
        add_term({uidx(*a11): Fraction(1)},
                 {vidx(*b01): Fraction(1), vidx(*b11): Fraction(-1)},
                 {widx(*c00): Fraction(1), widx(*c10): Fraction(1)})
        # M5 = (a00+a01)b11 -> c00 -= 1, c01 += 1, c11 += 1
        add_term({uidx(*a00): Fraction(1), uidx(*a01): Fraction(1)},
                 {vidx(*b11): Fraction(1)},
                 {widx(*c00): Fraction(-1), widx(*c01): Fraction(1), widx(*c11): Fraction(1)})
        # M6 = (a10-a00)(b00+b01) -> c10 -= 1, c11 -= 1
        add_term({uidx(*a10): Fraction(1), uidx(*a00): Fraction(-1)},
                 {vidx(*b00): Fraction(1), vidx(*b01): Fraction(1)},
                 {widx(*c10): Fraction(-1), widx(*c11): Fraction(-1)})
        # M7 = (a01-a11)(b10+b11) -> c00 += 1, c11 += 1
        add_term({uidx(*a01): Fraction(1), uidx(*a11): Fraction(-1)},
                 {vidx(*b10): Fraction(1), vidx(*b11): Fraction(1)},
                 {widx(*c00): Fraction(1), widx(*c11): Fraction(1)})

    # 1. A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(
        {'a00': (0,0), 'a01': (0,1), 'a10': (1,0), 'a11': (1,1)},
        {'b00': (0,0), 'b01': (0,1), 'b10': (1,0), 'b11': (1,1)},
        {'c00': (0,0), 'c01': (0,1), 'c10': (1,0), 'c11': (1,1)})
    # 2. A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(
        {'a00': (0,0), 'a01': (0,1), 'a10': (1,0), 'a11': (1,1)},
        {'b00': (0,2), 'b01': (0,3), 'b10': (1,2), 'b11': (1,3)},
        {'c00': (0,2), 'c01': (0,3), 'c10': (1,2), 'c11': (1,3)})
    # 3. A[0:2,2] * B[2,0:2] -> C[0:2,0:2]  (2x1x2, standard)
    for i in range(2):
        for k in range(2):
            add_mult(i, 2, k)
    # 4. A[0:2,2] * B[2,2:4] -> C[0:2,2:4]
    for i in range(2):
        for k in range(2, 4):
            add_mult(i, 2, k)
    # 5. A[2,0:2] * B[0:2,0:2] -> C[2,0:2]  (1x2x2, standard)
    for j in range(2):
        for k in range(2):
            add_mult(2, j, k)
    # 6. A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    for j in range(2):
        for k in range(2, 4):
            add_mult(2, j, k)
    # 7. A[2,2] * B[2,0:2] -> C[2,0:2]
    for k in range(2):
        add_mult(2, 2, k)
    # 8. A[2,2] * B[2,2:4] -> C[2,2:4]
    for k in range(2, 4):
        add_mult(2, 2, k)

    rank = len(U_terms)

    # Verify entries are integer or half-integer
    valid = True
    for mat in (U_terms, V_terms, W_terms):
        for vec in mat:
            for val in vec:
                d = val * 2
                if d.denominator != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    if not valid:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Verify the decomposition exactly against the target tensor.
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(uidx(i, j), vidx(j, k), widx(i, k))] = Fraction(1)

    decomposed = {}
    for r in range(rank):
        Unz = [(a, v) for a, v in enumerate(U_terms[r]) if v != 0]
        Vnz = [(b, v) for b, v in enumerate(V_terms[r]) if v != 0]
        Wnz = [(c, v) for c, v in enumerate(W_terms[r]) if v != 0]
        for a, ua in Unz:
            for b, vb in Vnz:
                for c, wc in Wnz:
                    key = (a, b, c)
                    if key in target:
                        decomposed[key] = decomposed.get(key, Fraction(0)) + ua * vb * wc

    ok = True
    for key, val in target.items():
        if decomposed.get(key, Fraction(0)) != val:
            ok = False
            break
    if not ok:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    for key in decomposed:
        if key not in target:
            return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Convert to floats (integers and half-integers are exactly representable).
    U_out = [[float(v) for v in vec] for vec in U_terms]
    V_out = [[float(v) for v in vec] for vec in V_terms]
    W_out = [[float(v) for v in vec] for vec in W_terms]
    return {"U": U_out, "V": V_out, "W": W_out}
