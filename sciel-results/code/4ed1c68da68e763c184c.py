def construct():
    n, m, p = 3, 3, 4
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 35: 7 for Strassen 2x2x2 subblock, 28 for rest
    rank = 35

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # Indices for the 2x2x2 subblock (i in {0,1}, j in {0,1}, k in {0,1})
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need to add these to U, V, W for r=0..6
    
    # Helper to add a term
    def add_term(r, a_coeffs, b_coeffs, c_coeffs):
        for idx, coeff in a_coeffs.items():
            U[idx][r] += coeff
        for idx, coeff in b_coeffs.items():
            V[idx][r] += coeff
        for idx, coeff in c_coeffs.items():
            W[idx][r] += coeff

    # Subblock indices
    A00, A01, A10, A11 = 0, 1, 3, 4
    B00, B01, B10, B11 = 0, 1, 4, 5
    C00, C01, C10, C11 = 0, 1, 3, 4

    # r=0: M1
    add_term(0, {A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
    # r=1: M2
    add_term(1, {A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
    # r=2: M3
    add_term(2, {A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
    # r=3: M4
    add_term(3, {A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
    # r=4: M5
    add_term(4, {A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
    # r=5: M6
    add_term(5, {A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
    # r=6: M7
    add_term(6, {A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1, C11: 1})

    # Schoolbook for the rest
    # Total terms 36. We skipped the 8 terms for (i,j,k) in {0,1}x{0,1}x{0,1}.
    # We iterate all 36 and skip those.
    r = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
