def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen components definitions
    # M1..M7 coefficients for A, B, C
    # A: a,b,c,d. B: e,f,g,h. C: c00,c01,c10,c11
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+c)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # Coefficients are (U_idx, coeff), (V_idx, coeff), (W_idx, coeff)
    # We iterate s in 0..3, t in 0..6
    
    # Map s to A block and B block and C block
    # s=0: A0, B00, C0
    # s=1: A0, B01, C1
    # s=2: A1, B10, C0
    # s=3: A1, B11, C1
    
    def get_A_indices(s):
        if s == 0 or s == 1:
            # A0: 2x2 block at [0:2, 0:2]
            # a: 0,0 -> 0; b: 0,1 -> 1; c: 1,0 -> 2; d: 1,1 -> 3
            return {
                'a': 0, 'b': 1, 'c': 2, 'd': 3
            }
        else:
            # A1: 2x2 block at [0:2, 2:4]
            # a: 0,2 -> 2; b: 0,3 -> 3; c: 1,2 -> 4; d: 1,3 -> 5
            return {
                'a': 2, 'b': 3, 'c': 4, 'd': 5
            }

    def get_B_indices(s):
        if s == 0:
            # B00: [0:2, 0:2]
            # e: 0,0->0; f: 0,1->1; g: 1,0->2; h: 1,1->3
            return {'e': 0, 'f': 1, 'g': 2, 'h': 3}
        elif s == 1:
            # B01: [0:2, 2:4]
            # e: 0,2->2; f: 0,3->3; g: 1,2->4; h: 1,3->5
            return {'e': 2, 'f': 3, 'g': 4, 'h': 5}
        elif s == 2:
            # B10: [2:4, 0:2]
            # e: 2,0->8; f: 2,1->9; g: 3,0->10; h: 3,1->11
            return {'e': 8, 'f': 9, 'g': 10, 'h': 11}
        else:
            # B11: [2:4, 2:4]
            # e: 2,2->10; f: 2,3->11; g: 3,2->12; h: 3,3->13
            return {'e': 10, 'f': 11, 'g': 12, 'h': 13}

    def get_C_indices(s):
        # C0: [0:2, 0:2] -> W indices k=0,1; i=0,1
        # c00: k=0,i=0 -> 0
        # c01: k=1,i=0 -> 2
        # c10: k=0,i=1 -> 1
        # c11: k=1,i=1 -> 3
        if s == 0 or s == 2:
            return {'c00': 0, 'c01': 2, 'c10': 1, 'c11': 3}
        else:
            # C1: [0:2, 2:4] -> W indices k=2,3; i=0,1
            # c00: k=2,i=0 -> 4
            # c01: k=3,i=0 -> 6
            # c10: k=2,i=1 -> 5
            # c11: k=3,i=1 -> 7
            return {'c00': 4, 'c01': 6, 'c10': 5, 'c11': 7}

    for s in range(4):
        a_idx = get_A_indices(s)
        b_idx = get_B_indices(s)
        c_idx = get_C_indices(s)
        
        for t in range(7):
            r = s * 7 + t
            if t == 0:
                # M1 = (a+d)(e+h)
                U[a_idx['a']][r] += 1
                U[a_idx['d']][r] += 1
                V[b_idx['e']][r] += 1
                V[b_idx['h']][r] += 1
                W[c_idx['c00']][r] += 1
                W[c_idx['c11']][r] += 1
            elif t == 1:
                # M2 = (c+d)e
                U[a_idx['c']][r] += 1
                U[a_idx['d']][r] += 1
                V[b_idx['e']][r] += 1
                W[c_idx['c10']][r] += 1
                W[c_idx['c11']][r] -= 1
            elif t == 2:
                # M3 = a(f-h)
                U[a_idx['a']][r] += 1
                V[b_idx['f']][r] += 1
                V[b_idx['h']][r] -= 1
                W[c_idx['c01']][r] += 1
                W[c_idx['c11']][r] += 1
            elif t == 3:
                # M4 = d(g-e)
                U[a_idx['d']][r] += 1
                V[b_idx['g']][r] += 1
                V[b_idx['e']][r] -= 1
                W[c_idx['c00']][r] += 1
                W[c_idx['c10']][r] += 1
            elif t == 4:
                # M5 = (a+c)h
                U[a_idx['a']][r] += 1
                U[a_idx['c']][r] += 1
                V[b_idx['h']][r] += 1
                W[c_idx['c00']][r] -= 1
                W[c_idx['c01']][r] += 1
            elif t == 5:
                # M6 = (c-a)(e+f)
                U[a_idx['c']][r] += 1
                U[a_idx['a']][r] -= 1
                V[b_idx['e']][r] += 1
                V[b_idx['f']][r] += 1
                W[c_idx['c11']][r] += 1
            elif t == 6:
                # M7 = (b-d)(g+h)
                U[a_idx['b']][r] += 1
                U[a_idx['d']][r] -= 1
                V[b_idx['g']][r] += 1
                V[b_idx['h']][r] += 1
                W[c_idx['c00']][r] += 1

    return {"U": U, "V": V, "W": W}
