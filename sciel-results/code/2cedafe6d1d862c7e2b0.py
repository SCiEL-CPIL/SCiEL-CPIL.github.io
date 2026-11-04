DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coords, v_coords, w_coords):
        nonlocal col
        if col >= rank:
            return
        for (i, j), coeff in u_coords:
            U[i * m + j][col] += coeff
        for (j, k), coeff in v_coords:
            V[j * p + k][col] += coeff
        for (k, i), coeff in w_coords:
            W[k * n + i][col] += coeff
        col += 1

    # Strassen 2x2x2 terms
    # Local indices: i,j in 0..1
    strassen = [
        # M1
        {
            'u': [((0,0),1), ((1,1),1)],
            'v': [((0,0),1), ((1,1),1)],
            'w': [((0,0),1), ((1,1),1), ((0,1),-1), ((1,0),-1)]
        },
        # M2
        {
            'u': [((1,0),1), ((1,1),1)],
            'v': [((0,0),1)],
            'w': [((1,0),1), ((1,1),-1)]
        },
        # M3
        {
            'u': [((0,0),1)],
            'v': [((1,0),1), ((1,1),-1)],
            'w': [((0,1),1), ((1,1),1)]
        },
        # M4
        {
            'u': [((1,1),1)],
            'v': [((0,1),1), ((0,0),-1)],
            'w': [((0,0),1), ((1,0),1)]
        },
        # M5
        {
            'u': [((0,0),1), ((0,1),1)],
            'v': [((1,1),1)],
            'w': [((0,0),-1), ((0,1),1)]
        },
        # M6
        {
            'u': [((1,0),1), ((0,0),-1)],
            'v': [((0,0),1), ((1,0),1)],
            'w': [((1,1),1)]
        },
        # M7
        {
            'u': [((0,1),1), ((1,1),-1)],
            'v': [((0,1),1), ((1,1),1)],
            'w': [((0,0),1)]
        }
    ]
    
    def add_strassen(a_i_base, a_j_base, b_j_base, b_k_base, c_k_base, c_i_base):
        nonlocal col
        for t in strassen:
            u_coords = [((i + a_i_base, j + a_j_base), c) for (i, j), c in t['u']]
            v_coords = [((j + b_j_base, k + b_k_base), c) for (j, k), c in t['v']]
            w_coords = [((k + c_k_base, i + c_i_base), c) for (k, i), c in t['w']]
            add_term(u_coords, v_coords, w_coords)

    # Schoolbook 2x2x3 terms
    # A: 2x2, B: 2x3, C: 3x2 (k in 0..2, i in 0..1)
    # Rank 12
    def get_schoolbook_2x2x3():
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    terms.append({
                        'u': [((i, j), 1)],
                        'v': [((j, k), 1)],
                        'w': [((k, i), 1)]
                    })
        return terms

    schoolbook_2x2x3 = get_schoolbook_2x2x3()

    def add_schoolbook_2x2x3(a_i_base, a_j_base, b_j_base, b_k_base, c_k_base, c_i_base):
        nonlocal col
        for t in schoolbook_2x2x3:
            u_coords = [((i + a_i_base, j + a_j_base), c) for (i, j), c in t['u']]
            v_coords = [((j + b_j_base, k + b_k_base), c) for (j, k), c in t['v']]
            w_coords = [((k + c_k_base, i + c_i_base), c) for (k, i), c in t['w']]
            add_term(u_coords, v_coords, w_coords)

    # 1. 2x4x4 part
    # A1 (j=0,1), B11 (j=0,1, k=0,1), C11 (k=0,1, i=0,1)
    add_strassen(0, 0, 0, 0, 0, 0)
    # A2 (j=2,3), B21 (j=2,3, k=0,1), C21 (k=0,1, i=0,1)
    add_strassen(0, 2, 2, 0, 0, 0)
    # A1 (j=0,1), B12 (j=0,1, k=2,3), C12 (k=2,3, i=0,1)
    add_strassen(0, 0, 0, 2, 2, 0)
    # A2 (j=2,3), B22 (j=2,3, k=2,3), C22 (k=2,3, i=0,1)
    add_strassen(0, 2, 2, 2, 2, 0)
    
    # 2. 2x4x3 part
    # A1 (j=0,1), B13 (j=0,1, k=4,5,6), C13 (k=4,5,6, i=0,1)
    add_schoolbook_2x2x3(0, 0, 0, 4, 4, 0)
    # A2 (j=2,3), B23 (j=2,3, k=4,5,6), C23 (k=4,5,6, i=0,1)
    add_schoolbook_2x2x3(0, 2, 2, 4, 4, 0)
    
    return {"U": U, "V": V, "W": W}
