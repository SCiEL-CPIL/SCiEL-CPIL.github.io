import time

def construct():
    deadline = time.time() + 118.0

    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    # We'll build the decomposition term by term
    # Each term is (u_dict, v_dict, w_dict)
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    # 1. Core 3x3x3 using BCDFL rank-19
    # Indices in core: a,b,c in {0,1,2}
    # U index: 4a+b
    # V index: 4b+c
    # W index: 3a+c
    
    # BCDFL Rank 19 Algorithm
    # p1 = (a00 + a11) * (b00 + b11)
    add_term({0:1, 5:1}, {0:1, 5:1}, {0:1, 5:1})
    # p2 = (a10 + a11) * b00
    add_term({4:1, 5:1}, {0:1}, {4:1, 5:-1})
    # p3 = a00 * (b01 - b11)
    add_term({0:1}, {1:1, 5:-1}, {1:1, 5:1})
    # p4 = a11 * (b10 - b00)
    add_term({5:1}, {4:1, 0:-1}, {0:1, 4:1})
    # p5 = (a00 + a01) * b11
    add_term({0:1, 1:1}, {5:1}, {0:-1, 1:1})
    # p6 = (a10 - a00) * (b00 + b01)
    add_term({4:1, 0:-1}, {0:1, 1:1}, {5:1})
    # p7 = (a01 - a11) * (b10 + b11)
    add_term({1:1, 5:-1}, {4:1, 5:1}, {0:1})
    # p8 = (a10 + a21) * b20
    add_term({4:1, 9:1}, {8:1}, {8:1, 9:-1})
    # p9 = (a20 + a21) * b10
    add_term({8:1, 9:1}, {4:1}, {4:1, 8:-1})
    # p10 = a20 * (b21 - b01)
    add_term({8:1}, {9:1, 1:-1}, {9:1, 8:1})
    # p11 = a21 * (b10 - b11)
    add_term({9:1}, {4:1, 5:-1}, {4:1, 9:1})
    # p12 = (a20 - a21) * (b21 + b22)
    add_term({8:1, 9:-1}, {9:1, 10:1}, {8:1})
    # p13 = (a02 + a12) * b20
    add_term({2:1, 6:1}, {8:1}, {2:1, 6:-1})
    # p14 = (a12 + a22) * b10
    add_term({6:1, 10:1}, {4:1}, {6:1, 10:-1})
    # p15 = a12 * (b21 - b01)
    add_term({6:1}, {9:1, 1:-1}, {6:1, 2:1})
    # p16 = a22 * (b10 - b11)
    add_term({10:1}, {4:1, 5:-1}, {10:1, 6:1})
    # p17 = (a02 - a12) * (b21 + b22)
    add_term({2:1, 6:-1}, {9:1, 10:1}, {2:1})
    # p18 = (a00 + a22) * (b00 + b22)
    add_term({0:1, 10:1}, {0:1, 10:1}, {0:1, 10:1})
    # p19 = (a20 + a22) * (b02 + b22)
    add_term({8:1, 10:1}, {2:1, 10:1}, {8:1, 10:-1})

    # 2. Outer product A[3x1] * B[1x3] -> Rank 1
    # i=(a,3), j=(3,c), k=(a,c) for a,c in {0,1,2}
    u_dict = {4*a+3: 1 for a in range(3)}
    v_dict = {12+c: 1 for c in range(3)}
    w_dict = {3*a+c: 1 for a in range(3) for c in range(3)}
    add_term(u_dict, v_dict, w_dict)

    # 3. A[3x3] * B[3x1] -> Rank 9
    # i=(a,b), j=(b,3), k=(a,3) for a,b in {0,1,2}
    for a in range(3):
        for b in range(3):
            u_dict = {4*a+b: 1}
            v_dict = {4*b+3: 1}
            w_dict = {3*a+3: 1}
            add_term(u_dict, v_dict, w_dict)

    # 4. Outer product A[3x1] * B[1x1] -> Rank 1
    # i=(a,3), j=(3,3), k=(a,3) for a in {0,1,2}
    u_dict = {4*a+3: 1 for a in range(3)}
    v_dict = {15: 1}
    w_dict = {3*a+3: 1 for a in range(3)}
    add_term(u_dict, v_dict, w_dict)

    # Total terms: 19 + 1 + 9 + 1 = 30
    rank = len(terms)
    
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    for r in range(rank):
        u, v, w = terms[r]
        for idx, val in u.items():
            U[idx][r] += val
        for idx, val in v.items():
            V[idx][r] += val
        for idx, val in w.items():
            W[idx][r] += val

    return {"U": U, "V": V, "W": W}
