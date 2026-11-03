"""Optimized decomposition for 2x3 x 3x8 matrix multiplication."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # U row: i*m+j, V row: j*p+k, W row: k*n+i

    terms = []  # each term: (u_dict, v_dict, w_dict)

    def add(u_rows, v_rows, w_rows):
        u_d, v_d, w_d = {}, {}, {}
        for r, val in u_rows:
            u_d[r] = u_d.get(r, 0) + val
        for r, val in v_rows:
            v_d[r] = v_d.get(r, 0) + val
        for r, val in w_rows:
            w_d[r] = w_d.get(r, 0) + val
        terms.append((u_d, v_d, w_d))

    # Part A: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    # U idx = i*3+j, V idx = j*8+k, W idx = k*2+i
    # M1=(a00+a11)(b00+b11): C00+=, C11+=, C10-=, C01+=
    add([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1),(2,-1),(1,1)])
    # M2=(a10+a11)b00: C10+=, C11-=
    add([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)])
    # M3=a00(b01-b11): C01+=, C10+=
    add([(0,1)], [(1,1),(3,-1)], [(1,1),(2,1)])
    # M4=a11(b10-b00): C00+=, C11+=
    add([(3,1)], [(2,1),(0,-1)], [(0,1),(3,1)])
    # M5=(a00+a01)b11: C00-=, C01+=
    add([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)])
    # M6=(a10-a00)(b00+b01): C11+=
    add([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)])
    # M7=(a01-a11)(b10+b11): C00+=
    add([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)])

    # Part B: direct for j=2, k in {0,1}
    for i in range(2):
        for k in range(0, 2):
            add([(i*3+2, 1)], [(2*8+k, 1)], [(k*2+i, 1)])

    # Part C: direct for j=2, k in {2,7}
    for i in range(2):
        for k in range(2, 8):
            add([(i*3+2, 1)], [(2*8+k, 1)], [(k*2+i, 1)])

    rank = len(terms)
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    for c, (u_d, v_d, w_d) in enumerate(terms):
        for r, val in u_d.items():
            U[r][c] = val
        for r, val in v_d.items():
            V[r][c] = val
        for r, val in w_d.items():
            W[r][c] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
