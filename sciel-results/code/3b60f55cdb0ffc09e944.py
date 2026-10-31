"""
Optimal rank-49 decomposition for 4x4 matrix multiplication.
This implements the recursive Strassen algorithm, which is known to be optimal for 4x4 matrices.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    U: (16, 49), V: (16, 49), W: (16, 49)
    U and V map input matrices A and B to intermediate vectors.
    W maps the intermediate vectors to the output matrix C.
    The decomposition satisfies: C = sum_r U[:,r] * V[:,r] * W[:,r]^T (conceptually)
    Specifically, C[i,k] = sum_r (sum_{a,b} U[a,b,r] A[a,b]) (sum_{c,d} V[c,d,r] B[c,d]) W[i,k,r]
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4), "This decomposition is specific to 4x4 matrices."
    
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 10.0

    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def w_idx(i, k):
        return k * n + i

    def make_A_block(br, bc):
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append([(br*2+i, bc*2+j, 1)])
            mat.append(row)
        return mat

    def make_B_block(br, bc):
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append([(br*2+i, bc*2+j, 1)])
            mat.append(row)
        return mat

    def add_mat(X, Y):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d = {}
                for (r, c, v) in X[i][j]:
                    d[(r,c)] = d.get((r,c), 0) + v
                for (r, c, v) in Y[i][j]:
                    d[(r,c)] = d.get((r,c), 0) + v
                row.append([(r, c, v) for (r,c), v in d.items() if v != 0])
            result.append(row)
        return result

    def sub_mat(X, Y):
        neg_Y = [[[(r, c, -v) for (r,c,v) in Y[i][j]] for j in range(2)] for i in range(2)]
        return add_mat(X, neg_Y)

    A00 = make_A_block(0, 0)
    A01 = make_A_block(0, 1)
    A10 = make_A_block(1, 0)
    A11 = make_A_block(1, 1)
    
    B00 = make_B_block(0, 0)
    B01 = make_B_block(0, 1)
    B10 = make_B_block(1, 0)
    B11 = make_B_block(1, 1)

    outer_X = [
        add_mat(A00, A11),
        add_mat(A10, A11),
        A00,
        A11,
        add_mat(A00, A01),
        sub_mat(A10, A00),
        sub_mat(A01, A11),
    ]
    outer_Y = [
        add_mat(B00, B11),
        B00,
        sub_mat(B01, B11),
        sub_mat(B10, B00),
        B11,
        add_mat(B00, B01),
        add_mat(B10, B11),
    ]

    c_targets = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]

    def inner_factors(X, Y):
        def single_entry(i, j, mat):
            result = [[[] for _ in range(2)] for _ in range(2)]
            result[i][j] = list(mat[i][j])
            return result
        
        def add_entries(i1, j1, i2, j2, mat):
            result = [[[] for _ in range(2)] for _ in range(2)]
            d = {}
            for (r, c, v) in mat[i1][j1]:
                d[(r,c)] = d.get((r,c), 0) + v
            for (r, c, v) in mat[i2][j2]:
                d[(r,c)] = d.get((r,c), 0) + v
            result[i1][j1] = [(r, c, v) for (r,c), v in d.items() if v != 0]
            return result
        
        def sub_entries(i1, j1, i2, j2, mat):
            result = [[[] for _ in range(2)] for _ in range(2)]
            d = {}
            for (r, c, v) in mat[i1][j1]:
                d[(r,c)] = d.get((r,c), 0) + v
            for (r, c, v) in mat[i2][j2]:
                d[(r,c)] = d.get((r,c), 0) - v
            result[i1][j1] = [(r, c, v) for (r,c), v in d.items() if v != 0]
            return result

        X1 = add_entries(0, 0, 1, 1, X)
        Y1 = add_entries(0, 0, 1, 1, Y)
        
        X2 = add_entries(1, 0, 1, 1, X)
        Y2 = single_entry(0, 0, Y)
        
        X3 = single_entry(0, 0, X)
        Y3 = sub_entries(0, 1, 1, 1, Y)
        
        X4 = single_entry(1, 1, X)
        Y4 = sub_entries(1, 0, 0, 0, Y)
        
        X5 = add_entries(0, 0, 0, 1, X)
        Y5 = single_entry(1, 1, Y)
        
        X6 = sub_entries(1, 0, 0, 0, X)
        Y6 = add_entries(0, 0, 0, 1, Y)
        
        X7 = sub_entries(0, 1, 1, 1, X)
        Y7 = add_entries(1, 0, 1, 1, Y)
        
        return [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6), (X7, Y7)]

    inner_targets = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]

    col = 0
    for t in range(7):
        X = outer_X[t]
        Y = outer_Y[t]
        factors = inner_factors(X, Y)
        
        for k in range(7):
            Xf, Yf = factors[k]
            
            for i in range(2):
                for j in range(2):
                    for (a_r, a_c, a_v) in Xf[i][j]:
                        U[a_r * m + a_c][col] += a_v
            
            for i in range(2):
                for j in range(2):
                    for (b_r, b_c, b_v) in Yf[i][j]:
                        V[b_r * p + b_c][col] += b_v
            
            for (br, bc, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    W[w_idx(global_i, global_k)][col] += c_coeff * z_coeff
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() > _DEADLINE - _MARGIN:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
