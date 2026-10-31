"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """Recursive Strassen decomposition for 4x4 matrix multiplication, rank 49."""
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 10.0
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    # W index: trace convention W[k*n + i, r] where C[i][k] gets contribution
    # So for C element at row=i, col=k: index = k*n + i
    
    def w_idx(i, k):
        """W index for C[i][k]"""
        return k * n + i
    
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Outer level: 4x4 as 2x2 blocks of 2x2
    # Each outer term is X*Y where X, Y are 2x2 matrices of linear combos
    
    # Represent a 2x2 matrix of linear combos as:
    # mat[i][j] = list of (global_row, global_col, coeff)
    # For A: global index = row * 4 + col (row, col in 0..3)
    # For B: global index = row * 4 + col (row, col in 0..3)
    
    def make_A_block(br, bc):
        """2x2 matrix where each entry is a single A element"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append([(br*2+i, bc*2+j, 1)])
            mat.append(row)
        return mat
    
    def make_B_block(br, bc):
        """2x2 matrix where each entry is a single B element"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append([(br*2+i, bc*2+j, 1)])
            mat.append(row)
        return mat
    
    def add_mat(X, Y):
        """Add two 2x2 matrices of linear combos"""
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
        """Subtract two 2x2 matrices of linear combos"""
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
    
    # Outer Strassen terms: (X, Y) pairs
    # X is 2x2 of A linear combos, Y is 2x2 of B linear combos
    outer_X = [
        add_mat(A00, A11),       # M1
        add_mat(A10, A11),       # M2
        A00,                       # M3
        A11,                       # M4
        add_mat(A00, A01),       # M5
        sub_mat(A10, A00),       # M6
        sub_mat(A01, A11),       # M7
    ]
    outer_Y = [
        add_mat(B00, B11),       # M1
        B00,                       # M2
        sub_mat(B01, B11),       # M3
        sub_mat(B10, B00),       # M4
        B11,                       # M5
        add_mat(B00, B01),       # M6
        add_mat(B10, B11),       # M7
    ]
    
    # For each outer term t, the resulting 2x2 block Z contributes to C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So outer term t contributes to C blocks with coefficients:
    # c_targets[t] = list of (block_row, block_col, coeff)
    c_targets = [
        [(0,0,1), (1,1,1)],       # M1 -> C00(+1), C11(+1)
        [(1,0,1), (1,1,-1)],      # M2 -> C10(+1), C11(-1)
        [(0,1,1), (1,1,1)],       # M3 -> C01(+1), C11(+1)
        [(0,0,1), (1,0,1)],       # M4 -> C00(+1), C10(+1)
        [(0,0,-1), (0,1,1)],      # M5 -> C00(-1), C01(+1)
        [(1,1,1)],                 # M6 -> C11(+1)
        [(0,0,1)],                 # M7 -> C00(+1)
    ]
    
    # Inner Strassen on 2x2 matrices X, Y (each entry is a linear combo)
    # Produces 7 rank-1 terms, each contributing to Z[li][lj]
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For inner term k, the Z contributions are:
    # inner_targets[k] = list of (local_i, local_j, coeff)
    inner_targets = [
        [(0,0,1), (1,1,1)],       # inner M1 -> Z00(+1), Z11(+1)
        [(1,0,1), (1,1,-1)],      # inner M2 -> Z10(+1), Z11(-1)
        [(0,1,1), (1,1,1)],       # inner M3 -> Z01(+1), Z11(+1)
        [(0,0,1), (1,0,1)],       # inner M4 -> Z00(+1), Z10(+1)
        [(0,0,-1), (0,1,1)],      # inner M5 -> Z00(-1), Z01(+1)
        [(1,1,1)],                 # inner M6 -> Z11(+1)
        [(0,0,1)],                 # inner M7 -> Z00(+1)
    ]
    
    # Inner Strassen factor pairs for 2x2: (X_factor, Y_factor)
    # where X_factor[i][j] and Y_factor[i][j] are linear combos from X and Y
    def inner_factors(X, Y):
        """Return list of 7 (X_f, Y_f) pairs for inner Strassen"""
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        def single_entry(i, j, mat):
            """Create 2x2 matrix with only entry (i,j) from mat"""
            result = [[[] for _ in range(2)] for _ in range(2)]
            result[i][j] = list(mat[i][j])
            return result
        
        def add_entries(i1, j1, i2, j2, mat):
            """Create 2x2 matrix with entries (i1,j1) and (i2,j2) from mat"""
            result = [[[] for _ in range(2)] for _ in range(2)]
            d = {}
            for (r, c, v) in mat[i1][j1]:
                d[(r,c)] = d.get((r,c), 0) + v
            for (r, c, v) in mat[i2][j2]:
                d[(r,c)] = d.get((r,c), 0) + v
            result[i1][j1] = [(r, c, v) for (r,c), v in d.items() if v != 0]
            return result
        
        def sub_entries(i1, j1, i2, j2, mat):
            """Create 2x2 matrix with entry (i1,j1) = mat[i1][j1] - mat[i2][j2]"""
            result = [[[] for _ in range(2)] for _ in range(2)]
            d = {}
            for (r, c, v) in mat[i1][j1]:
                d[(r,c)] = d.get((r,c), 0) + v
            for (r, c, v) in mat[i2][j2]:
                d[(r,c)] = d.get((r,c), 0) - v
            result[i1][j1] = [(r, c, v) for (r,c), v in d.items() if v != 0]
            return result
        
        # M1: (X00+X11)(Y00+Y11)
        X1 = add_entries(0, 0, 1, 1, X)
        Y1 = add_entries(0, 0, 1, 1, Y)
        
        # M2: (X10+X11)Y00
        X2 = add_entries(1, 0, 1, 1, X)
        Y2 = single_entry(0, 0, Y)
        
        # M3: X00(Y01-Y11)
        X3 = single_entry(0, 0, X)
        Y3 = sub_entries(0, 1, 1, 1, Y)
        
        # M4: X11(Y10-Y00)
        X4 = single_entry(1, 1, X)
        Y4 = sub_entries(1, 0, 0, 0, Y)
        
        # M5: (X00+X01)Y11
        X5 = add_entries(0, 0, 0, 1, X)
        Y5 = single_entry(1, 1, Y)
        
        # M6: (X10-X00)(Y00+Y01)
        X6 = sub_entries(1, 0, 0, 0, X)
        Y6 = add_entries(0, 0, 0, 1, Y)
        
        # M7: (X01-X11)(Y10+Y11)
        X7 = sub_entries(0, 1, 1, 1, X)
        Y7 = add_entries(1, 0, 1, 1, Y)
        
        return [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6), (X7, Y7)]
    
    col = 0
    for t in range(7):
        X = outer_X[t]
        Y = outer_Y[t]
        factors = inner_factors(X, Y)
        
        for k in range(7):
            Xf, Yf = factors[k]
            
            # Fill U from Xf (linear combos of A entries)
            for i in range(2):
                for j in range(2):
                    for (a_r, a_c, a_v) in Xf[i][j]:
                        U[a_r * m + a_c][col] += a_v
            
            # Fill V from Yf (linear combos of B entries)
            for i in range(2):
                for j in range(2):
                    for (b_r, b_c, b_v) in Yf[i][j]:
                        V[b_r * p + b_c][col] += b_v
            
            # Fill W from c_targets[t] and inner_targets[k]
            for (br, bc, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    # Z[li][lj] contributes to C block (br, bc) with total coeff
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    W[w_idx(global_i, global_k)][col] += c_coeff * z_coeff
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() > _DEADLINE - _MARGIN:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
