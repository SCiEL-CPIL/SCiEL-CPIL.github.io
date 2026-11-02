"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_off_i, a_off_j, b_off_i, b_off_j, c_off_i, c_off_j):
        nonlocal col
        # A is 2x2 at (a_off_i, a_off_j)
        # B is 2x2 at (b_off_i, b_off_j)
        # C is 2x2 at (c_off_i, c_off_j)
        
        # M1 = (A11 + A22)(B11 + B22)
        # C11 += M1, C22 += M1
        U[a_off_i * m + a_off_j][col] += 1
        U[(a_off_i + 1) * m + (a_off_j + 1)][col] += 1
        V[b_off_i * p + b_off_j][col] += 1
        V[(b_off_i + 1) * p + (b_off_j + 1)][col] += 1
        W[c_off_j * n + c_off_i][col] += 1
        W[(c_off_j + 1) * n + (c_off_i + 1)][col] += -1
        col += 1
        
        # M2 = (A21 + A22)B11
        # C21 += M2, C22 -= M2
        U[(a_off_i + 1) * m + a_off_j][col] += 1
        U[(a_off_i + 1) * m + (a_off_j + 1)][col] += 1
        V[b_off_i * p + b_off_j][col] += 1
        W[c_off_j * n + (c_off_i + 1)][col] += 1
        W[(c_off_j + 1) * n + (c_off_i + 1)][col] += -1
        col += 1
        
        # M3 = A11(B12 - B22)
        # C12 += M3, C22 += M3
        U[a_off_i * m + a_off_j][col] += 1
        V[b_off_i * p + (b_off_j + 1)][col] += 1
        V[(b_off_i + 1) * p + (b_off_j + 1)][col] += -1
        W[(c_off_j + 1) * n + c_off_i][col] += 1
        W[(c_off_j + 1) * n + (c_off_i + 1)][col] += 1
        col += 1
        
        # M4 = A22(B21 - B11)
        # C11 += M4, C21 += M4
        U[(a_off_i + 1) * m + (a_off_j + 1)][col] += 1
        V[(b_off_i + 1) * p + b_off_j][col] += 1
        V[b_off_i * p + b_off_j][col] += -1
        W[c_off_j * n + c_off_i][col] += 1
        W[c_off_j * n + (c_off_i + 1)][col] += 1
        col += 1
        
        # M5 = (A11 + A12)B22
        # C11 -= M5, C12 += M5
        U[a_off_i * m + a_off_j][col] += 1
        U[a_off_i * m + (a_off_j + 1)][col] += 1
        V[(b_off_i + 1) * p + (b_off_j + 1)][col] += 1
        W[c_off_j * n + c_off_i][col] += -1
        W[(c_off_j + 1) * n + c_off_i][col] += 1
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        # C22 += M6
        U[(a_off_i + 1) * m + a_off_j][col] += 1
        U[a_off_i * m + a_off_j][col] += -1
        V[b_off_i * p + b_off_j][col] += 1
        V[b_off_i * p + (b_off_j + 1)][col] += 1
        W[(c_off_j + 1) * n + (c_off_i + 1)][col] += 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        # C11 += M7
        U[a_off_i * m + (a_off_j + 1)][col] += 1
        U[(a_off_i + 1) * m + (a_off_j + 1)][col] += -1
        V[(b_off_i + 1) * p + b_off_j][col] += 1
        V[(b_off_i + 1) * p + (b_off_j + 1)][col] += 1
        W[c_off_j * n + c_off_i][col] += 1
        col += 1

    def add_standard(a_off_i, a_off_j, a_h, a_w, b_off_i, b_off_j, b_h, b_w, c_off_i, c_off_j):
        nonlocal col
        # A is a_h x a_w at (a_off_i, a_off_j)
        # B is b_h x b_w at (b_off_i, b_off_j)
        # C is (a_h x b_w) at (c_off_i, c_off_j)
        # Rank is a_h * b_w
        for i in range(a_h):
            for j in range(b_w):
                U[(a_off_i + i) * m + (a_off_j + 0)][col] = 1 # A index i, 0? No.
                # A is a_h x a_w. B is a_w x b_w.
                # C[i, j] = sum_k A[i, k] B[k, j]
                # This standard addition assumes the inner dimension is 1?
                # No, this function is for M_{a_h, b_w}^1?
                # In our case, the terms were:
                # A13 (2x1) B31 (1x2) -> M_{2,2}^1. Rank 4.
                # A21 (1x2) B11 (2x2) -> M_{1,2}^2. Rank 2.
                # A23 (1x1) B31 (1x2) -> M_{1,2}^1. Rank 2.
                
                # This generic function is tricky. Let's write specific ones.
                pass

    # Block 1: i=0 (rows 0,1), j=0 (cols 0,1)
    # A11 (2x2) B11 (2x2)
    add_strassen(0, 0, 0, 0, 0, 0)
    # A12 (2x2) B21 (2x2)
    add_strassen(0, 2, 2, 0, 0, 0)
    # A13 (2x1) B31 (1x2) -> Rank 4
    for i in range(2):
        for j in range(2):
            U[(0 + i) * m + (4)][col] = 1
            V[(4) * p + (0 + j)][col] = 1
            W[(0 + j) * n + (0 + i)][col] = 1
            col += 1

    # Block 2: i=0, j=1 (cols 2,3)
    add_strassen(0, 0, 0, 2, 0, 2)
    add_strassen(0, 2, 2, 2, 0, 2)
    for i in range(2):
        for j in range(2):
            U[(0 + i) * m + (4)][col] = 1
            V[(4) * p + (2 + j)][col] = 1
            W[(2 + j) * n + (0 + i)][col] = 1
            col += 1

    # Block 3: i=0, j=2 (cols 4,5)
    add_strassen(0, 0, 0, 4, 0, 4)
    add_strassen(0, 2, 2, 4, 0, 4)
    for i in range(2):
        for j in range(2):
            U[(0 + i) * m + (4)][col] = 1
            V[(4) * p + (4 + j)][col] = 1
            W[(4 + j) * n + (0 + i)][col] = 1
            col += 1

    # Block 4: i=1 (row 2), j=0 (cols 0,1)
    # A21 (1x2) B11 (2x2) -> M_{1,2}^2. Rank 2.
    # C[0, 0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0, 1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    U[(2) * m + (0)][col] = 1
    V[(0) * p + (0)][col] = 1
    W[(0) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (1)][col] = 1
    V[(1) * p + (0)][col] = 1
    W[(0) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (0)][col] = 1
    V[(0) * p + (1)][col] = 1
    W[(1) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (1)][col] = 1
    V[(1) * p + (1)][col] = 1
    W[(1) * n + (2)][col] = 1
    col += 1
    
    # A22 (1x2) B21 (2x2) -> M_{1,2}^2. Rank 2.
    U[(2) * m + (2)][col] = 1
    V[(2) * p + (0)][col] = 1
    W[(0) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (3)][col] = 1
    V[(3) * p + (0)][col] = 1
    W[(0) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (2)][col] = 1
    V[(2) * p + (1)][col] = 1
    W[(1) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (3)][col] = 1
    V[(3) * p + (1)][col] = 1
    W[(1) * n + (2)][col] = 1
    col += 1
    
    # A23 (1x1) B31 (1x2) -> M_{1,2}^1. Rank 2.
    U[(2) * m + (4)][col] = 1
    V[(4) * p + (0)][col] = 1
    W[(0) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (4)][col] = 1
    V[(4) * p + (1)][col] = 1
    W[(1) * n + (2)][col] = 1
    col += 1

    # Block 5: i=1, j=1 (cols 2,3)
    U[(2) * m + (0)][col] = 1
    V[(0) * p + (2)][col] = 1
    W[(2) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (1)][col] = 1
    V[(1) * p + (2)][col] = 1
    W[(2) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (0)][col] = 1
    V[(0) * p + (3)][col] = 1
    W[(3) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (1)][col] = 1
    V[(1) * p + (3)][col] = 1
    W[(3) * n + (2)][col] = 1
    col += 1
    
    U[(2) * m + (2)][col] = 1
    V[(2) * p + (2)][col] = 1
    W[(2) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (3)][col] = 1
    V[(3) * p + (2)][col] = 1
    W[(2) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (2)][col] = 1
    V[(2) * p + (3)][col] = 1
    W[(3) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (3)][col] = 1
    V[(3) * p + (3)][col] = 1
    W[(3) * n + (2)][col] = 1
    col += 1
    
    U[(2) * m + (4)][col] = 1
    V[(4) * p + (2)][col] = 1
    W[(2) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (4)][col] = 1
    V[(4) * p + (3)][col] = 1
    W[(3) * n + (2)][col] = 1
    col += 1

    # Block 6: i=1, j=2 (cols 4,5)
    U[(2) * m + (0)][col] = 1
    V[(0) * p + (4)][col] = 1
    W[(4) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (1)][col] = 1
    V[(1) * p + (4)][col] = 1
    W[(4) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (0)][col] = 1
    V[(0) * p + (5)][col] = 1
    W[(5) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (1)][col] = 1
    V[(1) * p + (5)][col] = 1
    W[(5) * n + (2)][col] = 1
    col += 1
    
    U[(2) * m + (2)][col] = 1
    V[(2) * p + (4)][col] = 1
    W[(4) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (3)][col] = 1
    V[(3) * p + (4)][col] = 1
    W[(4) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (2)][col] = 1
    V[(2) * p + (5)][col] = 1
    W[(5) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (3)][col] = 1
    V[(3) * p + (5)][col] = 1
    W[(5) * n + (2)][col] = 1
    col += 1
    
    U[(2) * m + (4)][col] = 1
    V[(4) * p + (4)][col] = 1
    W[(4) * n + (2)][col] = 1
    col += 1
    U[(2) * m + (4)][col] = 1
    V[(4) * p + (5)][col] = 1
    W[(5) * n + (2)][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
