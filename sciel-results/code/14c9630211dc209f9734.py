"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_2x2x3():
    """
    Build 2x2x3 decomposition.
    A is 2x2, B is 2x3, C is 2x3.
    Split B and C into 2x2 and 2x1 parts.
    Rank = 7 + 4 = 11.
    """
    U_strassen, V_strassen, W_strassen = _build_strassen_2x2()
    
    rank = 11
    U = [[0]*rank for _ in range(4)] # 2x2 = 4 rows
    V = [[0]*rank for _ in range(6)] # 2x3 = 6 rows
    W = [[0]*rank for _ in range(6)] # 2x3 = 6 rows
    
    # Part 1: 2x2x2 Strassen for cols 0,1
    for t in range(7):
        col = t
        # U
        for i in range(4):
            U[i][col] = U_strassen[i][t]
        # V
        for j in range(4):
            V[j][col] = V_strassen[j][t]
        # W
        for k in range(4):
            W[k][col] = W_strassen[k][t]
    
    # Part 2: 2x2x1 Schoolbook for col 2
    # Rank 4
    # C[:,2] = A * B[:,2]
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # Terms:
    # 1. A[0,0]*B[0,2] -> C[0,2]
    # 2. A[0,1]*B[1,2] -> C[0,2]
    # 3. A[1,0]*B[0,2] -> C[1,2]
    # 4. A[1,1]*B[1,2] -> C[1,2]
    
    base = 7
    for t in range(4):
        col = base + t
        # U (A indices)
        if t == 0: U[0][col] = 1  # A[0,0]
        elif t == 1: U[1][col] = 1  # A[0,1]
        elif t == 2: U[2][col] = 1  # A[1,0]
        elif t == 3: U[3][col] = 1  # A[1,1]
        
        # V (B indices)
        # B is 2x3. Row j, Col k. Index j*3+k.
        # We need B[0,2] and B[1,2]. Indices 0*3+2=2, 1*3+2=5.
        if t == 0: V[2][col] = 1  # B[0,2]
        elif t == 1: V[5][col] = 1  # B[1,2]
        elif t == 2: V[2][col] = 1  # B[0,2]
        elif t == 3: V[5][col] = 1  # B[1,2]
        
        # W (C indices)
        # C is 2x3. Row i, Col k. Index i*3+k.
        # We need C[0,2] and C[1,2]. Indices 0*3+2=2, 1*3+2=5.
        if t == 0: W[2][col] = 1  # C[0,2]
        elif t == 1: W[2][col] = 1  # C[0,2]
        elif t == 2: W[5][col] = 1  # C[1,2]
        elif t == 3: W[5][col] = 1  # C[1,2]
        
    return U, V, W

def _build_strassen_2x3x6():
    """
    Build 2x3x6 decomposition.
    A is 2x3, B is 3x6, C is 2x6.
    Split A into 2x2 and 2x1. B into 2x6 and 1x6. C into 2x6.
    Block 1: A11 (2x2) * B11 (2x6) -> C11 (2x6). Rank 22.
    Block 2: A12 (2x1) * B21 (1x6) -> C12 (2x6). Rank 1.
    Total Rank = 23.
    """
    U223, V223, W223 = _build_strassen_2x2x3()
    
    rank = 23
    U = [[0]*rank for _ in range(6)] # 2x3 = 6 rows
    V = [[0]*rank for _ in range(18)] # 3x6 = 18 rows
    W = [[0]*rank for _ in range(12)] # 2x6 = 12 rows
    
    def add_2x2x3_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(11):
            col = base_rank + t
            # U
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff = U223[local_a_row][t]
                    if coeff != 0:
                        gi = a_i_start + li
                        gj = a_j_start + lj
                        u_row = gi * 3 + gj
                        U[u_row][col] += coeff
                        
            # V
            for lj in range(2):
                for lk in range(3):
                    local_b_row = lj * 3 + lk
                    coeff = V223[local_b_row][t]
                    if coeff != 0:
                        gj = b_j_start + lj
                        gk = c_k_start + lk
                        v_row = gj * 6 + gk
                        V[v_row][col] += coeff
                        
            # W
            for lk in range(3):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff = W223[local_c_row][t]
                    if coeff != 0:
                        gk = c_k_start + lk
                        gi = c_i_start + li
                        w_row = gk * 2 + gi
                        W[w_row][col] += coeff

    def add_2x1x6_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        col = base_rank
        # U
        for li in range(2):
            gi = a_i_start + li
            gj = a_j_start
            u_row = gi * 3 + gj
            U[u_row][col] = 1
        # V
        for lk in range(6):
            gj = b_j_start
            gk = c_k_start + lk
            v_row = gj * 6 + gk
            V[v_row][col] = 1
        # W
        for lk in range(6):
            for li in range(2):
                gk = c_k_start + lk
                gi = c_i_start + li
                w_row = gk * 2 + gi
                W[w_row][col] = 1

    # 1. A11 * B11 (Left) -> C11 (Left)
    add_2x2x3_block(0, 0, 0, 0, 0, 0)
    # 2. A11 * B11 (Right) -> C11 (Right)
    add_2x2x3_block(0, 0, 0, 0, 3, 11)
    # 3. A12 * B21 -> C12
    add_2x1x6_block(0, 2, 2, 0, 0, 22)
    
    return U, V, W

def _build_strassen_2x1x6():
    """
    Build 2x1x6 decomposition.
    A is 2x1, B is 1x6, C is 2x6.
    Rank = 1.
    """
    rank = 1
    U = [[0]*rank for _ in range(2)] # 2x1 = 2 rows
    V = [[0]*rank for _ in range(6)] # 1x6 = 6 rows
    W = [[0]*rank for _ in range(12)] # 2x6 = 12 rows
    
    # U
    for li in range(2):
        U[li][0] = 1
    # V
    for lk in range(6):
        V[lk][0] = 1
    # W
    for lk in range(6):
        for li in range(2):
            W[lk * 2 + li][0] = 1
            
    return U, V, W

def _build_strassen_4x5x6():
    """
    Build 4x5x6 decomposition.
    A is 4x5, B is 5x6, C is 4x6.
    Split A into 2x5 and 2x5. B into 2x6 and 3x6. C into 2x6 and 2x6.
    Block 1: A11 (2x5) * B11 (2x6) -> C11 (2x6). Rank 23.
    Block 2: A12 (2x5) * B21 (3x6) -> C12 (2x6). Rank 46.
    Block 3: A21 (2x5) * B11 (2x6) -> C21 (2x6). Rank 23.
    Block 4: A22 (2x5) * B21 (3x6) -> C22 (2x6). Rank 46.
    Total Rank = 23 + 46 + 23 + 46 = 138.
    """
    U256, V256, W256 = _build_strassen_2x5x6()
    U236, V236, W236 = _build_strassen_2x3x6()
    
    rank = 138
    U = [[0]*rank for _ in range(20)] # 4x5 = 20 rows
    V = [[0]*rank for _ in range(30)] # 5x6 = 30 rows
    W = [[0]*rank for _ in range(24)] # 4x6 = 24 rows
    
    def add_2x5x6_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(23):
            col = base_rank + t
            # U
            for li in range(2):
                for lj in range(5):
                    local_a_row = li * 5 + lj
                    coeff = U256[local_a_row][t]
                    if coeff != 0:
                        gi = a_i_start + li
                        gj = a_j_start + lj
                        u_row = gi * 5 + gj
                        U[u_row][col] += coeff
                        
            # V
            for lj in range(2):
                for lk in range(6):
                    local_b_row = lj * 6 + lk
                    coeff = V256[local_b_row][t]
                    if coeff != 0:
                        gj = b_j_start + lj
                        gk = c_k_start + lk
                        v_row = gj * 6 + gk
                        V[v_row][col] += coeff
                        
            # W
            for lk in range(6):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff = W256[local_c_row][t]
                    if coeff != 0:
                        gk = c_k_start + lk
                        gi = c_i_start + li
                        w_row = gk * 4 + gi
                        W[w_row][col] += coeff

    def add_2x3x6_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(23):
            col = base_rank + t
            # U
            for li in range(2):
                for lj in range(3):
                    local_a_row = li * 3 + lj
                    coeff = U236[local_a_row][t]
                    if coeff != 0:
                        gi = a_i_start + li
                        gj = a_j_start + lj
                        u_row = gi * 5 + gj
                        U[u_row][col] += coeff
                        
            # V
            for lj in range(3):
                for lk in range(6):
                    local_b_row = lj * 6 + lk
                    coeff = V236[local_b_row][t]
                    if coeff != 0:
                        gj = b_j_start + lj
                        gk = c_k_start + lk
                        v_row = gj * 6 + gk
                        V[v_row][col] += coeff
                        
            # W
            for lk in range(6):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff = W236[local_c_row][t]
                    if coeff != 0:
                        gk = c_k_start + lk
                        gi = c_i_start + li
                        w_row = gk * 4 + gi
                        W[w_row][col] += coeff

    # 1. A11 * B11 -> C11
    add_2x5x6_block(0, 0, 0, 0, 0, 0)
    # 2. A12 * B21 -> C12
    add_2x3x6_block(0, 2, 2, 0, 0, 23)
    # 3. A21 * B11 -> C21
    add_2x5x6_block(2, 0, 0, 2, 0, 46)
    # 4. A22 * B21 -> C22
    add_2x3x6_block(2, 2, 2, 2, 0, 69)
    
    return U, V, W

def _build_strassen_2x5x6():
    """
    Build 2x5x6 decomposition.
    A is 2x5, B is 5x6, C is 2x6.
    Split A into 2x2 and 2x3. B into 2x6 and 3x6. C into 2x6.
    Block 1: A11 (2x2) * B11 (2x6) -> C11 (2x6). Rank 22.
    Block 2: A12 (2x3) * B21 (3x6) -> C12 (2x6). Rank 46.
    Total Rank = 68.
    """
    U226, V226, W226 = _build_strassen_2x2x6()
    U236, V236, W236 = _build_strassen_2x3x6()
    
    rank = 68
    U = [[0]*rank for _ in range(10)] # 2x5 = 10 rows
    V = [[0]*rank for _ in range(30)] # 5x6 = 30 rows
    W = [[0]*rank for _ in range(12)] # 2x6 = 12 rows
    
    def add_2x2x6_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(22):
            col = base_rank + t
            # U
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff = U226[local_a_row][t]
                    if coeff != 0:
                        gi = a_i_start + li
                        gj = a_j_start + lj
                        u_row = gi * 5 + gj
                        U[u_row][col] += coeff
                        
            # V
            for lj in range(2):
                for lk in range(6):
                    local_b_row = lj * 6 + lk
                    coeff = V226[local_b_row][t]
                    if coeff != 0:
                        gj = b_j_start + lj
                        gk = c_k_start + lk
                        v_row = gj * 6 + gk
                        V[v_row][col] += coeff
                        
            # W
            for lk in range(6):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff = W226[local_c_row][t]
                    if coeff != 0:
                        gk = c_k_start + lk
                        gi = c_i_start + li
                        w_row = gk * 2 + gi
                        W[w_row][col] += coeff

    def add_2x3x6_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(23):
            col = base_rank + t
            # U
            for li in range(2):
                for lj in range(3):
                    local_a_row = li * 3 + lj
                    coeff = U236[local_a_row][t]
                    if coeff != 0:
                        gi = a_i_start + li
                        gj = a_j_start + lj
                        u_row = gi * 5 + gj
                        U[u_row][col] += coeff
                        
            # V
            for lj in range(3):
                for lk in range(6):
                    local_b_row = lj * 6 + lk
                    coeff = V236[local_b_row][t]
                    if coeff != 0:
                        gj = b_j_start + lj
                        gk = c_k_start + lk
                        v_row = gj * 6 + gk
                        V[v_row][col] += coeff
                        
            # W
            for lk in range(6):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff = W236[local_c_row][t]
                    if coeff != 0:
                        gk = c_k_start + lk
                        gi = c_i_start + li
                        w_row = gk * 2 + gi
                        W[w_row][col] += coeff

    # 1. A11 * B11 -> C11
    add_2x2x6_block(0, 0, 0, 0, 0, 0)
    # 2. A12 * B21 -> C12
    add_2x3x6_block(0, 2, 2, 0, 0, 22)
    
    return U, V, W

def _build_strassen_2x2x6():
    """
    Build 2x2x6 decomposition.
    A is 2x2, B is 2x6, C is 2x6.
    Split B into 2x3 and 2x3. C into 2x3 and 2x3.
    Block 1: A11 * B11 (Left) -> C11 (Left). Rank 11.
    Block 2: A11 * B11 (Right) -> C11 (Right). Rank 11.
    Total Rank = 22.
    """
    U223, V223, W223 = _build_strassen_2x2x3()
    
    rank = 22
    U = [[0]*rank for _ in range(4)] # 2x2 = 4 rows
    V = [[0]*rank for _ in range(12)] # 2x6 = 12 rows
    W = [[0]*rank for _ in range(12)] # 2x6 = 12 rows
    
    def add_2x2x3_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(11):
            col = base_rank + t
            # U
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff = U223[local_a_row][t]
                    if coeff != 0:
                        gi = a_i_start + li
                        gj = a_j_start + lj
                        u_row = gi * 2 + gj
                        U[u_row][col] += coeff
                        
            # V
            for lj in range(2):
                for lk in range(3):
                    local_b_row = lj * 3 + lk
                    coeff = V223[local_b_row][t]
                    if coeff != 0:
                        gj = b_j_start + lj
                        gk = c_k_start + lk
                        v_row = gj * 6 + gk
                        V[v_row][col] += coeff
                        
            # W
            for lk in range(3):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff = W223[local_c_row][t]
                    if coeff != 0:
                        gk = c_k_start + lk
                        gi = c_i_start + li
                        w_row = gk * 2 + gi
                        W[w_row][col] += coeff

    # 1. A11 * B11 (Left) -> C11 (Left)
    add_2x2x3_block(0, 0, 0, 0, 0, 0)
    # 2. A11 * B11 (Right) -> C11 (Right)
    add_2x2x3_block(0, 0, 0, 0, 3, 11)
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # Option 1: 4x5x6 Strassen (138)
        U, V, W = _build_strassen_4x5x6()
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
