"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-84 decomposition for 3x5x6 matrix multiplication.
    Strategy:
      - Split M_{3,5,6} into M_{2,5,6} + M_{1,5,6} (rank <= R(M_{2,5,6}) + 30)
      - Split M_{2,5,6} into M_{2,2,6} + M_{2,3,6} (rank <= 21 + 33 = 54)
      - Split M_{2,3,6} into M_{2,2,6} + M_{2,1,6} (rank <= 21 + 12 = 33)
      - M_{2,2,6} = 3 * M_{2,2,2} using Strassen (rank 7 each, total 21)
      - M_{2,1,6} is schoolbook (rank 12)
      - M_{1,5,6} is schoolbook (rank 30)
    Total rank: 54 + 30 = 84
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u, v, w):
        """Add a term with coefficients u, v, w at current rank index."""
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)
    
    # Helper: create a unit vector for a given index in a list of length length
    def unit(index, length):
        vec = [0] * length
        vec[index] = 1
        return vec
    
    # Helper: create a vector with values at specific indices
    def make_vec(indices_values, length):
        vec = [0] * length
        for idx, val in indices_values.items():
            vec[idx] = val
        return vec
    
    # --- M_{1,5,6}: Schoolbook, rank 30 ---
    # A is 1x5, B is 5x6, C is 6x1
    # C_k = sum_j A_j B_jk
    for j in range(5):
        for k in range(6):
            # u: index j in [0..4]
            # v: index j*6+k in [0..29]
            # w: index k*1+0 = k in [0..5]
            add_term(unit(j, 5), unit(j * 6 + k, 30), unit(k, 6))
    
    # --- M_{2,5,6}: Split into M_{2,2,6} + M_{2,3,6} ---
    # M_{2,2,6}: A is 2x2, B is 2x6, C is 6x2
    # Use Strassen for the 2x2 part, applied to each of the 3 column blocks of B
    # Split B's 6 columns into 3 blocks of 2 columns each
    # For each block b (0,1,2), we compute a 2x2 matrix product using Strassen (rank 7)
    # Total rank for M_{2,2,6} = 3 * 7 = 21
    
    # Strassen coefficients for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For M_{2,2,6}, A has indices (i,j) with i,j in {0,1}
    # B has indices (j,k) with j in {0,1}, k in {0..5}
    # C has indices (k,i) with k in {0..5}, i in {0,1}
    #
    # For block b (0,1,2), B's columns are 2*b and 2*b+1
    # Let B11 = B[0, 2b], B12 = B[0, 2b+1]
    #     B21 = B[1, 2b], B22 = B[1, 2b+1]
    
    for b in range(3):
        col1 = 2 * b
        col2 = 2 * b + 1
        
        # M1: (A11+A22)(B11+B22)
        # u: A[0,0] and A[1,1] -> indices 0 and 1 in A (2x2)
        # v: B[0,col1] and B[1,col1] -> indices col1 and 2+col1 in B (2x6)
        add_term(make_vec({0: 1, 1: 1}, 4), make_vec({col1: 1, 2 + col1: 1}, 12), make_vec({col1: 1, 6 + col1: 1}, 12))
        
        # M2: (A21+A22)B11
        # u: A[1,0] and A[1,1] -> indices 2 and 1
        # v: B[0,col1] -> index col1
        add_term(make_vec({1: 1, 2: 1}, 4), make_vec({col1: 1}, 12), make_vec({col1: 1}, 12))
        
        # M3: A11(B12-B22)
        # u: A[0,0] -> index 0
        # v: B[0,col2] and B[1,col2] -> indices col2 and 2+col2
        add_term(make_vec({0: 1}, 4), make_vec({col2: 1, 2 + col2: -1}, 12), make_vec({col2: 1}, 12))
        
        # M4: A22(B21-B11)
        # u: A[1,1] -> index 1
        # v: B[1,col1] and B[0,col1] -> indices 2+col1 and col1
        add_term(make_vec({1: 1}, 4), make_vec({col1: -1, 2 + col1: 1}, 12), make_vec({col1: 1, 6 + col1: 1}, 12))
        
        # M5: (A11+A12)B22
        # u: A[0,0] and A[0,1] -> indices 0 and 3
        # v: B[1,col2] -> index 2+col2
        add_term(make_vec({0: 1, 3: 1}, 4), make_vec({2 + col2: 1}, 12), make_vec({6 + col2: 1}, 12))
        
        # M6: (A21-A11)(B11+B12)
        # u: A[1,0] and A[0,0] -> indices 2 and 0
        # v: B[0,col1] and B[0,col2] -> indices col1 and col2
        add_term(make_vec({0: -1, 2: 1}, 4), make_vec({col1: 1, col2: 1}, 12), make_vec({6 + col1: 1, 6 + col2: 1}, 12))
        
        # M7: (A12-A22)(B21+B22)
        # u: A[0,1] and A[1,1] -> indices 3 and 1
        # v: B[1,col1] and B[1,col2] -> indices 2+col1 and 2+col2
        add_term(make_vec({1: -1, 3: 1}, 4), make_vec({2 + col1: 1, 2 + col2: 1}, 12), make_vec({col1: 1, col2: 1}, 12))
    
    # --- M_{2,3,6}: Split into M_{2,2,6} + M_{2,1,6} ---
    # M_{2,2,6}: A is 2x2, B is 2x6, C is 6x2 (same as before, rank 21)
    for b in range(3):
        col1 = 2 * b
        col2 = 2 * b + 1
        
        # M1: (A11+A22)(B11+B22)
        add_term(make_vec({0: 1, 1: 1}, 4), make_vec({col1: 1, 2 + col1: 1}, 12), make_vec({col1: 1, 6 + col1: 1}, 12))
        
        # M2: (A21+A22)B11
        add_term(make_vec({1: 1, 2: 1}, 4), make_vec({col1: 1}, 12), make_vec({col1: 1}, 12))
        
        # M3: A11(B12-B22)
        add_term(make_vec({0: 1}, 4), make_vec({col2: 1, 2 + col2: -1}, 12), make_vec({col2: 1}, 12))
        
        # M4: A22(B21-B11)
        add_term(make_vec({1: 1}, 4), make_vec({col1: -1, 2 + col1: 1}, 12), make_vec({col1: 1, 6 + col1: 1}, 12))
        
        # M5: (A11+A12)B22
        add_term(make_vec({0: 1, 3: 1}, 4), make_vec({2 + col2: 1}, 12), make_vec({6 + col2: 1}, 12))
        
        # M6: (A21-A11)(B11+B12)
        add_term(make_vec({0: -1, 2: 1}, 4), make_vec({col1: 1, col2: 1}, 12), make_vec({6 + col1: 1, 6 + col2: 1}, 12))
        
        # M7: (A12-A22)(B21+B22)
        add_term(make_vec({1: -1, 3: 1}, 4), make_vec({2 + col1: 1, 2 + col2: 1}, 12), make_vec({col1: 1, col2: 1}, 12))
    
    # M_{2,1,6}: A is 2x1, B is 1x6, C is 6x2, schoolbook rank 12
    for j in range(1):  # j=0 only
        for k in range(6):
            # u: A[i,0] for i in {0,1}
            # v: B[0,k] -> index k
            # w: C[k,i] -> index k*2+i
            for i in range(2):
                add_term(unit(i, 2), unit(k, 6), unit(k * 2 + i, 12))
    
    # Convert to the required format
    rank = len(U_rows)
    U = [[0] * rank for _ in range(n * m)]  # 15 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0] * rank for _ in range(p * n)]  # 18 x rank
    
    # Wait, the dimensions are wrong. Let me re-read the task.
    # U: (n*m, rank) = (15, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (18, rank)
    #
    # But in my decomposition above, I used:
    # For M_{1,5,6}: A is 1x5 (5 elements), B is 5x6 (30 elements), C is 6x1 (6 elements)
    # For M_{2,2,6}: A is 2x2 (4 elements), B is 2x6 (12 elements), C is 6x2 (12 elements)
    # For M_{2,3,6} part: A is 2x2 (4 elements), B is 2x6 (12 elements), C is 6x2 (12 elements)
    # For M_{2,1,6}: A is 2x1 (2 elements), B is 1x6 (6 elements), C is 6x2 (12 elements)
    #
    # This doesn't match the required shapes. I need to map the sub-tensor indices to the full tensor indices.
    #
    # Let me restart with a cleaner approach using explicit index mappings.
    
    # Clear the previous attempts
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u, v, w):
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)
    
    # Full dimensions: n=3, m=5, p=6
    # U: 15 rows (i=0..2, j=0..4)
    # V: 30 rows (j=0..4, k=0..5)
    # W: 18 rows (k=0..5, i=0..2)
    
    # --- M_{1,5,6}: rows i=0 only ---
    # A[0,j] for j=0..4, B[j,k] for j=0..4, k=0..5, C[k,0] for k=0..5
    for j in range(5):
        for k in range(6):
            u = [0] * 15
            v = [0] * 30
            w = [0] * 18
            u[0 * 5 + j] = 1  # A[0,j]
            v[j * 6 + k] = 1  # B[j,k]
            w[k * 3 + 0] = 1  # C[k,0]
            add_term(u, v, w)
    
    # --- M_{2,5,6}: rows i=1,2 ---
    # Split columns j=0..4 into j=0,1 and j=2,3,4
    # M_{2,2,6}: i=1,2, j=0,1
    # M_{2,3,6}: i=1,2, j=2,3,4
    
    # M_{2,2,6}: Strassen applied to 3 blocks of 2 columns each
    for b in range(3):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # M1: (A11+A22)(B11+B22)
        # A11 = A[1,0], A22 = A[2,1]
        # B11 = B[0,k1], B22 = B[1,k2]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[1 * 5 + 0] = 1
        u[2 * 5 + 1] = 1
        v[0 * 6 + k1] = 1
        v[1 * 6 + k2] = 1
        w[k1 * 3 + 1] = 1
        w[k2 * 3 + 2] = 1
        add_term(u, v, w)
        
        # M2: (A21+A22)B11
        # A21 = A[2,0], A22 = A[2,1], B11 = B[0,k1]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[2 * 5 + 0] = 1
        u[2 * 5 + 1] = 1
        v[0 * 6 + k1] = 1
        w[k1 * 3 + 2] = 1
        add_term(u, v, w)
        
        # M3: A11(B12-B22)
        # A11 = A[1,0], B12 = B[0,k2], B22 = B[1,k2]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[1 * 5 + 0] = 1
        v[0 * 6 + k2] = 1
        v[1 * 6 + k2] = -1
        w[k2 * 3 + 1] = 1
        add_term(u, v, w)
        
        # M4: A22(B21-B11)
        # A22 = A[2,1], B21 = B[1,k1], B11 = B[0,k1]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[2 * 5 + 1] = 1
        v[1 * 6 + k1] = 1
        v[0 * 6 + k1] = -1
        w[k1 * 3 + 2] = 1
        add_term(u, v, w)
        
        # M5: (A11+A12)B22
        # A11 = A[1,0], A12 = A[1,1], B22 = B[1,k2]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[1 * 5 + 0] = 1
        u[1 * 5 + 1] = 1
        v[1 * 6 + k2] = 1
        w[k2 * 3 + 2] = 1
        add_term(u, v, w)
        
        # M6: (A21-A11)(B11+B12)
        # A21 = A[2,0], A11 = A[1,0], B11 = B[0,k1], B12 = B[0,k2]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[2 * 5 + 0] = 1
        u[1 * 5 + 0] = -1
        v[0 * 6 + k1] = 1
        v[0 * 6 + k2] = 1
        w[k1 * 3 + 2] = 1
        w[k2 * 3 + 2] = 1
        add_term(u, v, w)
        
        # M7: (A12-A22)(B21+B22)
        # A12 = A[1,1], A22 = A[2,1], B21 = B[1,k1], B22 = B[1,k2]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[1 * 5 + 1] = 1
        u[2 * 5 + 1] = -1
        v[1 * 6 + k1] = 1
        v[1 * 6 + k2] = 1
        w[k1 * 3 + 1] = 1
        w[k2 * 3 + 1] = 1
        add_term(u, v, w)
    
    # M_{2,3,6}: i=1,2, j=2,3,4
    # Split into M_{2,2,6} (j=2,3) and M_{2,1,6} (j=4)
    
    # M_{2,2,6} part: j=2,3
    for b in range(3):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # M1: (A11+A22)(B11+B22)
        # A11 = A[1,2], A22 = A[2,3]
        # B11 = B[2,k1], B22 = B[3,k2]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[1 * 5 + 2] = 1
        u[2 * 5 + 3] = 1
        v[2 * 6 + k1] = 1
        v[3 * 6 + k2] = 1
        w[k1 * 3 + 1] = 1
        w[k2 * 3 + 2] = 1
        add_term(u, v, w)
        
        # M2: (A21+A22)B11
        # A21 = A[2,2], A22 = A[2,3], B11 = B[2,k1]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[2 * 5 + 2] = 1
        u[2 * 5 + 3] = 1
        v[2 * 6 + k1] = 1
        w[k1 * 3 + 2] = 1
        add_term(u, v, w)
        
        # M3: A11(B12-B22)
        # A11 = A[1,2], B12 = B[2,k2], B22 = B[3,k2]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[1 * 5 + 2] = 1
        v[2 * 6 + k2] = 1
        v[3 * 6 + k2] = -1
        w[k2 * 3 + 1] = 1
        add_term(u, v, w)
        
        # M4: A22(B21-B11)
        # A22 = A[2,3], B21 = B[3,k1], B11 = B[2,k1]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[2 * 5 + 3] = 1
        v[3 * 6 + k1] = 1
        v[2 * 6 + k1] = -1
        w[k1 * 3 + 2] = 1
        add_term(u, v, w)
        
        # M5: (A11+A12)B22
        # A11 = A[1,2], A12 = A[1,3], B22 = B[3,k2]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[1 * 5 + 2] = 1
        u[1 * 5 + 3] = 1
        v[3 * 6 + k2] = 1
        w[k2 * 3 + 2] = 1
        add_term(u, v, w)
        
        # M6: (A21-A11)(B11+B12)
        # A21 = A[2,2], A11 = A[1,2], B11 = B[2,k1], B12 = B[2,k2]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[2 * 5 + 2] = 1
        u[1 * 5 + 2] = -1
        v[2 * 6 + k1] = 1
        v[2 * 6 + k2] = 1
        w[k1 * 3 + 2] = 1
        w[k2 * 3 + 2] = 1
        add_term(u, v, w)
        
        # M7: (A12-A22)(B21+B22)
        # A12 = A[1,3], A22 = A[2,3], B21 = B[3,k1], B22 = B[3,k2]
        u = [0] * 15
        v = [0] * 30
        w = [0] * 18
        u[1 * 5 + 3] = 1
        u[2 * 5 + 3] = -1
        v[3 * 6 + k1] = 1
        v[3 * 6 + k2] = 1
        w[k1 * 3 + 1] = 1
        w[k2 * 3 + 1] = 1
        add_term(u, v, w)
    
    # M_{2,1,6} part: j=4
    # A is 2x1, B is 1x6, C is 6x2
    for k in range(6):
        for i in [1, 2]:
            u = [0] * 15
            v = [0] * 30
            w = [0] * 18
            u[i * 5 + 4] = 1
            v[4 * 6 + k] = 1
            w[k * 3 + i] = 1
            add_term(u, v, w)
    
    rank = len(U_rows)
    U = U_rows
    V = V_rows
    W = W_rows
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
