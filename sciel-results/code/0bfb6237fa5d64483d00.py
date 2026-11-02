"""Matrix multiplication 2x5x5 via block decomposition (Strassen + Schoolbook)."""
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[a, c] * V_arr[b, c] * W_arr[c_idx, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 matmul with rank 7."""
    # Indices:
    # A: 2x2 -> 4 entries (0..3)
    # B: 2x2 -> 4 entries (0..3)
    # C: 2x2 -> 4 entries (0..3)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    
    # Strassen terms:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A11 - A10)(B10 + B11)
    
    # Mapping to indices:
    # A: [0, 1, 2, 3] -> [0,0], [0,1], [1,0], [1,1]
    # B: [0, 1, 2, 3] -> [0,0], [0,1], [1,0], [1,1]
    # C: [0, 1, 2, 3] -> [0,0], [0,1], [1,0], [1,1]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[1][5] = 1
    
    # M7
    U[3][6] = 1; U[2][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = 1
    
    return U, V, W

def _schoolbook_2x2x1():
    """Returns U, V, W for 2x2x1 matmul with rank 4."""
    # A: 2x2, B: 2x1, C: 2x1
    # U: (4, 4), V: (2, 4), W: (2, 4)
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    
    # Term 0: A00 B00 -> C0
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: A01 B10 -> C0
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    # Term 2: A10 B00 -> C1
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
    # Term 3: A11 B10 -> C1
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    
    return U, V, W

def _rank1_2x1x5():
    """Returns U, V, W for 2x1x5 matmul with rank 1."""
    # A: 2x1, B: 1x5, C: 2x5
    # U: (2, 1), V: (5, 1), W: (10, 1)
    U = [[1], [1]]
    V = [[1], [1], [1], [1], [1]]
    W = [[1] for _ in range(10)]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Total rank 37
    total_rank = 37
    U = [[0] * total_rank for _ in range(n * m)] # 10 x 37
    V = [[0] * total_rank for _ in range(m * p)] # 25 x 37
    W = [[0] * total_rank for _ in range(p * n)] # 10 x 37
    
    # We need to map the sub-problems to the global indices.
    # Global A: 2x5. Indices i*m+j.
    # Global B: 5x5. Indices j*p+k.
    # Global C: 2x5. Indices k*n+i.
    
    # Sub-problem 1: A[:, 0:2] * B[0:2, :]
    # A1: 2x2 (cols 0,1). B1: 2x5 (rows 0,1).
    # Split B1 into B1a (cols 0,1), B1b (cols 2,3), B1c (col 4).
    
    # Sub 1a: A1 * B1a (2x2 * 2x2)
    U1, V1, W1 = _strassen_2x2x2()
    # Map A1 indices (0..3) to Global A (0..3)
    # Map B1a indices (0..3) to Global B (0..3)
    # Map C1a indices (0..3) to Global C (0..3)
    for c in range(7):
        for i in range(4):
            U[i][c] += U1[i][c]
        for j in range(4):
            V[j][c] += V1[j][c]
        for k in range(4):
            W[k][c] += W1[k][c]
            
    # Sub 1b: A1 * B1b (2x2 * 2x2)
    # B1b is B[0:2, 2:4]. Global B indices: 2,3, 7,8 (j*p+k).
    # j=0, k=2 -> 2. j=0, k=3 -> 3. j=1, k=2 -> 7. j=1, k=3 -> 8.
    # A1 is same.
    # C1b is C[:, 2:4]. Global C indices: k*n+i.
    # k=2, i=0 -> 4. k=2, i=1 -> 5. k=3, i=0 -> 6. k=3, i=1 -> 7.
    
    U1, V1, W1 = _strassen_2x2x2()
    base_c = 7
    # Map V indices
    v_map = [2, 3, 7, 8]
    # Map W indices
    w_map = [4, 5, 6, 7]
    for c in range(7):
        for i in range(4):
            U[i][base_c + c] += U1[i][c]
        for j in range(4):
            V[v_map[j]][base_c + c] += V1[j][c]
        for k in range(4):
            W[w_map[k]][base_c + c] += W1[k][c]

    # Sub 1c: A1 * B1c (2x2 * 2x1)
    # B1c is B[0:2, 4]. Global B indices: 4, 9.
    # C1c is C[:, 4]. Global C indices: 8, 9.
    
    U1, V1, W1 = _schoolbook_2x2x1()
    base_c = 14
    v_map = [4, 9]
    w_map = [8, 9]
    for c in range(4):
        for i in range(4):
            U[i][base_c + c] += U1[i][c]
        for j in range(2):
            V[v_map[j]][base_c + c] += V1[j][c]
        for k in range(2):
            W[w_map[k]][base_c + c] += W1[k][c]

    # Sub-problem 2: A[:, 2:4] * B[2:4, :]
    # A2: 2x2 (cols 2,3). Global A indices: 2,3, 7,8.
    # B2: 2x5 (rows 2,3).
    # Split B2 into B2a (cols 0,1), B2b (cols 2,3), B2c (col 4).
    
    # Sub 2a: A2 * B2a
    # B2a is B[2:4, 0:2]. Global B indices: 10, 11, 15, 16.
    # C2a is C[:, 0:2]. Global C indices: 0, 1, 2, 3.
    
    U1, V1, W1 = _strassen_2x2x2()
    base_c = 18
    u_map = [2, 3, 7, 8]
    v_map = [10, 11, 15, 16]
    w_map = [0, 1, 2, 3]
    for c in range(7):
        for i in range(4):
            U[u_map[i]][base_c + c] += U1[i][c]
        for j in range(4):
            V[v_map[j]][base_c + c] += V1[j][c]
        for k in range(4):
            W[w_map[k]][base_c + c] += W1[k][c]

    # Sub 2b: A2 * B2b
    # B2b is B[2:4, 2:4]. Global B indices: 12, 13, 17, 18.
    # C2b is C[:, 2:4]. Global C indices: 4, 5, 6, 7.
    
    U1, V1, W1 = _strassen_2x2x2()
    base_c = 25
    u_map = [2, 3, 7, 8]
    v_map = [12, 13, 17, 18]
    w_map = [4, 5, 6, 7]
    for c in range(7):
        for i in range(4):
            U[u_map[i]][base_c + c] += U1[i][c]
        for j in range(4):
            V[v_map[j]][base_c + c] += V1[j][c]
        for k in range(4):
            W[w_map[k]][base_c + c] += W1[k][c]

    # Sub 2c: A2 * B2c
    # B2c is B[2:4, 4]. Global B indices: 14, 19.
    # C2c is C[:, 4]. Global C indices: 8, 9.
    
    U1, V1, W1 = _schoolbook_2x2x1()
    base_c = 32
    u_map = [2, 3, 7, 8]
    v_map = [14, 19]
    w_map = [8, 9]
    for c in range(4):
        for i in range(4):
            U[u_map[i]][base_c + c] += U1[i][c]
        for j in range(2):
            V[v_map[j]][base_c + c] += V1[j][c]
        for k in range(2):
            W[w_map[k]][base_c + c] += W1[k][c]

    # Sub-problem 3: A[:, 4:5] * B[4:5, :]
    # A3: 2x1 (col 4). Global A indices: 4, 9.
    # B3: 1x5 (row 4). Global B indices: 20, 21, 22, 23, 24.
    # C3: 2x5. Global C indices: 0..9.
    
    U1, V1, W1 = _rank1_2x1x5()
    base_c = 36
    u_map = [4, 9]
    v_map = [20, 21, 22, 23, 24]
    w_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for c in range(1):
        for i in range(2):
            U[u_map[i]][base_c + c] += U1[i][c]
        for j in range(5):
            V[v_map[j]][base_c + c] += V1[j][c]
        for k in range(10):
            W[w_map[k]][base_c + c] += W1[k][c]

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
