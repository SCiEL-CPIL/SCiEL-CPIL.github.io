import time
import itertools

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i*m+j, j*p+k, k*n+i)
                T[key] = T.get(key, 0) + 1
    return T

def _from_columns(cols, n, m, p, rank):
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (Ac, Bc, Cc) in enumerate(cols):
        for (i, j), v in Ac.items():
            U[i*m+j][r] += v
        for (j, k), v in Bc.items():
            V[j*p+k][r] += v
        for (i, k), v in Cc.items():
            W[k*n+i][r] += v
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    target = _target_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                s = 0
                for r in range(rank):
                    s += U[a][r]*V[b][r]*W[c][r]
                if abs(s - target.get((a, b, c), 0)) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _build_low_rank(n, m, p):
    cols = []
    def strassen(ai, aj, bj, bk, ci, ck):
        def A(i, j): return (ai+i, aj+j)
        def B(j, k): return (bj+j, bk+k)
        def C(i, k): return (ci+i, ck+k)
        M = [
            ([(A(0,0),1),(A(1,1),1)], [(B(0,0),1),(B(1,1),1)], [(C(0,0),1),(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),1)], [(B(0,0),1)], [(C(1,0),1),(C(1,1),-1)]),
            ([(A(0,0),1)], [(B(1,0),1),(B(1,1),-1)], [(C(0,1),1),(C(1,1),1)]),
            ([(A(1,1),1)], [(B(0,1),1),(B(1,0),-1)], [(C(0,0),1),(C(1,0),1)]),
            ([(A(0,0),1),(A(0,1),1)], [(B(1,0),1)], [(C(0,0),-1),(C(0,1),1)]),
            ([(A(0,1),1),(A(0,0),-1)], [(B(0,0),1),(B(1,0),1)], [(C(1,1),1)]),
            ([(A(1,0),1),(A(1,1),-1)], [(B(1,1),1),(B(0,1),1)], [(C(0,0),1),(C(0,1),-1)]),
        ]
        for Ac, Bc, Cc in M:
            cols.append((dict(Ac), dict(Bc), dict(Cc)))
    
    # 2x2 x 2x2 -> 2x2 blocks using Strassen (rank 7 each)
    strassen(0, 0, 0, 0, 0, 0)   # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    strassen(0, 0, 0, 2, 0, 2)   # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    
    # A[0:2,2] (2x1) * B[2:3,0:2] (1x2) -> C[0:2,0:2] (2x2): rank 4
    for i in range(2):
        for k in range(2):
            cols.append(( {(i,2):1}, {(2,k):1}, {(i,k):1} ))
    
    # A[0:2,2] (2x1) * B[2:3,2:4] (1x2) -> C[0:2,2:4] (2x2): rank 4
    for i in range(2):
        for k in range(2):
            cols.append(( {(i,2):1}, {(2,2+k):1}, {(i,2+k):1} ))
    
    # A[2,0:2] (1x2) * B[0:2,0:2] (2x2) -> C[2,0:2] (1x2): rank 4
    for j in range(2):
        for k in range(2):
            cols.append(( {(2,j):1}, {(j,k):1}, {(2,k):1} ))
    
    # A[2,0:2] (1x2) * B[0:2,2:4] (2x2) -> C[2,2:4] (1x2): rank 4
    for j in range(2):
        for k in range(2):
            cols.append(( {(2,j):1}, {(j,2+k):1}, {(2,2+k):1} ))
    
    # A[2,2] (1x1) * B[2:3,0:2] (1x2) -> C[2,0:2] (1x2): rank 2
    for k in range(2):
        cols.append(( {(2,2):1}, {(2,k):1}, {(2,k):1} ))
    
    # A[2,2] (1x1) * B[2:3,2:4] (1x2) -> C[2,2:4] (1x2): rank 2
    for k in range(2):
        cols.append(( {(2,2):1}, {(2,2+k):1}, {(2,2+k):1} ))
    
    rank = len(cols)
    return _from_columns(cols, n, m, p, rank)

def _build_block_decomposition(n, m, p):
    """
    Build a block decomposition using Strassen for 2x2 blocks.
    For 3x3 * 3x4:
    - Split A into blocks: A[0:2,0:2], A[0:2,2:3], A[2:3,0:2], A[2:3,2:3]
    - Split B into blocks: B[0:2,0:2], B[0:2,2:4], B[2:3,0:2], B[2:3,2:4]
    - Use Strassen for 2x2 * 2x2 -> 2x2 multiplications
    - Use direct methods for remaining blocks
    """
    cols = []
    
    # Define Strassen for 2x2 * 2x2 -> 2x2
    def strassen_2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        # M1 = (A11 + A22)(B11 + B22)
        A1 = {(A_rows[0], A_cols[0]): 1, (A_rows[1], A_cols[1]): 1}
        B1 = {(B_rows[0], B_cols[0]): 1, (B_rows[1], B_cols[1]): 1}
        C1 = {(C_rows[0], C_cols[0]): 1, (C_rows[1], C_cols[1]): 1}
        cols.append((A1, B1, C1))
        
        # M2 = A21(B11 + B12)
        A2 = {(A_rows[1], A_cols[0]): 1, (A_rows[1], A_cols[1]): 1}
        B2 = {(B_rows[0], B_cols[0]): 1}
        C2 = {(C_rows[1], C_cols[0]): 1, (C_rows[1], C_cols[1]): -1}
        cols.append((A2, B2, C2))
        
        # M3 = (A11 + A12)B21
        A3 = {(A_rows[0], A_cols[0]): 1}
        B3 = {(B_rows[1], B_cols[0]): 1, (B_rows[1], B_cols[1]): -1}
        C3 = {(C_rows[0], C_cols[1]): 1, (C_rows[1], C_cols[1]): 1}
        cols.append((A3, B3, C3))
        
        # M4 = A22(B21 - B11)
        A4 = {(A_rows[1], A_cols[1]): 1}
        B4 = {(B_rows[0], B_cols[1]): 1, (B_rows[1], B_cols[0]): -1}
        C4 = {(C_rows[0], C_cols[0]): 1, (C_rows[1], C_cols[0]): 1}
        cols.append((A4, B4, C4))
        
        # M5 = (A11 - A21)(B11 + B21)
        A5 = {(A_rows[0], A_cols[0]): 1, (A_rows[0], A_cols[1]): 1}
        B5 = {(B_rows[1], B_cols[0]): 1}
        C5 = {(C_rows[0], C_cols[0]): -1, (C_rows[0], C_cols[1]): 1}
        cols.append((A5, B5, C5))
        
        # M6 = (A12 - A22)(B12 + B22)
        A6 = {(A_rows[0], A_cols[1]): 1, (A_rows[0], A_cols[0]): -1}
        B6 = {(B_rows[0], B_cols[0]): 1, (B_rows[1], B_cols[0]): 1}
        C6 = {(C_rows[1], C_cols[1]): 1}
        cols.append((A6, B6, C6))
        
        # M7 = (A21 + A22)(B12 - B22)
        A7 = {(A_rows[1], A_cols[0]): 1, (A_rows[1], A_cols[1]): -1}
        B7 = {(B_rows[1], B_cols[1]): 1, (B_rows[0], B_cols[1]): 1}
        C7 = {(C_rows[0], C_cols[0]): 1, (C_rows[0], C_cols[1]): -1}
        cols.append((A7, B7, C7))
    
    # Main 2x2 * 2x2 -> 2x2 block multiplications
    # C[0:2,0:2] += A[0:2,0:2] * B[0:2,0:2]
    strassen_2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # C[0:2,2:4] += A[0:2,0:2] * B[0:2,2:4]
    strassen_2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # C[0:2,0:2] += A[0:2,2:3] * B[2:3,0:2]
    # A[0:2,2:3] is 2x1, B[2:3,0:2] is 1x2, result is 2x2
    for i in range(2):
        for k in range(2):
            cols.append(({(i, 2): 1}, {(2, k): 1}, {(i, k): 1}))
    
    # C[0:2,2:4] += A[0:2,2:3] * B[2:3,2:4]
    # A[0:2,2:3] is 2x1, B[2:3,2:4] is 1x2, result is 2x2
    for i in range(2):
        for k in range(2):
            cols.append(({(i, 2): 1}, {(2, 2+k): 1}, {(i, 2+k): 1}))
    
    # C[2:3,0:2] += A[2:3,0:2] * B[0:2,0:2]
    # A[2:3,0:2] is 1x2, B[0:2,0:2] is 2x2, result is 1x2
    for j in range(2):
        for k in range(2):
            cols.append(({(2, j): 1}, {(j, k): 1}, {(2, k): 1}))
    
    # C[2:3,2:4] += A[2:3,0:2] * B[0:2,2:4]
    # A[2:3,0:2] is 1x2, B[0:2,2:4] is 2x2, result is 1x2
    for j in range(2):
        for k in range(2):
            cols.append(({(2, j): 1}, {(j, 2+k): 1}, {(2, 2+k): 1}))
    
    # C[2:3,0:2] += A[2:3,2:3] * B[2:3,0:2]
    # A[2:3,2:3] is 1x1, B[2:3,0:2] is 1x2, result is 1x2
    for k in range(2):
        cols.append(({(2, 2): 1}, {(2, k): 1}, {(2, k): 1}))
    
    # C[2:3,2:4] += A[2:3,2:3] * B[2:3,2:4]
    # A[2:3,2:3] is 1x1, B[2:3,2:4] is 1x2, result is 1x2
    for k in range(2):
        cols.append(({(2, 2): 1}, {(2, 2+k): 1}, {(2, 2+k): 1}))
    
    rank = len(cols)
    return _from_columns(cols, n, m, p, rank)

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 115.0
    
    # Try block decomposition
    U, V, W = _build_block_decomposition(n, m, p)
    rank = len(U[0])
    
    if _verify(U, V, W, n, m, p, rank) and (time.time() - t0) < DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    if time.time() - t0 < DEADLINE:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    return {"U": [[0]], "V": [[0]], "W": [[0]]}
