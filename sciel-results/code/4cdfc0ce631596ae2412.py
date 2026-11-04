"""Tensor decomposition of 5x5x5 matrix multiplication."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 110.0  # safety margin before 120s limit

    n = 5
    m = 5
    p = 5

    U_cols = []
    V_cols = []
    W_cols = []

    # Part A: Frame terms for (i,j) where i==4 or j==4, with each k
    # These are 9 pairs of (i,j) × 5 values of k = 45 terms
    for ii in range(n):
        for jj in range(m):
            if ii == 4 or jj == 4:
                for kk in range(p):
                    U_cols.append({ii * m + jj: 1})
                    V_cols.append({jj * p + kk: 1})
                    W_cols.append({kk * n + ii: 1})

    # Part B: 4x4x4 core via Kronecker product of Strassen (rank 49)
    # Define 2x2x2 Strassen decomposition (rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U2 = [
        [1, 0, 1, 0, 1, 0, 0], # M1, M2, M3, M4, M5, M6, M7
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ]
    
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    
    W2 = [
        [1, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0]
    ]
    
    # Construct 4x4 decomposition by Kronecker product
    # U4 = U2 ⊗ U2
    # Index mapping for 4x4:
    # i4, j4 in 0..3
    # i4 = 2*i2 + i2', j4 = 2*j2 + j2'
    # idx4 = i4 * 4 + j4
    # idx2 = i2 * 2 + j2
    
    for r1 in range(7):
        for r2 in range(7):
            u_col = {}
            v_col = {}
            w_col = {}
            
            # U4 entries
            for i2 in range(2):
                for j2 in range(2):
                    idx2_u = i2 * 2 + j2
                    u_val1 = U2[idx2_u][r1]
                    if u_val1 == 0: continue
                    
                    for i2_p in range(2):
                        for j2_p in range(2):
                            idx2_u_p = i2_p * 2 + j2_p
                            u_val2 = U2[idx2_u_p][r2]
                            if u_val2 == 0: continue
                            
                            u_val = u_val1 * u_val2
                            if u_val == 0: continue
                            
                            # Map to 4x4 index
                            i4 = 2 * i2 + i2_p
                            j4 = 2 * j2 + j2_p
                            idx4_u = i4 * 4 + j4
                            u_col[idx4_u] = u_col.get(idx4_u, 0) + u_val
                            
            # V4 entries
            for j2 in range(2):
                for k2 in range(2):
                    idx2_v = j2 * 2 + k2
                    v_val1 = V2[idx2_v][r1]
                    if v_val1 == 0: continue
                    
                    for j2_p in range(2):
                        for k2_p in range(2):
                            idx2_v_p = j2_p * 2 + k2_p
                            v_val2 = V2[idx2_v_p][r2]
                            if v_val2 == 0: continue
                            
                            v_val = v_val1 * v_val2
                            if v_val == 0: continue
                            
                            j4 = 2 * j2 + j2_p
                            k4 = 2 * k2 + k2_p
                            idx4_v = j4 * 4 + k4
                            v_col[idx4_v] = v_col.get(idx4_v, 0) + v_val
                            
            # W4 entries
            for k2 in range(2):
                for i2 in range(2):
                    idx2_w = k2 * 2 + i2
                    w_val1 = W2[idx2_w][r1]
                    if w_val1 == 0: continue
                    
                    for k2_p in range(2):
                        for i2_p in range(2):
                            idx2_w_p = k2_p * 2 + i2_p
                            w_val2 = W2[idx2_w_p][r2]
                            if w_val2 == 0: continue
                            
                            w_val = w_val1 * w_val2
                            if w_val == 0: continue
                            
                            k4 = 2 * k2 + k2_p
                            i4 = 2 * i2 + i2_p
                            idx4_w = k4 * 4 + i4
                            w_col[idx4_w] = w_col.get(idx4_w, 0) + w_val
                            
            # Embed 4x4 into 5x5
            # 4x4 indices 0..3 map to 5x5 indices 0..3
            # 5x5 index = i4 * 5 + j4
            U5_col = {}
            V5_col = {}
            W5_col = {}
            
            for idx4_u, val in u_col.items():
                if val != 0:
                    i4, j4 = divmod(idx4_u, 4)
                    idx5_u = i4 * 5 + j4
                    U5_col[idx5_u] = val
                    
            for idx4_v, val in v_col.items():
                if val != 0:
                    j4, k4 = divmod(idx4_v, 4)
                    idx5_v = j4 * 5 + k4
                    V5_col[idx5_v] = val
                    
            for idx4_w, val in w_col.items():
                if val != 0:
                    k4, i4 = divmod(idx4_w, 4)
                    idx5_w = k4 * 5 + i4
                    W5_col[idx5_w] = val
                    
            U_cols.append(U5_col)
            V_cols.append(V5_col)
            W_cols.append(W5_col)

    rank = len(U_cols)

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify with random integer tests using NumPy for speed
    try:
        import numpy as np
        rng = np.random.RandomState(12345)
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        for trial in range(500):
            if time.time() > tmax:
                break
            A = rng.randint(0, 6, size=(n, m))
            B = rng.randint(0, 6, size=(m, p))
            C_true = A @ B
            C_test = np.zeros((n, p), dtype=np.int64)
            for r in range(rank):
                u_col = U_np[:, r]
                v_col = V_np[:, r]
                w_col = W_np[:, r]
                u_mat = u_col.reshape(n, m)
                v_mat = v_col.reshape(m, p)
                w_mat = w_col.reshape(p, n)
                temp = u_mat @ v_mat
                C_test += temp * w_mat.T
            if not np.array_equal(C_test, C_true):
                return _schoolbook()
    except ImportError:
        import random
        rng = random.Random(12345)
        for trial in range(200):
            if time.time() > tmax:
                break
            A = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(0, 5) for _ in range(p)] for _ in range(m)]
            C_true = [[sum(A[ii][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)] for ii in range(n)]
            C_test = [[0] * p for _ in range(n)]
            for ii in range(n):
                for kk in range(p):
                    s = 0
                    for r in range(rank):
                        for jj in range(m):
                            s += U[ii * m + jj][r] * V[jj * p + kk][r] * W[kk * n + ii][r]
                    C_test[ii][kk] = s
            if C_test != C_true:
                return _schoolbook()

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
