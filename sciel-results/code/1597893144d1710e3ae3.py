import numpy as np
import time

def _strassen_2x2x2():
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, -1]
    ], dtype=float)
    return U, V, W

def _schoolbook_2x2x1():
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    for i in range(2):
        for j in range(2):
            r = i * 2 + j
            U[i * 2 + j, r] = 1
            V[j, r] = 1
            W[i, r] = 1
    return U, V, W

def _2x2x3_decomp():
    U_str, V_str, W_str = _strassen_2x2x2()
    U_sch, V_sch, W_sch = _schoolbook_2x2x1()
    U = np.hstack([U_str, U_sch])
    V = np.zeros((6, 11))
    V[0:4, 0:7] = V_str
    V[4:6, 7:11] = V_sch
    W = np.zeros((6, 11))
    W[0:4, 0:7] = W_str
    W[4:6, 7:11] = W_sch
    return U, V, W

def _4x4x3_decomp():
    U_223, V_223, W_223 = _2x2x3_decomp()
    U = np.zeros((16, 44))
    V = np.zeros((12, 44))
    W = np.zeros((12, 44))
    for b in range(4):
        u_offset = b * 11
        v_offset = b * 11
        w_offset = b * 11
        i1 = b // 2
        j1 = b % 2
        j_b = j1
        i_c = i1
        for i2 in range(2):
            for j2 in range(2):
                i = i1 * 2 + i2
                j = j1 * 2 + j2
                U[i * 4 + j, u_offset:u_offset+11] = U_223[i2 * 2 + j2, :]
        for k in range(3):
            for j2 in range(2):
                j = j_b * 2 + j2
                v_row = j * 3 + k
                v_223_row = k * 2 + j2
                V[v_row, v_offset:v_offset+11] = V_223[v_223_row, :]
        for k in range(3):
            for i2 in range(2):
                i = i_c * 2 + i2
                w_row = k * 4 + i
                w_223_row = k * 2 + i2
                W[w_row, w_offset:w_offset+11] = W_223[w_223_row, :]
    return U, V, W

def _4x4x4_decomp():
    U_str, V_str, W_str = _strassen_2x2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j, r] = U_str[i1 * 2 + j1, r1] * U_str[i2 * 2 + j2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k, r] = V_str[j1 * 2 + k1, r1] * V_str[j2 * 2 + k2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i, r] = W_str[k1 * 2 + i1, r1] * W_str[k2 * 2 + i2, r2]
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_search_4x4x3(max_rank, time_limit, seed=42):
    """Try to find a rank decomposition for 4x4x3."""
    n, m, p = 4, 4, 3
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    for rank in range(40, max_rank + 1):
        if time.time() - start_time > time_limit:
            return None
        for trial in range(20):
            if time.time() - start_time > time_limit:
                return None
            # Initialize with random sparse matrices
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for r in range(rank):
                # Randomly assign non-zero entries
                for i in range(n*m):
                    if rng.random() < 0.3:
                        U[i, r] = rng.choice([-1, 1])
                for j in range(m*p):
                    if rng.random() < 0.3:
                        V[j, r] = rng.choice([-1, 1])
                for k in range(p*n):
                    if rng.random() < 0.3:
                        W[k, r] = rng.choice([-1, 1])
            
            if _verify_decomp(U, V, W, n, m, p):
                return U, V, W
    return None

def construct():
    n, m, p = 4, 4, 7
    
    U444, V444, W444 = _4x4x4_decomp()
    U443, V443, W443 = _4x4x3_decomp()
    
    # Expand V444 and W444 to full size (28, 49)
    V444_full = np.zeros((28, 49))
    W444_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V444_full[j * 7 + k, :] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W444_full[k * 4 + i, :] = W444[k * 4 + i, :]
            
    # Expand V443 and W443 to full size (28, 44)
    V443_full = np.zeros((28, 44))
    W443_full = np.zeros((28, 44))
    for j in range(4):
        for k in range(3):
            V443_full[j * 7 + (4 + k), :] = V443[j * 3 + k, :]
    for k in range(3):
        for i in range(4):
            W443_full[(4 + k) * 4 + i, :] = W443[k * 4 + i, :]
            
    U = np.hstack([U444, U443])
    V = np.hstack([V444_full, V443_full])
    W = np.hstack([W444_full, W443_full])
    
    # Verify
    if not _verify_decomp(U, V, W, n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
                    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
