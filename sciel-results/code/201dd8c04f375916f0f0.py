import time
import numpy as np

def construct():
    deadline = time.time() + 140
    
    n, m, p = 4, 4, 5
    
    # Strassen 2x2x2 factors
    # U: (i,j) -> c
    # V: (j',k) -> c
    # W: (k',i') -> c
    Us = np.array([
        [1, 1, 0, 0, 0, -1, 1],   # (0,0)
        [0, 1, 0, 0, 1, 0, 0],    # (0,1)
        [0, 0, 1, 0, 0, 1, 0],    # (1,0)
        [0, 0, 0, 1, 1, 0, -1],   # (1,1)
    ], dtype=int)
    Vs = np.array([
        [0, 0, 1, 1, 1, 0, 1],    # (0,0)
        [0, 0, 0, 0, 0, 1, 1],    # (0,1)
        [-1, 1, -1, 0, 0, 0, 0],  # (1,0)
        [1, 0, 0, 1, 0, 1, 0],    # (1,1)
    ], dtype=int)
    Ws = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 0, 1, 0, 1, 0, 0],    # (0,1)
        [0, 1, 0, 1, 0, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ], dtype=int)

    # 4x4x4 part (columns 0-3 of B and C)
    # Rank 49
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i in range(4):
        i1, i2 = i // 2, i % 2
        for j in range(4):
            j1, j2 = j // 2, j % 2
            u_idx = i * 4 + j
            for c1 in range(7):
                for c2 in range(7):
                    c_idx = c1 * 7 + c2
                    U4[u_idx, c_idx] = Us[i1 * 2 + j1, c1] * Us[i2 * 2 + j2, c2]
                    
    for j in range(4):
        j1, j2 = j // 2, j % 2
        for k in range(4):
            k1, k2 = k // 2, k % 2
            v_idx = j * 4 + k
            for c1 in range(7):
                for c2 in range(7):
                    c_idx = c1 * 7 + c2
                    V4[v_idx, c_idx] = Vs[j1 * 2 + k1, c1] * Vs[j2 * 2 + k2, c2]
                    
    for kp in range(4):
        k1, k2 = kp // 2, kp % 2
        for ip in range(4):
            i1, i2 = ip // 2, ip % 2
            w_idx = kp * 4 + ip
            for c1 in range(7):
                for c2 in range(7):
                    c_idx = c1 * 7 + c2
                    W4[w_idx, c_idx] = Ws[k1 * 2 + i1, c1] * Ws[k2 * 2 + i2, c2]

    # 4x4x1 part (column 4 of B and C)
    # Rank 16
    U5 = np.zeros((16, 16), dtype=int)
    V5 = np.zeros((20, 16), dtype=int)
    W5 = np.zeros((20, 16), dtype=int)
    
    col = 0
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 5 + 4  # j in 0..3, k=4
            w_idx = 4 * 4 + i  # k'=4, i' in 0..3
            U5[u_idx, col] = 1
            V5[v_idx, col] = 1
            W5[w_idx, col] = 1
            col += 1

    # Assemble
    total_rank = 49 + 16
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((20, total_rank), dtype=int)
    W = np.zeros((20, total_rank), dtype=int)
    
    U[:, 0:49] = U4
    V[:16, 0:49] = V4
    W[:16, 0:49] = W4
    
    U[:, 49:65] = U5
    V[:, 49:65] = V5
    W[:, 49:65] = W5

    # Verification
    # T[(i,j),(j',k),(k',i')] = delta_{j,j'} delta_{i,i'}
    # sum_c U[(i,j),c] V[(j',k),c] W[(k',i'),c]
    
    # Convert to float for dot product
    Uf = U.astype(float)
    Vf = V.astype(float)
    Wf = W.astype(float)
    
    max_err = 0.0
    # Iterate over all indices
    for i in range(4):
        for j in range(4):
            u_row = Uf[i * 4 + j]
            for jp in range(4):
                for k in range(5):
                    v_row = Vf[jp * 5 + k]
                    # Compute dot product with W rows
                    # We need sum_c U_c V_c W_c for each (k', i')
                    # This is equivalent to (U * V) dot W_col? No.
                    # It is sum_c (U_c * V_c) * W_c.
                    # Let P = U * V (element-wise). Then result for (k', i') is P dot W[k', i'].
                    prod = u_row * v_row
                    for kp in range(5):
                        for ip in range(4):
                            w_row = Wf[kp * 4 + ip]
                            val = np.dot(prod, w_row)
                            expected = 1.0 if (jp == j and ip == i) else 0.0
                            err = abs(val - expected)
                            if err > max_err:
                                max_err = err
                                if max_err > 1e-9:
                                    # Fallback to schoolbook if error found
                                    rank = 80
                                    U_s = np.zeros((16, rank), dtype=int)
                                    V_s = np.zeros((20, rank), dtype=int)
                                    W_s = np.zeros((20, rank), dtype=int)
                                    c = 0
                                    for ii in range(4):
                                        for jj in range(4):
                                            for kk in range(5):
                                                U_s[ii * 4 + jj, c] = 1
                                                V_s[jj * 5 + kk, c] = 1
                                                W_s[kk * 4 + ii, c] =1
                                                c += 1
                                    return {
                                        "U": [[int(x) for x in row] for row in U_s],
                                        "V": [[int(x) for x in row] for row in V_s],
                                        "W": [[int(x) for x in row] for row in W_s]
                                    }
    
    if time.time() > deadline:
         # This should not happen, but safety check
         pass

    return {
        "U": [[int(x) for x in row] for row in U],
        "V": [[int(x) for x in row] for row in V],
        "W": [[int(x) for x in row] for row in W]
    }
