import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy: 6x6 = 3x3 blocks of 2x2
    # Use schoolbook for 3x3 block structure (9 block products)
    # Use Strassen (7 terms) for each 2x2 block multiplication
    # Total rank = 9 * 7 = 63
    
    B = 3  # number of blocks per dimension
    S = 2  # size of each block
    
    total_rank = B * B * B * 7  # 9 * 7 = 63
    
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    # Strassen 2x2 decomposition
    # u indexed by (li, lk) for A[li, lk]
    # v indexed by (lk, lj) for B[lk, lj]
    # w indexed by (li, lj) for C[li, lj]
    # Vector index: first*2 + second
    
    strassen_2x2 = [
        # P1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        {'u': [1,0,0,1], 'v': [1,0,0,1], 'w': [1,0,0,1]},
        # P2 = (A21+A22)B11 -> C21 += 1, C22 -= 1
        {'u': [0,0,1,1], 'v': [1,0,0,0], 'w': [0,0,1,-1]},
        # P3 = A11(B12-B22) -> C12 += 1, C22 += 1
        {'u': [1,0,0,0], 'v': [0,1,0,-1], 'w': [0,1,0,1]},
        # P4 = A22(B21-B11) -> C11 += 1, C21 += 1
        {'u': [0,0,0,1], 'v': [0,0,1,-1], 'w': [1,0,1,0]},
        # P5 = (A11+A12)B22 -> C12 += 1, C11 -= 1
        {'u': [1,1,0,0], 'v': [0,0,0,1], 'w': [0,1,-1,0]},
        # P6 = (A21-A11)(B11+B12) -> C22 += 1
        {'u': [0,0,1,-1], 'v': [1,1,0,0], 'w': [0,0,0,1]},
        # P7 = (A12-A22)(B21+B22) -> C11 += 1
        {'u': [0,1,0,-1], 'v': [0,0,1,1], 'w': [1,0,0,0]},
    ]
    
    # Construct the full 6x6 decomposition
    # A[bi*S+li, bj*S+lj] where bi,bj are block indices, li,lj are local indices
    # B[bk*S+lk, bj*S+lj]
    # C[bi*S+li, bj*S+lj] at W index (bj*S+lj)*6 + (bi*S+li)
    
    col_offset = 0
    for bi in range(B):
        for bj in range(B):
            for bk in range(B):
                for t in range(7):
                    st = strassen_2x2[t]
                    
                    # U: A[bi*S+li, bj*S+lj]
                    for li in range(S):
                        for lj in range(S):
                            u_val = st['u'][li*S+lj]
                            if u_val != 0:
                                u_idx = (bi*S + li) * 6 + (bj*S + lj)
                                U6[u_idx, col_offset] += u_val
                    
                    # V: B[bk*S+lk, bj*S+lj]
                    for lk in range(S):
                        for lj in range(S):
                            v_val = st['v'][lk*S+lj]
                            if v_val != 0:
                                v_idx = (bk*S + lk) * 6 + (bj*S + lj)
                                V6[v_idx, col_offset] += v_val
                    
                    # W: C[bi*S+li, bj*S+lj] at W index (bj*S+lj)*6 + (bi*S+li)
                    for li in range(S):
                        for lj in range(S):
                            w_val = st['w'][li*S+lj]
                            if w_val != 0:
                                w_idx = (bj*S + lj) * 6 + (bi*S + li)
                                W6[w_idx, col_offset] += w_val
                    
                    col_offset += 1
    
    # Time guard
    if time.time() - start_time > 120:
        # Fallback to schoolbook
        rank = 216
        U_f = np.zeros((36, rank))
        V_f = np.zeros((36, rank))
        W_f = np.zeros((36, rank))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    idx = i*6 + j
                    jk = j*6 + k
                    ki = k*6 + i
                    col = i*36 + j*6 + k
                    U_f[idx, col] = 1.0
                    V_f[jk, col] = 1.0
                    W_f[ki, col] = 1.0
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    # Verify
    U3 = U6.reshape(6, 6, -1)
    V3 = V6.reshape(6, 6, -1)
    W3 = W6.reshape(6, 6, -1)
    
    try:
        reconstructed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        target = np.ones((6, 6, 6))
        if np.allclose(reconstructed, target, atol=1e-9):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        else:
            # Fallback to schoolbook
            rank = 216
            U_f = np.zeros((36, rank))
            V_f = np.zeros((36, rank))
            W_f = np.zeros((36, rank))
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        idx = i*6 + j
                        jk = j*6 + k
                        ki = k*6 + i
                        col = i*36 + j*6 + k
                        U_f[idx, col] = 1.0
                        V_f[jk, col] = 1.0
                        W_f[ki, col] = 1.0
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    except:
        # Fallback to schoolbook
        rank = 216
        U_f = np.zeros((36, rank))
        V_f = np.zeros((36, rank))
        W_f = np.zeros((36, rank))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    idx = i*6 + j
                    jk = j*6 + k
                    ki = k*6 + i
                    col = i*36 + j*6 + k
                    U_f[idx, col] = 1.0
                    V_f[jk, col] = 1.0
                    W_f[ki, col] = 1.0
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
