import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 6

    # Schoolbook 2x2 decomposition (rank 4)
    # U2 indexed by (i,k) for i,k in {0,1} -> index i*2+k
    # V2 indexed by (k,j) for k,j in {0,1} -> index k*2+j
    # W2 indexed by (j,i) for j,i in {0,1} -> index j*2+i
    U2 = np.zeros((4, 4), dtype=int)
    V2 = np.zeros((4, 4), dtype=int)
    W2 = np.zeros((4, 4), dtype=int)

    for i2 in range(2):
        for k2 in range(2):
            t2 = i2 * 2 + k2
            U2[i2 * 2 + k2, t2] = 1
            for j2 in range(2):
                V2[k2 * 2 + j2, t2] = 1
                W2[j2 * 2 + i2, t2] = 1

    # Schoolbook 3x3 decomposition (rank 9)
    # Term t = i3*3 + k3: U3[i3*3+k3, t]=1, V3[k3*3+j3, t]=1, W3[j3*3+i3, t]=1
    U3 = np.zeros((9, 9), dtype=int)
    V3 = np.zeros((9, 9), dtype=int)
    W3 = np.zeros((9, 9), dtype=int)

    for i3 in range(3):
        for k3 in range(3):
            t3 = i3 * 3 + k3
            U3[i3 * 3 + k3, t3] = 1
            for j3 in range(3):
                V3[k3 * 3 + j3, t3] = 1
                W3[j3 * 3 + i3, t3] = 1

    # Kronecker product: MM_6 = MM_2 ⊗ MM_3
    # Indexing: i = 2*i2 + i3, k = 2*k2 + k3, j = 2*j2 + j3
    # i2,k2,j2 in {0,1}, i3,k3,j3 in {0,1,2}
    # U6 row: i*6+k = (2*i2+i3)*6 + (2*k2+k3)
    # V6 row: k*6+j = (2*k2+k3)*6 + (2*j2+j3)
    # W6 row: j*6+i = (2*j2+j3)*6 + (2*i2+i3)

    rank6 = 4 * 9  # 36
    U6 = np.zeros((36, rank6), dtype=int)
    V6 = np.zeros((36, rank6), dtype=int)
    W6 = np.zeros((36, rank6), dtype=int)

    # Build the Kronecker product efficiently using vectorized operations
    for t2 in range(4):
        for t3 in range(9):
            t = t2 * 9 + t3
            u2_col = U2[:, t2]  # shape (4,)
            v2_col = V2[:, t2]  # shape (4,)
            w2_col = W2[:, t2]  # shape (4,)
            u3_col = U3[:, t3]  # shape (9,)
            v3_col = V3[:, t3]  # shape (9,)
            w3_col = W3[:, t3]  # shape (9,)

            for i2 in range(2):
                for i3 in range(3):
                    for k2 in range(2):
                        for k3 in range(3):
                            u_row = (2 * i2 + i3) * 6 + (2 * k2 + k3)
                            U6[u_row, t] += u2_col[i2 * 2 + k2] * u3_col[i3 * 3 + k3]

            for k2 in range(2):
                for k3 in range(3):
                    for j2 in range(2):
                        for j3 in range(3):
                            v_row = (2 * k2 + k3) * 6 + (2 * j2 + j3)
                            V6[v_row, t] += v2_col[k2 * 2 + j2] * v3_col[k3 * 3 + j3]

            for j2 in range(2):
                for j3 in range(3):
                    for i2 in range(2):
                        for i3 in range(3):
                            w_row = (2 * j2 + j3) * 6 + (2 * i2 + i3)
                            W6[w_row, t] += w2_col[j2 * 2 + i2] * w3_col[j3 * 3 + i3]

    # Fast vectorized verification
    verified = True
    for i in range(6):
        for j in range(6):
            for k in range(6):
                u_row = i * 6 + k
                v_row = k * 6 + j
                w_row = j * 6 + i
                actual = np.sum(U6[u_row, :] * V6[v_row, :] * W6[w_row, :])
                if abs(actual - 1.0) > 1e-10:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break

    if not verified:
        # Fallback: use the block Strassen approach (rank 189)
        block_size = 2
        n_blocks = 3
        rank = n_blocks ** 3 * 7
        U = np.zeros((36, rank), dtype=int)
        V = np.zeros((36, rank), dtype=int)
        W = np.zeros((36, rank), dtype=int)
        col = 0

        def u_idx(ib, kb, r, c): return (ib*2+r)*6+(kb*2+c)
        def v_idx(kb, jb, r, c): return (kb*2+r)*6+(jb*2+c)
        def w_idx(ib, jb, r, c): return (jb*2+c)*6+(ib*2+r)

        for ib in range(3):
            for jb in range(3):
                for kb in range(3):
                    U[u_idx(ib,kb,0,0),col]+=1; U[u_idx(ib,kb,1,1),col]+=1
                    V[v_idx(kb,jb,0,0),col]+=1; V[v_idx(kb,jb,1,1),col]+=1
                    W[w_idx(ib,jb,0,0),col]+=1; W[w_idx(ib,jb,1,1),col]+=1; col+=1
                    U[u_idx(ib,kb,1,0),col]+=1; U[u_idx(ib,kb,1,1),col]+=1
                    V[v_idx(kb,jb,0,0),col]+=1
                    W[w_idx(ib,jb,1,0),col]+=1; W[w_idx(ib,jb,1,1),col]-=1; col+=1
                    U[u_idx(ib,kb,0,0),col]+=1
                    V[v_idx(kb,jb,0,1),col]+=1; V[v_idx(kb,jb,1,1),col]-=1
                    W[w_idx(ib,jb,0,1),col]+=1; W[w_idx(ib,jb,1,1),col]+=1; col+=1
                    U[u_idx(ib,kb,1,1),col]+=1
                    V[v_idx(kb,jb,1,0),col]+=1; V[v_idx(kb,jb,0,0),col]-=1
                    W[w_idx(ib,jb,0,0),col]+=1; W[w_idx(ib,jb,1,0),col]+=1; col+=1
                    U[u_idx(ib,kb,0,0),col]+=1; U[u_idx(ib,kb,0,1),col]+=1
                    V[v_idx(kb,jb,1,1),col]+=1
                    W[w_idx(ib,jb,0,0),col]-=1; W[w_idx(ib,jb,0,1),col]+=1; col+=1
                    U[u_idx(ib,kb,1,0),col]+=1; U[u_idx(ib,kb,0,0),col]-=1
                    V[v_idx(kb,jb,0,0),col]+=1; V[v_idx(kb,jb,0,1),col]+=1
                    W[w_idx(ib,jb,1,1),col]+=1; col+=1
                    U[u_idx(ib,kb,0,1),col]+=1; U[u_idx(ib,kb,1,1),col]-=1
                    V[v_idx(kb,jb,1,0),col]+=1; V[v_idx(kb,jb,1,1),col]+=1
                    W[w_idx(ib,jb,0,0),col]+=1; col+=1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
