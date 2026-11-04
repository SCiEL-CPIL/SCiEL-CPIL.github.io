import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin under 150s

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Strassen 2x2x2 factors under convention T[(i,j),(j',k),(k',i')] = delta_{j,j'}*delta_{i,i'}
    Us = np.array([
        [1, 1, 0, 0, 0, -1, 1],   # (i,j)=(0,0)
        [0, 1, 0, 0, 1, 0, 0],    # (0,1)
        [0, 0, 1, 0, 0, 1, 0],    # (1,0)
        [0, 0, 0, 1, 1, 0, -1],   # (1,1)
    ], dtype=int)
    Vs = np.array([
        [0, 0, 1, 1, 1, 0, 1],    # (j',k)=(0,0)
        [0, 0, 0, 0, 0, 1, 1],    # (0,1)
        [-1, 1, -1, 0, 0, 0, 0],  # (1,0)
        [1, 0, 0, 1, 0, 1, 0],    # (1,1)
    ], dtype=int)
    Ws = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # (k',i')=(0,0)
        [0, 0, 1, 0, 1, 0, 0],    # (0,1)
        [0, 1, 0, 1, 0, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ], dtype=int)

    # Verify Strassen 2x2x2: sum_c Us[i,c]*Vs[j,c]*Ws[k,c] == delta_{i,k} (with summed j)
    # The 2x2x2 tensor: T[i, j, k] such that C_{k,i} = sum_j A_{i,j} B_{j,k};
    # tensor T2[(i,j),(j',k),(k',i')] = delta_{j,j'} delta_{i,i'}; rank form sum_c U[(i,j),c]V[(j',k),c]W[(k',i'),c]
    U2 = Us.reshape(2, 2, 7)  # [i, j, c]
    V2 = Vs.reshape(2, 2, 7)  # [j', k, c]
    W2 = Ws.reshape(2, 2, 7)  # [k', i', c]
    ok = True
    for i in range(2):
        for j in range(2):
            for jp in range(2):
                for k in range(2):
                    for kp in range(2):
                        for ip in range(2):
                            lhs = 1 if (jp == j and ip == i) else 0
                            rhs = int(np.sum(U2[i, j] * V2[jp, k] * W2[kp, ip]))
                            if lhs != rhs:
                                ok = False
    strassen_ok = ok

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def to_list(M):
        return [[int(M[r, c]) for c in range(M.shape[1])] for r in range(M.shape[0])]

    if not strassen_ok or time.time() > deadline - 30:
        U, V, W = schoolbook()
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Kronecker: 4x4x4 part (first 4 columns of C, i.e. k' in 0..3)
    # U4[(i,j), c] = Us[i%2, c%7] * Us[i//2, c//7]; row index (i,j) in 0..15, c in 0..48
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)  # (j,k) for k in 0..3 -> 4x4 = 16 rows
    W4 = np.zeros((16, 49), dtype=int)  # (k',i') for k' in 0..3 -> 16 rows
    for i in range(2):
        for j in range(2):
            for jp in range(2):
                for k in range(2):
                    for kp in range(2):
                        for ip in range(2):
                            c = (i * 2 + jp) * 7 + (j * 2 + k)
                            U4[(i * 2 + jp) * 4 + (j * 2 + k), c] = Us[i, i * 2 + jp] * Us[j, j * 2 + k]
                            # careful: U4 row = (I, J) where I=i*2+jp, J=j*2+k ; U4[I,J,c]=Us[i,c1]*Us[j,c2]
    # Redo cleanly
    U4 = np.zeros((16, 49), dtype=int)
    for I in range(4):
        for J in range(4):
            i1, i2 = I // 2, I % 2
            j1, j2 = J // 2, J % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    U4[I * 4 + J, c] = Us[i1, c1] * Us[j2, c2]
    # U row index is (i, j) -> I = i*2 + ... no. For 4x4, A has rows i in 0..3, cols j in 0..3.
    # U row = (i, j) with i in 0..3, j in 0..3 -> index i*4 + j (n*m = 16).
    # We split i = i1*2 + i2, j = j1*2 + j2. Kronecker: U[(i,j), (c1,c2)] = U2[(i1,j1),c1] * U2[(i2,j2),c2]
    # where U2 is the 2x2x2 U factor indexed by (i', j').
    U4 = np.zeros((16, 49), dtype=int)
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    U4[i * 4 + j, c] = Us[i1 * 2 + j1, c1] * Us[i2 * 2 + j2, c2]
    V4 = np.zeros((16, 49), dtype=int)  # (j, k) j in 0..3, k in 0..3 -> j*4+k
    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    V4[j * 4 + k, c] = Vs[j1 * 2 + k1, c1] * Vs[j2 * 2 + k2, c2]
    W4 = np.zeros((16, 49), dtype=int)  # (k', i') k' in 0..3, i' in 0..3 -> k'*4+i'
    for kp in range(4):
        for ip in range(4):
            k1, k2 = kp // 2, kp % 2
            i1, i2 = ip // 2, ip % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    W4[kp * 4 + ip, c] = Ws[k1 * 2 + i1, c1] * Ws[k2 * 2 + i2, c2]

    # Schoolbook for 5th column of C (k' = 4): rank 16 terms
    U5 = np.zeros((16, 16), dtype=int)
    V5 = np.zeros((20, 16), dtype=int)
    W5 = np.zeros((20, 16), dtype=int)
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U5[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W5[4 * 4 + i, col] = 1  # k'*n + i' = 4*4 + i

    # Assemble
    rank = 49 + 16
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((20, rank), dtype=int)
    W = np.zeros((20, rank), dtype=int)
    U[:, 0:49] = U4
    V[:16, 0:49] = V4
    W[:16, 0:49] = W4
    U[:, 49:65] = U5
    V[:, 49:65] = V5
    W[:, 49:65] = W5

    # Verify full tensor identity numerically
    # T[(i,j),(j',k),(k',i')] = delta_{j,j'} delta_{i,i'}
    # sum_c U[(i,j),c] V[(j',k),c] W[(k',i'),c]
    Uv, Vv, Wv = U.astype(float), V.astype(float), W.astype(float)
    maxerr = 0.0
    for i in range(4):
        for j in range(4):
            for jp in range(4):
                for k in range(5):
                    for kp in range(5):
                        for ip in range(4):
                            lhs = 1 if (jp == j and ip == i) else 0
                            rhs = float(np.dot(Uv[i * 4 + j], Vv[jp * 5 + k] * Wv[kp * 4 + ip]))
                            e = abs(lhs - rhs)
                            if e > maxerr:
                                maxerr = e
    if maxerr > 1e-9:
        U, V, W = schoolbook()
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
