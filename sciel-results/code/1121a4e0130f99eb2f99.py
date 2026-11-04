import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Strassen decomposition for 2x2x2: rank 7
    # Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # This corresponds to the tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Note: The problem statement says W[k*n+i, r]. For n=2, this is W[k*2+i, r].
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # Standard Strassen 7 multiplications:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B10)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M2 - M3 - M4
    # C01 = M1 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Indices for U (A part): row = i*2+j
    # Indices for V (B part): row = j*2+k
    # Indices for W (C part): row = k*2+i
    
    # M0
    # A: A00, A11 -> U[0,0]=1, U[3,0]=1
    # B: B00, B11 -> V[0,0]=1, V[3,0]=1
    # C: C00, C11 -> W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1
    # A: A10, A11 -> U[2,1]=1, U[3,1]=1
    # B: B00 -> V[0,1]=1
    # C: C01, C10 -> W[1,1]=1 (k=0,i=1), W[2,1]=1 (k=1,i=0)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[2, 1] = 1

    # M2
    # A: A00 -> U[0,2]=1
    # B: B10, B11 -> V[1,2]=1, V[3,2]=-1
    # C: C00, C11 -> W[0,2]=1, W[3,2]=1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[0, 2] = 1; W2[3, 2] = 1

    # M3
    # A: A11 -> U[3,3]=1
    # B: B01, B10 -> V[2,3]=1, V[1,3]=-1
    # C: C00, C10 -> W[0,3]=-1 (k=0,i=0), W[2,3]=1 (k=1,i=0)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[1, 3] = -1
    W2[0, 3] = -1; W2[2, 3] = 1

    # M4
    # A: A00, A01 -> U[0,4]=1, U[1,4]=1
    # B: B11 -> V[3,4]=1
    # C: C00, C01 -> W[0,4]=-1 (k=0,i=0), W[1,4]=1 (k=0,i=1)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5
    # A: A10, A00 -> U[2,5]=1, U[0,5]=-1
    # B: B00, B01 -> V[0,5]=1, V[2,5]=1
    # C: C11 -> W[3,5]=1 (k=1,i=1)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[3, 5] = 1

    # M6
    # A: A01, A11 -> U[1,6]=1, U[3,6]=-1
    # B: B10, B11 -> V[1,6]=1, V[3,6]=1
    # C: C11 -> W[3,6]=1 (k=1,i=1)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    W2[3, 6] = 1

    # Verify 2x2 Strassen
    valid2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if s != 1:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break

    if not valid2:
        # Fallback to schoolbook rank-64
        n, m, p = 4, 4, 4
        rank = 64
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Construct 4x4x4 via Kronecker product of two 2x2 Strassen decompositions
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # U row index: i*4+j. Split i, j into (i1,i2), (j1,j2)
    # U[i,j] = kron(U2[i1,j1], U2[i2,j2])
    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        for j in range(m):
            j1 = j // 2
            j2 = j % 2
            u_row = i * m + j
            U[u_row, :] = np.kron(U2[i1 * 2 + j1, :], U2[i2 * 2 + j2, :])

    # V row index: j*4+k. Split j, k into (j1,j2), (k1,k2)
    # V[j,k] = kron(V2[j1,k1], V2[j2,k2])
    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 2
            k2 = k % 2
            v_row = j * p + k
            V[v_row, :] = np.kron(V2[j1 * 2 + k1, :], V2[j2 * 2 + k2, :])

    # W row index: k*4+i. Split k, i into (k1,k2), (i1,i2)
    # W[k,i] = kron(W2[k1,i1], W2[k2,i2])
    for k in range(p):
        k1 = k // 2
        k2 = k % 2
        for i in range(n):
            i1 = i // 2
            i2 = i % 2
            w_row = k * n + i
            W[w_row, :] = np.kron(W2[k1 * 2 + i1, :], W2[k2 * 2 + i2, :])

    # Verify the 4x4x4 decomposition
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
                if s != 1:
                    valid4 = False
                    break
            if not valid4:
                break
        if not valid4:
            break

    if not valid4:
        # Fallback to schoolbook rank-64
        n, m, p = 4, 4, 4
        rank = 64
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
