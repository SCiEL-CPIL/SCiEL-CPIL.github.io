import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    # Rank 56: 8 blocks of Strassen (2x2x2), each rank 7
    # A: 2x4, split into 2 column blocks of width 2
    # B: 4x8, split into 2 row blocks x 4 col blocks, each 2x2
    # C: 2x8, split into 4 col blocks of width 2
    # Total: 2 * 4 * 7 = 56
    
    rank = 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # For each (u_block, v_block), we have 7 Strassen terms
    # u_block in {0, 1}: which 2 columns of A
    # v_block in {0, 1, 2, 3}: which 2 columns of B (and rows of B are 2*u_block, 2*u_block+1)
    
    col = 0
    for u in range(2):
        for v in range(4):
            base = 7 * (4 * u + v)
            
            # A indices: A[0, 2u], A[0, 2u+1], A[1, 2u], A[1, 2u+1]
            a00 = 0 * m + 2 * u
            a01 = 0 * m + 2 * u + 1
            a10 = 1 * m + 2 * u
            a11 = 1 * m + 2 * u + 1
            
            # B indices: B[2u, 2v], B[2u, 2v+1], B[2u+1, 2v], B[2u+1, 2v+1]
            b00 = (2 * u) * p + 2 * v
            b01 = (2 * u) * p + 2 * v + 1
            b10 = (2 * u + 1) * p + 2 * v
            b11 = (2 * u + 1) * p + 2 * v + 1
            
            # C/W indices: C[0, 2v], C[0, 2v+1], C[1, 2v], C[1, 2v+1]
            # W index for C[i, k] is k * n + i
            c00 = (2 * v) * n + 0      # C[0, 2v]
            c01 = (2 * v + 1) * n + 0  # C[0, 2v+1]
            c10 = (2 * v) * n + 1      # C[1, 2v]
            c11 = (2 * v + 1) * n + 1  # C[1, 2v+1]
            
            # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            U[a00, base] += 1
            U[a11, base] += 1
            V[b00, base] += 1
            V[b11, base] += 1
            W[c00, base] += 1
            W[c11, base] += 1
            
            # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
            U[a10, base + 1] += 1
            U[a11, base + 1] += 1
            V[b00, base + 1] += 1
            W[c10, base + 1] += 1
            W[c11, base + 1] -= 1
            
            # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
            U[a00, base + 2] += 1
            V[b01, base + 2] += 1
            V[b11, base + 2] -= 1
            W[c01, base + 2] += 1
            W[c11, base + 2] += 1
            
            # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
            U[a11, base + 3] += 1
            V[b10, base + 3] += 1
            V[b00, base + 3] -= 1
            W[c00, base + 3] += 1
            W[c10, base + 3] += 1
            
            # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
            U[a00, base + 4] += 1
            U[a01, base + 4] += 1
            V[b11, base + 4] += 1
            W[c00, base + 4] -= 1
            W[c01, base + 4] += 1
            
            # M6 = (A10-A00)(B00+B01) -> C11 += M6
            U[a10, base + 5] += 1
            U[a00, base + 5] -= 1
            V[b00, base + 5] += 1
            V[b01, base + 5] += 1
            W[c11, base + 5] += 1
            
            # M7 = (A01-A11)(B10+B11) -> C00 += M7
            U[a01, base + 6] += 1
            U[a11, base + 6] -= 1
            V[b10, base + 6] += 1
            V[b11, base + 6] += 1
            W[c00, base + 6] += 1
            
            col += 7
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    if time.time() > deadline:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fast vectorized verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
