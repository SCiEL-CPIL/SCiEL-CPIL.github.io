import time
import numpy as np

def construct():
    deadline = time.time() + 140
    
    n, m, p = 4, 4, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Strassen 2x2x2 factors (rank 7)
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 1, 0, -1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, -1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, -1, 1, 0, 1, 0]
    ], dtype=float)
    
    # Verify Strassen 2x2x2
    def verify_strassen_2x2x2(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Check (i,j,k) -> should be 1
                    s = np.dot(U[i*2+j, :], V[j*2+k, :] * W[k*2+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
                    # Check other i'
                    for i2 in range(2):
                        if i2 != i:
                            s = np.dot(U[i2*2+j, :], V[j*2+k, :] * W[k*2+i, :])
                            if abs(s) > 1e-8:
                                return False
                    # Check other j'
                    for j2 in range(2):
                        if j2 != j:
                            s = np.dot(U[i*2+j2, :], V[j2*2+k, :] * W[k*2+i, :])
                            if abs(s) > 1e-8:
                                return False
                    # Check other k'
                    for k2 in range(2):
                        if k2 != k:
                            s = np.dot(U[i*2+j, :], V[j*2+k2, :] * W[k2*2+i, :])
                            if abs(s) > 1e-8:
                                return False
        return True
    
    if not verify_strassen_2x2x2(U2, V2, W2):
        # Fallback to schoolbook for 2x2x2
        U2 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        V2 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        W2 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
    
    # Build 4x4x4 using Kronecker product of two Strassen 2x2x2
    # Index mapping: i = i1*2 + i0, j = j1*2 + j0, k = k1*2 + k0
    # r = r1*7 + r0
    # U[i,j,r] = U2[i1,j1,r1] * U2[i0,j0,r0]
    # V[j,k,r] = V2[j1,k1,r1] * V2[j0,k0,r0]
    # W[k,i,r] = W2[k1,i1,r1] * W2[k0,i0,r0]
    
    strassen_rank = 7
    rank_444 = strassen_rank * strassen_rank  # 49
    
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))
    W_444 = np.zeros((16, rank_444))
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i0 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            k = k1 * 2 + k0
                            for r1 in range(strassen_rank):
                                for r0 in range(strassen_rank):
                                    r = r1 * strassen_rank + r0
                                    U_444[i*4+j, r] = U2[i1*2+j1, r1] * U2[i0*2+j0, r0]
                                    V_444[j*4+k, r] = V2[j1*2+k1, r1] * V2[j0*2+k0, r0]
                                    W_444[k*4+i, r] = W2[k1*2+i1, r1] * W2[k0*2+i0, r0]
    
    # Now build the full 4x4x5 decomposition
    # For k=0,1,2,3: use the 4x4x4 Strassen construction
    # For k=4: use schoolbook (16 terms)
    
    total_rank = rank_444 + 16  # 49 + 16 = 65
    
    U_full = np.zeros((nm, total_rank))
    V_full = np.zeros((mp, total_rank))
    W_full = np.zeros((pn, total_rank))
    
    # Fill in the Strassen part for k=0,1,2,3
    # The 4x4x4 construction handles all (i,j,k) for k in 0..3
    # U[i*m+j, r] where m=4, so U[i*4+j, r]
    # V[j*p+k, r] where p=5, so V[j*5+k, r]
    # W[k*n+i, r] where n=4, so W[k*4+i, r]
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for r in range(rank_444):
                    U_full[i*4+j, r] = U_444[i*4+j, r]
                    V_full[j*5+k, r] = V_444[j*4+k, r]
                    W_full[k*4+i, r] = W_444[k*4+i, r]
    
    # Fill in the schoolbook part for k=4
    r_offset = rank_444
    for i in range(4):
        for j in range(4):
            U_full[i*4+j, r_offset] = 1.0
            V_full[j*5+4, r_offset] = 1.0
            W_full[4*4+i, r_offset] = 1.0
            r_offset += 1
    
    # Verify the full decomposition
    def verify_full(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        # Check orthogonality for a sample of zero entries
        import random
        for _ in range(500):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            choice = random.randint(0, 2)
            if choice == 0:
                i2 = random.randint(0, n-1)
                if i2 != i:
                    s = np.dot(U[i2*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s) > tol:
                        return False
            elif choice == 1:
                j2 = random.randint(0, m-1)
                if j2 != j:
                    s = np.dot(U[i*m+j2, :], V[j2*p+k, :] * W[k*n+i, :])
                    if abs(s) > tol:
                        return False
            else:
                k2 = random.randint(0, p-1)
                if k2 != k:
                    s = np.dot(U[i*m+j, :], V[j*p+k2, :] * W[k2*n+i, :])
                    if abs(s) > tol:
                        return False
        return True
    
    if not verify_full(U_full, V_full, W_full):
        # Fallback to full schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, r] = 1.0
                    V[j*p + k, r] = 1.0
                    W[k*n + i, r] = 1.0
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
