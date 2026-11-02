import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Strassen rank-7 decomposition for 2x2 x 2x2
    def strassen_2x2():
        """Return (U, V, W) for 2x2 x 2x2 Strassen, each shape (4,7)"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Index: U[row, col] where row = i*2+j for A[i,j], col = strassen step
        # V[row, col] where row = i*2+j for B[i,j], col = strassen step
        # W[row, col] where row = i*2+j for C[i,j], col = strassen step
        
        # Step 0: M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        W[0, 0] = 1  # C11
        W[3, 0] = 1  # C22
        
        # Step 1: M2 = (A21+A22)B11
        U[2, 1] = 1  # A21
        U[3, 1] = 1  # A22
        V[0, 1] = 1  # B11
        W[2, 1] = 1  # C21
        W[3, 1] = -1  # C22
        
        # Step 2: M3 = A11(B21-B22)
        U[0, 2] = 1  # A11
        V[2, 2] = 1  # B21
        V[3, 2] = -1  # B22
        W[1, 2] = 1  # C12
        W[3, 2] = 1  # C22
        
        # Step 3: M4 = A22(B12-B11)
        U[3, 3] = 1  # A22
        V[1, 3] = 1  # B12
        V[0, 3] = -1  # B11
        W[0, 3] = 1  # C11
        W[2, 3] = 1  # C21
        
        # Step 4: M5 = (A11+A12)B22
        U[0, 4] = 1  # A11
        U[1, 4] = 1  # A12
        V[3, 4] = 1  # B22
        W[0, 4] = -1  # C11
        W[1, 4] = 1  # C12
        
        # Step 5: M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1  # A21
        U[0, 5] = -1  # A11
        V[0, 5] = 1  # B11
        V[1, 5] = 1  # B12
        W[3, 5] = 1  # C22
        
        # Step 6: M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1  # A12
        U[3, 6] = -1  # A22
        V[2, 6] = 1  # B21
        V[3, 6] = 1  # B22
        W[0, 6] = 1  # C11
        
        return U, V, W

    def schoolbook_1x2x2():
        """Return (U, V, W) for 1x2 x 2x2 schoolbook, each shape (2,4)"""
        # A is 1x2, B is 2x2, C is 2x1 (or 1x2? W shape is p*n)
        # Wait, W shape is (p*n, rank).
        # For sub-block: n_sub=1, m_sub=2, p_sub=2.
        # U shape: (1*2, 4) = (2, 4)
        # V shape: (2*2, 4) = (4, 4)
        # W shape: (2*1, 4) = (2, 4)
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # C_k = A_0 B_0k + A_1 B_1k
        # k=0: C_0 = A_0 B_00 + A_1 B_10
        # k=1: C_1 = A_0 B_01 + A_1 B_11
        
        # Col 0: A_0 B_00
        U[0, 0] = 1 # A[0,0]
        V[0, 0] = 1 # B[0,0]
        W[0, 0] = 1 # C[0,0]
        
        # Col 1: A_0 B_01
        U[0, 1] = 1 # A[0,0]
        V[1, 1] = 1 # B[0,1]
        W[1, 1] = 1 # C[1,0]
        
        # Col 2: A_1 B_10
        U[1, 2] = 1 # A[0,1]
        V[2, 2] = 1 # B[1,0]
        W[0, 2] = 1 # C[0,0]
        
        # Col 3: A_1 B_11
        U[1, 3] = 1 # A[0,1]
        V[3, 3] = 1 # B[1,1]
        W[1, 3] = 1 # C[1,0]
        
        return U, V, W

    def build_optimized():
        # Partition
        # n=3: [0,2) and [2,3)
        # m=4: [0,2) and [2,4)
        # p=6: [0,2), [2,4), [4,6)
        
        U_parts = []
        V_parts = []
        W_parts = []
        
        # Loop over n splits
        n_splits = [(0, 2), (2, 3)]
        # Loop over m splits
        m_splits = [(0, 2), (2, 4)]
        # Loop over p splits
        p_splits = [(0, 2), (2, 4), (4, 6)]
        
        for ni, nj in n_splits:
            for mi, mj in m_splits:
                for pi, pj in p_splits:
                    sn = nj - ni
                    sm = mj - mi
                    sp = pj - pi
                    
                    if sn == 2 and sm == 2 and sp == 2:
                        sU, sV, sW = strassen_2x2()
                        U_parts.append((ni, mi, pi, sn, sm, sp, sU, sV, sW))
                    elif sn == 1 and sm == 2 and sp == 2:
                        sU, sV, sW = schoolbook_1x2x2()
                        U_parts.append((ni, mi, pi, sn, sm, sp, sU, sV, sW))
                    else:
                        # Should not happen with this partition
                        raise ValueError(f"Unexpected block size {sn}x{sm}x{sp}")
        
        # Assemble
        total_rank = sum(part[6].shape[1] for part in U_parts)
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        for (ni, mi, pi, sn, sm, sp, U_sub, V_sub, W_sub) in U_parts:
            R_sub = U_sub.shape[1]
            # Map U_sub
            for row_local in range(sn * sm):
                li = row_local // sm
                lj = row_local % sm
                gi = ni + li
                gj = mi + lj
                U[gi*m + gj, col:col+R_sub] = U_sub[row_local, :]
            # Map V_sub
            for row_local in range(sm * sp):
                lj = row_local // sp
                lk = row_local % sp
                gi = mi + lj
                gj = pi + lk
                V[gi*p + gj, col:col+R_sub] = V_sub[row_local, :]
            # Map W_sub
            for row_local in range(sp * sn):
                lk = row_local // sn
                li = row_local % sn
                gi = pi + lk
                gj = ni + li
                W[gi*n + gj, col:col+R_sub] = W_sub[row_local, :]
            col += R_sub
        
        return U, V, W

    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-10):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        rng = np.random.default_rng(42)
        for _ in range(500):
            i = int(rng.integers(n))
            j = int(rng.integers(m))
            jp = int(rng.integers(m))
            k = int(rng.integers(p))
            kp = int(rng.integers(p))
            ip = int(rng.integers(n))
            if i == ip and j == jp and k == kp:
                continue
            val = 0.0
            for c in range(R):
                val += U[i*m + j, c] * V[jp*p + k, c] * W[kp*n + ip, c]
            if abs(val) > tol:
                return False
        return True

    try:
        if time.time() < deadline:
            U, V, W = build_optimized()
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
