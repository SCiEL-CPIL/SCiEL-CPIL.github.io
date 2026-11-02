import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    def strassen_2x2():
        """Return (U, V, W) for 2x2 x 2x2 Strassen, each shape (4,7)"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Step 0: M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Step 1: M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # Step 2: M3 = A11(B21-B22)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # Step 3: M4 = A22(B12-B11)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # Step 4: M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # Step 5: M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Step 6: M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
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
        # Sample zero entries
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
    
    def build_structured():
        """
        Partition:
        n=3: {0,1} and {2}
        m=4: {0,1} and {2,3}
        p=6: {0,1}, {2,3}, {4,5}
        
        Use Strassen for 2x2 x 2x2 sub-blocks, schoolbook for the rest.
        """
        n_splits = [(0,2), (2,3)]
        m_splits = [(0,2), (2,4)]
        p_splits = [(0,2), (2,4), (4,6)]
        
        U_parts = []
        V_parts = []
        W_parts = []
        
        for (ni, nj) in n_splits:
            for (mi, mj) in m_splits:
                for (pi, pj) in p_splits:
                    sn = nj - ni
                    sm = mj - mi
                    sp = pj - pi
                    
                    if sn == 2 and sm == 2 and sp == 2:
                        sU, sV, sW = strassen_2x2()
                        U_parts.append((ni, mi, pi, sn, sm, sp, sU, sV, sW))
                    else:
                        rank_sub = sn * sm * sp
                        U_sub = np.zeros((sn*sm, rank_sub))
                        V_sub = np.zeros((sm*sp, rank_sub))
                        W_sub = np.zeros((sp*sn, rank_sub))
                        col = 0
                        for li in range(sn):
                            for lj in range(sm):
                                for lk in range(sp):
                                    U_sub[li*sm+lj, col] = 1
                                    V_sub[lj*sp+lk, col] = 1
                                    W_sub[lk*sn+li, col] = 1
                                    col += 1
                        U_parts.append((ni, mi, pi, sn, sm, sp, U_sub, V_sub, W_sub))
        
        total_rank = sum(part[6].shape[1] for part in U_parts)
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        for (ni, mi, pi, sn, sm, sp, U_sub, V_sub, W_sub) in U_parts:
            R_sub = U_sub.shape[1]
            for row_local in range(sn * sm):
                li = row_local // sm
                lj = row_local % sm
                gi = ni + li
                gj = mi + lj
                U[gi*m + gj, col:col+R_sub] = U_sub[row_local, :]
            for row_local in range(sm * sp):
                lj = row_local // sp
                lk = row_local % sp
                gi = mi + lj
                gj = pi + lk
                V[gi*p + gj, col:col+R_sub] = V_sub[row_local, :]
            for row_local in range(sp * sn):
                lk = row_local // sn
                li = row_local % sn
                gi = pi + lk
                gj = ni + li
                W[gi*n + gj, col:col+R_sub] = W_sub[row_local, :]
            col += R_sub
        
        return U, V, W
    
    def build_strassen_3x4_4x6():
        """
        More aggressive Strassen-based decomposition.
        
        Strategy: Split m=4 into 2+2.
        A = [A1 | A2] (3x2 each), B = [B1; B2] (2x6 each)
        C = A1*B1 + A2*B2
        
        For each 3x2 x 2x6 product, further split:
        n: {0,1} + {2}, p: {0,1} + {2,3} + {4,5}
        
        Sub-blocks for A1*B1 (and similarly A2*B2):
        - (2x2) x (2x2) -> (2x2): Strassen rank 7  [3 times per product]
        - (2x2) x (2x2) -> (2x2): Strassen rank 7
        - (1x2) x (2x2) -> (1x2): schoolbook rank 4 [3 times per product]
        
        Per 3x2 x 2x6: 3*7 + 3*4 = 21 + 12 = 33
        Total: 2 * 33 = 66
        """
        # n_splits, m_splits, p_splits for the full problem
        n_splits = [(0,2), (2,3)]
        m_splits = [(0,2), (2,4)]
        p_splits = [(0,2), (2,4), (4,6)]
        
        U_parts = []
        
        for (ni, nj) in n_splits:
            for (mi, mj) in m_splits:
                for (pi, pj) in p_splits:
                    sn = nj - ni
                    sm = mj - mi
                    sp = pj - pi
                    
                    if sn == 2 and sm == 2 and sp == 2:
                        sU, sV, sW = strassen_2x2()
                        U_parts.append((ni, mi, pi, sn, sm, sp, sU, sV, sW))
                    else:
                        rank_sub = sn * sm * sp
                        U_sub = np.zeros((sn*sm, rank_sub))
                        V_sub = np.zeros((sm*sp, rank_sub))
                        W_sub = np.zeros((sp*sn, rank_sub))
                        col = 0
                        for li in range(sn):
                            for lj in range(sm):
                                for lk in range(sp):
                                    U_sub[li*sm+lj, col] = 1
                                    V_sub[lj*sp+lk, col] = 1
                                    W_sub[lk*sn+li, col] = 1
                                    col += 1
                        U_parts.append((ni, mi, pi, sn, sm, sp, U_sub, V_sub, W_sub))
        
        total_rank = sum(part[6].shape[1] for part in U_parts)
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        for (ni, mi, pi, sn, sm, sp, U_sub, V_sub, W_sub) in U_parts:
            R_sub = U_sub.shape[1]
            for row_local in range(sn * sm):
                li = row_local // sm
                lj = row_local % sm
                gi = ni + li
                gj = mi + lj
                U[gi*m + gj, col:col+R_sub] = U_sub[row_local, :]
            for row_local in range(sm * sp):
                lj = row_local // sp
                lk = row_local % sp
                gi = mi + lj
                gj = pi + lk
                V[gi*p + gj, col:col+R_sub] = V_sub[row_local, :]
            for row_local in range(sp * sn):
                lk = row_local // sn
                li = row_local % sn
                gi = pi + lk
                gj = ni + li
                W[gi*n + gj, col:col+R_sub] = W_sub[row_local, :]
            col += R_sub
        
        return U, V, W
    
    # Try structured decomposition (rank 70)
    try:
        if time.time() < deadline:
            U, V, W = build_structured()
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Try the alternative structured decomposition (rank 66)
    try:
        if time.time() < deadline:
            U, V, W = build_strassen_3x4_4x6()
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 72)
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
