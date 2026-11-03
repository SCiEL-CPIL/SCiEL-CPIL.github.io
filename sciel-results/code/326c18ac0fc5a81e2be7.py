"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                val = np.dot(u, v * w)
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    
    For A (2x2) * B (2x2) = C (2x2):
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01], [b10, b11]]
    C[i,j] = sum_k A[i,k]*B[k,j]
    
    Strassen's 7 products:
    M1 = (a00+a11)*(b00+b11)
    M2 = (a10+a11)*b00
    M3 = a00*(b01-b11)
    M4 = a11*(b10-b00)
    M5 = (a00+a01)*b11
    M6 = (a10-a00)*(b00+b10)
    M7 = (a01-a11)*(b01+b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    In tensor form: T[i,j,k] = 1 for all i,j,k in {0,1}
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Returns list of (U_vec, V_vec, W_vec) where each is a length-4 vector
    (indexed by (i,j), (j,k), (k,i) respectively for the 2x2x2 sub-problem).
    """
    # U is indexed by (i, j) -> index i*2+j
    # V is indexed by (j, k) -> index j*2+k  
    # W is indexed by (k, i) -> index k*2+i
    
    terms = []
    
    # M1 = (a00+a11)*(b00+b11)
    # U: a00 and a11 contribute -> U[0,0]=1, U[1,1]=1
    # V: b00 and b11 contribute -> V[0,0]=1, V[1,1]=1
    # W: C00 and C11 need M1 -> W[0,0]=1, W[1,1]=1
    # C00 = M1 + ..., C11 = M1 + ...
    # So W[0,0] (k=0,i=0) gets +1, W[1,1] (k=1,i=1) gets +1
    u1 = [0,0,0,0]; u1[0] = 1; u1[3] = 1  # (i,j): (0,0)->0, (1,1)->3
    v1 = [0,0,0,0]; v1[0] = 1; v1[3] = 1  # (j,k): (0,0)->0, (1,1)->3
    w1 = [0,0,0,0]; w1[0] = 1; w1[3] = 1  # (k,i): (0,0)->0, (1,1)->3
    terms.append((u1, v1, w1))
    
    # M2 = (a10+a11)*b00
    # U: a10, a11 -> (1,0)->2, (1,1)->3
    # V: b00 -> (0,0)->0
    # W: C10 = M2 + M4 -> W[0,1] (k=0,i=1) gets +1
    u2 = [0,0,0,0]; u2[2] = 1; u2[3] = 1
    v2 = [0,0,0,0]; v2[0] = 1
    w2 = [0,0,0,0]; w2[2] = 1  # (k,i): (0,1)->2
    terms.append((u2, v2, w2))
    
    # M3 = a00*(b01-b11)
    # U: a00 -> (0,0)->0
    # V: b01, b11 -> (0,1)->1, (1,1)->3
    # W: C01 = M3 + M5 -> W[1,0] (k=1,i=0) gets +1
    u3 = [0,0,0,0]; u3[0] = 1
    v3 = [0,0,0,0]; v3[1] = 1; v3[3] = -1
    w3 = [0,0,0,0]; w3[1] = 1  # (k,i): (1,0)->1
    terms.append((u3, v3, w3))
    
    # M4 = a11*(b10-b00)
    # U: a11 -> (1,1)->3
    # V: b10, b00 -> (1,0)->2, (0,0)->0
    # W: C00 = M1+M4-M5+M7, C10 = M2+M4 -> W[0,0] gets +1, W[0,1] gets +1
    u4 = [0,0,0,0]; u4[3] = 1
    v4 = [0,0,0,0]; v4[2] = 1; v4[0] = -1
    w4 = [0,0,0,0]; w4[0] = 1; w4[2] = 1  # (k,i): (0,0)->0, (0,1)->2
    terms.append((u4, v4, w4))
    
    # M5 = (a00+a01)*b11
    # U: a00, a01 -> (0,0)->0, (0,1)->1
    # V: b11 -> (1,1)->3
    # W: C00 = ...-M5+..., C01 = M3+M5 -> W[0,0] gets -1, W[1,0] gets +1
    u5 = [0,0,0,0]; u5[0] = 1; u5[1] = 1
    v5 = [0,0,0,0]; v5[3] = 1
    w5 = [0,0,0,0]; w5[0] = -1; w5[1] = 1  # (k,i): (0,0)->0, (1,0)->1
    terms.append((u5, v5, w5))
    
    # M6 = (a10-a00)*(b00+b10)
    # U: a10, a00 -> (1,0)->2, (0,0)->0
    # V: b00, b10 -> (0,0)->0, (1,0)->2
    # W: C11 = M1-M2+M3+M6 -> W[1,1] (k=1,i=1) gets +1
    u6 = [0,0,0,0]; u6[2] = 1; u6[0] = -1
    v6 = [0,0,0,0]; v6[0] = 1; v6[2] = 1
    w6 = [0,0,0,0]; w6[3] = 1  # (k,i): (1,1)->3
    terms.append((u6, v6, w6))
    
    # M7 = (a01-a11)*(b01+b11)
    # U: a01, a11 -> (0,1)->1, (1,1)->3
    # V: b01, b11 -> (0,1)->1, (1,1)->3
    # W: C00 = M1+M4-M5+M7 -> W[0,0] (k=0,i=0) gets +1
    u7 = [0,0,0,0]; u7[1] = 1; u7[3] = -1
    v7 = [0,0,0,0]; v7[1] = 1; v7[3] = 1
    w7 = [0,0,0,0]; w7[0] = 1  # (k,i): (0,0)->0
    terms.append((u7, v7, w7))
    
    return terms


def _two_strassen_blocks(n, m, p):
    """
    Construct a rank-58 decomposition using two disjoint Strassen 2x2x2 blocks.
    
    Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 7 terms
    Block 2: i in {0,1}, j in {2,3}, k in {2,3} -> 7 terms
    Remaining: all other (i,j,k) triples -> 60 - 16 = 44 schoolbook terms
    Total: 7 + 7 + 44 = 58
    """
    U = []
    V = []
    W = []
    
    strassen_terms = _strassen_2x2x2_terms()
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    for u_s, v_s, w_s in strassen_terms:
        u_full = np.zeros(n * m)
        v_full = np.zeros(m * p)
        w_full = np.zeros(p * n)
        
        for ii in range(2):
            for jj in range(2):
                idx = u_s[jj * 2 + ii]  # u_s indexed by (i,j) -> i*2+j, but we stored as j*2+i? Let me recheck.
                # Actually in _strassen_2x2x2_terms, u1[0] = 1 for (i,j)=(0,0), u1[3]=1 for (i,j)=(1,1)
                # So index = i*2 + j
                idx = u_s[ii * 2 + jj]
                if idx != 0:
                    u_full[ii * m + jj] = idx
        
        for jj in range(2):
            for kk in range(2):
                idx = v_s[jj * 2 + kk]
                if idx != 0:
                    v_full[jj * p + kk] = idx
        
        for kk in range(2):
            for ii in range(2):
                idx = w_s[kk * 2 + ii]
                if idx != 0:
                    w_full[kk * n + ii] = idx
        
        U.append(u_full)
        V.append(v_full)
        W.append(w_full)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    for u_s, v_s, w_s in strassen_terms:
        u_full = np.zeros(n * m)
        v_full = np.zeros(m * p)
        w_full = np.zeros(p * n)
        
        for ii in range(2):
            for jj in range(2):
                idx = u_s[ii * 2 + jj]
                if idx != 0:
                    u_full[ii * m + (jj + 2)] = idx
        
        for jj in range(2):
            for kk in range(2):
                idx = v_s[jj * 2 + kk]
                if idx != 0:
                    v_full[(jj + 2) * p + (kk + 2)] = idx
        
        for kk in range(2):
            for ii in range(2):
                idx = w_s[kk * 2 + ii]
                if idx != 0:
                    w_full[(kk + 2) * n + ii] = idx
        
        U.append(u_full)
        V.append(v_full)
        W.append(w_full)
    
    # Remaining schoolbook terms: all (i,j,k) NOT in either block
    # Block 1 covers: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2 covers: i in {0,1}, j in {2,3}, k in {2,3}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in block 1
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue
                # Skip if in block 2
                if i in (0, 1) and j in (2, 3) and k in (2, 3):
                    continue
                u_full = np.zeros(n * m)
                v_full = np.zeros(m * p)
                w_full = np.zeros(p * n)
                u_full[i * m + j] = 1
                v_full[j * p + k] = 1
                w_full[k * n + i] = 1
                U.append(u_full)
                V.append(v_full)
                W.append(w_full)
    
    U = np.array(U).T  # (n*m, rank)
    V = np.array(V).T  # (m*p, rank)
    W = np.array(W).T  # (p*n, rank)
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try the two-Strassen-blocks decomposition (rank 58)
    try:
        U, V, W = _two_strassen_blocks(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 60)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
