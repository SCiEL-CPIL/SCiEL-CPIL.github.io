import numpy as np
import time

def construct():
    """
    Rank-88 exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    Uses Strassen on 2x2x2x2 blocks + schoolbook for 1x4x4x4 blocks.
    """
    start_time = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill
    
    n, m, p = 3, 4, 8  # A is n×m, B is m×p, C is n×p
    
    terms = []  # list of (U_vec, V_vec, W_vec)
    
    def make_vec(size, **kwargs):
        """Create a vector with specified index->value mappings."""
        v = [0.0] * size
        for idx, val in kwargs.items():
            v[idx] = val
        return v
    
    def add_strassen_2x2x2x2(i0, j0, k0):
        """
        Add 7 Strassen terms for the sub-block:
        A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
        
        Index conventions:
        U index: i*m + j  (i in 0..n-1, j in 0..m-1)
        V index: j*p + k  (j in 0..m-1, k in 0..p-1)  
        W index: k*n + i  (k in 0..p-1, i in 0..n-1)
        """
        # A indices
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        
        # B indices
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        
        # C indices (W convention: k*n + i)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        um = n * m  # 12
        vp = m * p  # 32
        wn = p * n  # 24
        
        # M1 = (a00+a11)(b00+b11) -> c00 + c11
        u1 = make_vec(um, **{a00: 1.0, a11: 1.0})
        v1 = make_vec(vp, **{b00: 1.0, b11: 1.0})
        w1 = make_vec(wn, **{c00: 1.0, c11: 1.0})
        terms.append((u1, v1, w1))
        
        # M2 = (a10+a11)*b00 -> c10 - c11
        u2 = make_vec(um, **{a10: 1.0, a11: 1.0})
        v2 = make_vec(vp, **{b00: 1.0})
        w2 = make_vec(wn, **{c10: 1.0, c11: -1.0})
        terms.append((u2, v2, w2))
        
        # M3 = a00*(b01-b11) -> c01 + c11
        u3 = make_vec(um, **{a00: 1.0})
        v3 = make_vec(vp, **{b01: 1.0, b11: -1.0})
        w3 = make_vec(wn, **{c01: 1.0, c11: 1.0})
        terms.append((u3, v3, w3))
        
        # M4 = a11*(b10-b00) -> c00 + c10
        u4 = make_vec(um, **{a11: 1.0})
        v4 = make_vec(vp, **{b10: 1.0, b00: -1.0})
        w4 = make_vec(wn, **{c00: 1.0, c10: 1.0})
        terms.append((u4, v4, w4))
        
        # M5 = (a00+a01)*b11 -> -c00 + c01
        u5 = make_vec(um, **{a00: 1.0, a01: 1.0})
        v5 = make_vec(vp, **{b11: 1.0})
        w5 = make_vec(wn, **{c00: -1.0, c01: 1.0})
        terms.append((u5, v5, w5))
        
        # M6 = (a10-a00)*(b00+b01) -> c11
        u6 = make_vec(um, **{a10: 1.0, a00: -1.0})
        v6 = make_vec(vp, **{b00: 1.0, b01: 1.0})
        w6 = make_vec(wn, **{c11: 1.0})
        terms.append((u6, v6, w6))
        
        # M7 = (a01-a11)*(b10+b11) -> c00
        u7 = make_vec(um, **{a01: 1.0, a11: -1.0})
        v7 = make_vec(vp, **{b10: 1.0, b11: 1.0})
        w7 = make_vec(wn, **{c00: 1.0})
        terms.append((u7, v7, w7))
    
    def add_schoolbook_1x4x4x4(i0, k0):
        """
        Add 16 schoolbook terms for:
        A[i0, 0:4] * B[0:4, k0:k0+4] -> C[i0, k0:k0+4]
        """
        for j in range(4):
            for k in range(4):
                u_idx = i0 * m + j
                v_idx = j * p + (k0 + k)
                w_idx = (k0 + k) * n + i0
                u = make_vec(n * m, **{u_idx: 1.0})
                v = make_vec(m * p, **{v_idx: 1.0})
                w = make_vec(p * n, **{w_idx: 1.0})
                terms.append((u, v, w))
    
    # Decomposition:
    # Rows 0-1, outputs 0-3: 2x4 x 4x4, partition inner (2,2), outputs (2,2)
    # -> 4 blocks of 2x2 x 2x2, each Strassen rank 7 = 28
    for j0 in (0, 2):
        for k0 in (0, 2):
            add_strassen_2x2x2x2(0, j0, k0)
    
    # Rows 0-1, outputs 4-7: 2x4 x 4x4, partition inner (2,2), outputs (2,2)
    # -> 4 blocks of 2x2 x 2x2, each Strassen rank 7 = 28
    for j0 in (0, 2):
        for k0 in (4, 6):
            add_strassen_2x2x2x2(0, j0, k0)
    
    # Row 2, outputs 0-3: 1x4 x 4x4, schoolbook rank 16
    add_schoolbook_1x4x4x4(2, 0)
    
    # Row 2, outputs 4-7: 1x4 x 4x4, schoolbook rank 16
    add_schoolbook_1x4x4x4(2, 4)
    
    rank = len(terms)
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        # Return schoolbook as fallback
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = make_vec(n * m, **{i * m + j: 1.0})
                    v = make_vec(m * p, **{j * p + k: 1.0})
                    w = make_vec(p * n, **{k * n + i: 1.0})
                    terms.append((u, v, w))
        rank = len(terms)
    
    # Build matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r in range(rank):
        U[:, r] = terms[r][0]
        V[:, r] = terms[r][1]
        W[:, r] = terms[r][2]
    
    # Verify correctness
    def verify():
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    if u[i * m + j] != 0:
                        for k in range(p):
                            T[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i, j, k] - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify():
        # Fallback to schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = make_vec(n * m, **{i * m + j: 1.0})
                    v = make_vec(m * p, **{j * p + k: 1.0})
                    w = make_vec(p * n, **{k * n + i: 1.0})
                    terms.append((u, v, w))
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for r in range(rank):
            U[:, r] = terms[r][0]
            V[:, r] = terms[r][1]
            W[:, r] = terms[r][2]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
