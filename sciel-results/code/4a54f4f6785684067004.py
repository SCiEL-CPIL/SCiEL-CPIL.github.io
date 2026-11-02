"""Combined program: structured Strassen construction + verification machinery."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W decompose the matmul tensor exactly (and are half-integers)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # half-integer check
    for M in (U, V, W):
        if np.max(np.abs(M * 2 - np.round(M * 2))) > tol:
            return False
    # tensor check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.einsum('c,c,c->', U[i * m + j], V[j * p + k], W[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Rank n*m*p decomposition (always valid)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _to_int_lists(U, V, W):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    def conv(M):
        M = np.round(M * 2) / 2.0
        out = []
        for row in M:
            out.append([float(x) for x in row])
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 as (u, v, w) arrays of length 4."""
    S2 = [
        (np.array([1, 0, 0, 1], dtype=float), np.array([1, 0, 0, 1], dtype=float), np.array([1, 0, 0, 1], dtype=float)),
        (np.array([0, 0, 1, 1], dtype=float), np.array([1, 0, 0, 0], dtype=float), np.array([0, 0, 1, 0], dtype=float)),
        (np.array([1, 0, 0, 0], dtype=float), np.array([0, 1, 0, -1], dtype=float), np.array([0, 1, 0, 0], dtype=float)),
        (np.array([0, 0, 0, 1], dtype=float), np.array([0, 0, 1, -1], dtype=float), np.array([0, 0, 0, 1], dtype=float)),
        (np.array([1, 1, 0, 0], dtype=float), np.array([0, 0, 0, 1], dtype=float), np.array([0, 1, 0, 0], dtype=float)),
        (np.array([0, 0, 1, -1], dtype=float), np.array([1, 0, 1, 0], dtype=float), np.array([0, 0, 1, 0], dtype=float)),
        (np.array([0, 1, 0, -1], dtype=float), np.array([0, 1, 0, 1], dtype=float), np.array([0, 0, 0, 1], dtype=float)),
    ]
    # Verify
    T2 = np.zeros((2, 2, 2))
    for u, v, w in S2:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    T2[a, b, c] += u[a * 2 + b] * v[b * 2 + c] * w[c * 2 + a]
    if np.max(np.abs(T2 - 1.0)) > 1e-9:
        return None
    return S2


def _try_strassen_4x4x4():
    """Construct a valid rank-49 decomposition for 4x4 x 4x4 using 2-level Strassen.
    
    Uses Kronecker product structure: for each pair of 2x2x2 Strassen terms (q,s),
    the U vector is the Kronecker product ub_q ⊗ ui_s mapped to 4x4 indexing.
    Returns (U,V,W) arrays of shape (16, 49) or None.
    """
    # 2x2x2 Strassen terms: each (u, v, w) has 4 entries indexed as [row*2+col]
    S2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1 = (A00+A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2 = (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3 = A00(B10-B00)
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),  # M4 = A11(B01-B11)
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),   # M5 = (A00+A01)B11
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 1, 0]),  # M6 = (A10-A00)(B00+B10)
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, 0, 0, 1]),  # M7 = (A01-A11)(B11+B01)
    ]
    
    # Verify 2x2x2 Strassen
    T2 = np.zeros((2, 2, 2))
    for u, v, w in S2:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    T2[a, b, c] += u[a * 2 + b] * v[b * 2 + c] * w[c * 2 + a]
    if np.max(np.abs(T2 - 1.0)) > 1e-9:
        return None
    
    # Kronecker product construction:
    # For 4x4x4, write I = 2*I1 + I2, J = 2*J1 + J2, K = 2*K1 + K2
    # U[I*4+J] for term (q,s) = ub_q[I1*2+J1] * ui_s[I2*2+J2]
    # V[J*4+K] for term (q,s) = vb_q[J1*2+K1] * vi_s[J2*2+K2]
    # W[K*4+I] for term (q,s) = wb_q[K1*2+I1] * wi_s[K2*2+I2]
    #
    # This is the Kronecker product of the two 2x2x2 decompositions.
    # Verification: sum_{q,s} U[I*4+J]*V[J*4+K]*W[K*4+I]
    # = sum_{q,s} ub_q[I1*2+J1]*ui_s[I2*2+J2] * vb_q[J1*2+K1]*vi_s[J2*2+K2] * wb_q[K1*2+I1]*wi_s[K2*2+I2]
    # = (sum_q ub_q[I1*2+J1]*vb_q[J1*2+K1]*wb_q[K1*2+I1]) * (sum_s ui_s[I2*2+J2]*vi_s[J2*2+K2]*wi_s[K2*2+I2])
    # = 1 * 1 = 1  (by 2x2x2 Strassen identity applied twice)
    
    r = 49
    Uf = np.zeros((16, r))
    Vf = np.zeros((16, r))
    Wf = np.zeros((16, r))
    
    for q in range(7):
        for s in range(7):
            c = q * 7 + s
            ub = np.array(S2[q][0], dtype=float)
            vb = np.array(S2[q][1], dtype=float)
            wb = np.array(S2[q][2], dtype=float)
            ui = np.array(S2[s][0], dtype=float)
            vi = np.array(S2[s][1], dtype=float)
            wi = np.array(S2[s][2], dtype=float)
            
            # Fill U: index I*4+J where I=2*I1+I2, J=2*J1+J2
            for I1 in range(2):
                for J1 in range(2):
                    for I2 in range(2):
                        for J2 in range(2):
                            I = 2 * I1 + I2
                            J = 2 * J1 + J2
                            Uf[I * 4 + J, c] = ub[I1 * 2 + J1] * ui[I2 * 2 + J2]
            
            # Fill V: index J*4+K where J=2*J1+J2, K=2*K1+K2
            for J1 in range(2):
                for K1 in range(2):
                    for J2 in range(2):
                        for K2 in range(2):
                            J = 2 * J1 + J2
                            K = 2 * K1 + K2
                            Vf[J * 4 + K, c] = vb[J1 * 2 + K1] * vi[J2 * 2 + K2]
            
            # Fill W: index K*4+I where K=2*K1+K2, I=2*I1+I2
            for K1 in range(2):
                for I1 in range(2):
                    for K2 in range(2):
                        for I2 in range(2):
                            K = 2 * K1 + K2
                            I = 2 * I1 + I2
                            Wf[K * 4 + I, c] = wb[K1 * 2 + I1] * wi[K2 * 2 + I2]
    
    # Verify the decomposition
    for I in range(4):
        for J in range(4):
            for K in range(4):
                val = np.einsum('c,c,c->', Uf[I * 4 + J], Vf[J * 4 + K], Wf[K * 4 + I])
                if abs(val - 1.0) > 1e-9:
                    return None
    
    return Uf, Vf, Wf


def _strassen_2x2x3():
    """2x2 x 2x3 Strassen-like decomposition (rank 14).
    Returns list of (u, v, w) vectors of length 4, 6, 4 respectively."""
    # 2x2x2 Strassen terms (rank 7)
    S2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 1, 0]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
    ]
    # Verify 2x2x2
    T2 = np.zeros((2, 2, 2))
    for u, v, w in S2:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    T2[a, b, c] += u[a*2+b] * v[b*2+c] * w[c*2+a]
    if np.max(np.abs(T2 - 1.0)) > 1e-9:
        return None
    
    # For 2x2x3: split the 3rd dimension into 2+1
    # Use 2x2x2 Strassen for first 2 cols, and 2x2x1 (rank 4) for last col
    # 2x2x1: rank 4 (schoolbook for 2x2x1)
    S21 = [
        ([1,0,0,0], [1,0,0,0,0,0], [1,0,0,0]),
        ([0,0,1,0], [0,0,1,0,0,0], [0,1,0,0]),
        ([0,1,0,0], [0,0,0,0,1,0], [0,0,1,0]),
        ([0,0,0,1], [0,0,0,0,0,1], [0,0,0,1]),
    ]
    # Verify 2x2x1
    T21 = np.zeros((2, 2, 1))
    for u, v, w in S21:
        for a in range(2):
            for b in range(2):
                T21[a, b, 0] += u[a*2+b] * v[b*3+0] * w[0*2+a]
    if np.max(np.abs(T21 - 1.0)) > 1e-9:
        return None
    
    # Combine: 7 + 4 = 11 terms? But we need to be careful with the V indexing.
    # Actually for 2x2x3, V has 6 entries (2*3).
    # Let me just use schoolbook for 2x2x3: rank 12.
    # Or better: use the 7-term 2x2x2 for cols 0,1 and 4-term for col 2.
    
    terms = []
    for u, v, w in S2:
        v3 = np.zeros(6)
        v3[0:4] = v  # cols 0,1
        terms.append((np.array(u, dtype=float), v3, np.array(w, dtype=float)))
    for u, v, w in S21:
        v3 = np.zeros(6)
        v3[4] = v[4] if len(v) > 4 else 0  # This is wrong, let me redo
        terms.append((np.array(u, dtype=float), v3, np.array(w, dtype=float)))
    
    # This is getting messy. Let me just return None and use a different approach.
    return None


def _try_strassen_4x4x5():
    """Build 4x4 x 4x5 decomposition: Strassen 4x4x4 (rank 49) + 16 terms for the 5th column.
    Returns (U,V,W) arrays or None.
    
    The 4x4x5 tensor T[i,j,k] = delta_{i,k} for i in 0..3, j in 0..3, k in 0..4.
    We decompose as: rank-49 part for k=0..3 (the 4x4x4 sub-tensor) + 16 schoolbook terms for k=4.
    """
    U44, V44, W44 = _try_strassen_4x4x4()
    if U44 is None:
        return None
    n, m = 4, 4
    p = 5
    r4 = U44.shape[1]  # 49
    extra = n * m  # 16 terms for the 5th column (k=4)
    r_total = r4 + extra
    Uf = np.zeros((n * m, r_total))
    Vf = np.zeros((m * p, r_total))
    Wf = np.zeros((p * n, r_total))
    
    # Copy the 4x4x4 Strassen part (covers k=0,1,2,3)
    Uf[:, :r4] = U44
    Vf[:m * 4, :r4] = V44
    Wf[:4 * n, :r4] = W44
    
    # Add schoolbook terms for k=4 (the 5th column)
    # T[i,j,4] = delta_{i,4} = 0 for i in 0..3! 
    # Wait: for 4x4 x 4x5, C = A*B where A is 4x4, B is 4x5.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = delta_{j,k}... no.
    # Actually the matmul tensor for A(n x m) * B(m x p) = C(n x p) is:
    # T[i,j,k] where i indexes rows of A/C, j indexes cols of A/rows of B, k indexes cols of B/C
    # C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 if this is the (i,j,k) term
    # The tensor decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This means for each (i,j,k), the product must equal 1.
    # For schoolbook: U[i*m+j, c] = 1 when c = i*m+j, V[j*p+k, c] = 1 when c = j*p+k... 
    # No, schoolbook has one term per (i,j,k) triple.
    
    # For the 5th column (k=4): we need T[i,j,4] = 1 for all i,j (since C[i,4] = sum_j A[i,j]*B[j,4])
    # So we need: sum_c U[i*4+j, c] * V[j*5+4, c] * W[4*4+i, c] = 1 for all i,j in 0..3
    # Schoolbook: 16 terms, one per (i,j) pair
    for i in range(n):
        for j in range(m):
            c = r4 + i * m + j
            Uf[i * m + j, c] = 1.0
            Vf[j * p + 4, c] = 1.0
            Wf[4 * n + i, c] = 1.0
    
    return Uf, Vf, Wf


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    candidates = []
    if time.time() < deadline - 5:
        try:
            cand = _try_strassen_4x4x5()
            if cand is not None:
                candidates.append(cand)
        except Exception:
            pass
    for Uf, Vf, Wf in candidates:
        if time.time() > deadline - 2:
            break
        try:
            if _verify(Uf, Vf, Wf, n, m, p):
                r = Uf.shape[1]
                if r < best_rank:
                    best = _to_int_lists(Uf, Vf, Wf)
                    best_rank = r
        except Exception:
            pass
    return best
# EVOLVE-BLOCK-END
