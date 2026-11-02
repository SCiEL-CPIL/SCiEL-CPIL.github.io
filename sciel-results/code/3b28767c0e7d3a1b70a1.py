import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 145.0

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Strassen rank-7 for column pairs (0,1), (2,3), (4,5)
    - Schoolbook rank-4 for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25

    if _time_left() < 3.0:
        return _schoolbook(n, m, p)

    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)

    # Strassen for column pair (0,1) -> cols 0-6 of U,V,W
    add_strassen(U, V, W, col_start=0, k_start=0)
    # Strassen for column pair (2,3) -> cols 7-13
    add_strassen(U, V, W, col_start=7, k_start=2)
    # Strassen for column pair (4,5) -> cols 14-20
    add_strassen(U, V, W, col_start=14, k_start=4)

    # Schoolbook for column 6 -> cols 21-24
    cs = 21
    U[0, cs] = 1; V[6, cs] = 1; W[12, cs] = 1
    U[1, cs+1] = 1; V[13, cs+1] = 1; W[12, cs+1] = 1
    U[2, cs+2] = 1; V[6, cs+2] = 1; W[13, cs+2] = 1
    U[3, cs+3] = 1; V[13, cs+3] = 1; W[13, cs+3] = 1

    # Verify if time permits
    if _time_left() > 5.0:
        if not _verify(U, V, W, n, m, p, rank):
            return _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def add_strassen(U, V, W, col_start, k_start):
    """
    Add Strassen rank-7 terms for multiplying 2x2 A by 2x2 B block at columns k_start, k_start+1.
    
    A = [[A00, A01], [A10, A11]]  (indices 0,1,2,3 in U)
    B = [[B00, B01], [B10, B11]]  (B0x at indices k_start, k_start+1; B1x at indices 7+k_start, 7+k_start+1)
    C = [[C00, C01], [C10, C11]]  (C0x at indices 2*k_start, 2*k_start+1; C1x at 2*(k_start+1), 2*(k_start+1)+1)
    
    Strassen formulas:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    c0 = col_start
    c1 = col_start + 1
    c2 = col_start + 2
    c3 = col_start + 3
    c4 = col_start + 4
    c5 = col_start + 5
    c6 = col_start + 6

    # M1 = (A00 + A11) * (B00 + B11)
    # A00 = U[0], A11 = U[3]
    # B00 = V[k_start], B11 = V[7 + k_start+1]
    # C00 += M1 (W[2*k_start, ...])
    # C11 += M1 (W[2*(k_start+1)+1, ...])
    U[0, c0] += 1
    U[3, c0] += 1
    V[k_start, c0] += 1
    V[7 + k_start + 1, c0] += 1
    W[2 * k_start, c0] += 1
    W[2 * (k_start + 1) + 1, c0] += 1

    # M2 = (A10 + A11) * B00
    # A10 = U[2], A11 = U[3]
    # B00 = V[k_start]
    # C10 += M2 (W[2*(k_start+1), ...])
    # C11 -= M2 (W[2*(k_start+1)+1, ...])
    U[2, c1] += 1
    U[3, c1] += 1
    V[k_start, c1] += 1
    W[2 * (k_start + 1), c1] += 1
    W[2 * (k_start + 1) + 1, c1] -= 1

    # M3 = A00 * (B01 - B11)
    # A00 = U[0]
    # B01 = V[k_start+1], B11 = V[7+k_start+1]
    # C01 += M3 (W[2*k_start+1, ...])
    # C11 += M3 (W[2*(k_start+1)+1, ...])
    U[0, c2] += 1
    V[k_start + 1, c2] += 1
    V[7 + k_start + 1, c2] -= 1
    W[2 * k_start + 1, c2] += 1
    W[2 * (k_start + 1) + 1, c2] += 1

    # M4 = A11 * (B10 - B00)
    # A11 = U[3]
    # B10 = V[7+k_start], B00 = V[k_start]
    # C00 += M4 (W[2*k_start, ...])
    # C10 += M4 (W[2*(k_start+1), ...])
    U[3, c3] += 1
    V[7 + k_start, c3] += 1
    V[k_start, c3] -= 1
    W[2 * k_start, c3] += 1
    W[2 * (k_start + 1), c3] += 1

    # M5 = (A00 + A01) * B11
    # A00 = U[0], A01 = U[1]
    # B11 = V[7+k_start+1]
    # C00 -= M5 (W[2*k_start, ...])
    # C01 += M5 (W[2*k_start+1, ...])
    U[0, c4] += 1
    U[1, c4] += 1
    V[7 + k_start + 1, c4] += 1
    W[2 * k_start, c4] -= 1
    W[2 * k_start + 1, c4] += 1

    # M6 = (A10 - A00) * (B00 + B01)
    # A10 = U[2], A00 = U[0]
    # B00 = V[k_start], B01 = V[k_start+1]
    # C11 += M6 (W[2*(k_start+1)+1, ...])
    U[2, c5] += 1
    U[0, c5] -= 1
    V[k_start, c5] += 1
    V[k_start + 1, c5] += 1
    W[2 * (k_start + 1) + 1, c5] += 1

    # M7 = (A01 - A11) * (B10 + B11)
    # A01 = U[1], A11 = U[3]
    # B10 = V[7+k_start], B11 = V[7+k_start+1]
    # C00 += M7 (W[2*k_start, ...])
    U[1, c6] += 1
    U[3, c6] -= 1
    V[7 + k_start, c6] += 1
    V[7 + k_start + 1, c6] += 1
    W[2 * k_start, c6] += 1

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i*m+k, idx] = 1
                V[k*p+j, idx] = 1
                W[j*n+i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, rank):
    """
    Verify tensor decomposition.
    T[a,b,c] = 1 iff a=i*m+k, b=k*p+j, c=j*n+i
    Check sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c.
    """
    # Build target tensor
    target = set()
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                target.add((a, b, c))

    # Check valid entries
    for (a, b, c) in target:
        s = 0
        for r in range(rank):
            ua = U[a, r]
            if ua == 0:
                continue
            vb = V[b, r]
            if vb == 0:
                continue
            wc = W[c, r]
            s += ua * vb * wc
        if s != 1:
            return False

    # Check invalid entries
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) in target:
                    continue
                s = 0
                for r in range(rank):
                    ua = U[a, r]
                    if ua == 0:
                        continue
                    vb = V[b, r]
                    if vb == 0:
                        continue
                    wc = W[c, r]
                    s += ua * vb * wc
                if s != 0:
                    return False

    return True

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    np.random.seed(42)
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(rank):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        C_W += W[:, r] * SA * SB
    
    C_mat = np.zeros((2, 7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k + i]
    
    if np.allclose(C_mat, C_expected):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
