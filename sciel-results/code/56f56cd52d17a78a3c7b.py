import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill

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
        # Emergency fallback: schoolbook
        return _schoolbook(n, m, p)

    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)

    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # C0,ks += M1, C1,ks+1 += M1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[k_start, col_start+0] += 1
        V[7 + k_start+1, col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        # C1,ks += M2, C1,ks+1 -= M2
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # C0,ks+1 += M3, C1,ks+1 += M3
        U[0, col_start+2] += 1
        V[k_start+1, col_start+2] += 1
        V[7 + k_start+1, col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        # C0,ks += M4, C1,ks += M4
        U[3, col_start+3] += 1
        V[7 + k_start, col_start+3] += 1
        V[k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        # C0,ks -= M5, C0,ks+1 += M5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7 + k_start+1, col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # C1,ks+1 += M6
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[k_start, col_start+5] += 1
        V[k_start+1, col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # C0,ks += M7
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7 + k_start, col_start+6] += 1
        V[7 + k_start+1, col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    # Strassen for column pairs (0,1), (2,3), (4,5)
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)

    # Schoolbook for column 6
    cs = 21
    U[0, cs+0] += 1; V[6, cs+0] += 1; W[12, cs+0] += 1   # A00*B0,6 -> C0,6
    U[1, cs+1] += 1; V[13, cs+1] += 1; W[12, cs+1] += 1   # A01*B1,6 -> C0,6
    U[2, cs+2] += 1; V[6, cs+2] += 1; W[13, cs+2] += 1    # A10*B0,6 -> C1,6
    U[3, cs+3] += 1; V[13, cs+3] += 1; W[13, cs+3] += 1   # A11*B1,6 -> C1,6

    # Full tensor verification: check all 4*14*14 = 784 entries
    if _time_left() > 5.0:
        if not _verify_full(U, V, W, n, m, p, rank):
            return _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    """Return the trivial schoolbook decomposition of rank n*m*p."""
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

def _verify_full(U, V, W, n, m, p, rank):
    """
    Verify the full tensor decomposition.
    The multiplication tensor T has T[a,b,c] = 1 iff:
      a = i*m+k, b = k*p+j, c = j*n+i for some valid i,k,j
    and T[a,b,c] = 0 otherwise.
    We check that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c.
    """
    # Build the target tensor as a sparse dict
    target = {}
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                target[(a, b, c)] = 1

    # Check all valid entries sum to 1
    for (a, b, c), expected in target.items():
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
        if s != expected:
            return False

    # Check a sample of invalid entries sum to 0 (to catch structural errors)
    # There are 4*14*14 - 28 = 756 invalid entries; check all of them for safety
    valid_set = set(target.keys())
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) in valid_set:
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
    
    # Quick sanity check with random matrices
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
