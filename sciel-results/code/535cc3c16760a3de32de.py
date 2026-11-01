"""Rank-minimal tensor decomposition for 2x3 x 3x7 matrix multiplication."""
import time
from fractions import Fraction

_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin below 150s kill


def _check_validity(U, V, W, n, m, p, max_sample=500):
    """Verify sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == sum_l A[i,l]*B[l,k]."""
    r = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for row in U + V + W:
        if len(row) != r:
            return False
    for trial in range(max_sample):
        import random
        rng = random.Random(trial * 12345 + 7)
        A = [[Fraction(rng.randint(0, 3)) for _ in range(m)] for _ in range(n)]
        B = [[Fraction(rng.randint(0, 3)) for _ in range(p)] for _ in range(m)]
        C_true = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = Fraction(0)
                for l in range(m):
                    s += A[i][l] * B[l][k]
                C_true[i][k] = s
        C_dec = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    uj = U[i * m + j][c]
                    if uj == 0:
                        continue
                    for k in range(p):
                        vk = V[j * p + k][c]
                        if vk == 0:
                            continue
                        wval = W[k * n + i][c]
                        if wval != 0:
                            C_dec[i][k] += uj * vk * wval
        for i in range(n):
            for k in range(p):
                if C_dec[i][k] != C_true[i][k]:
                    return False
    return True


def _schoolbook(n, m, p):
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


def _strassen_222():
    """Return (U,V,W) for 2x2x2 with rank 7 (Strassen).
    U shape (4,7), V shape (4,7), W shape (4,7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00: P1, P3, P5, P6
        [0, 0, 0, 0, 1, 0, 1],    # A01: P5, P7
        [0, 1, 0, 0, 0, 1, 0],    # A10: P2, P6
        [1, 1, 0, 1, 0, 0, -1],   # A11: P1, P2, P4, P7
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0],    # B00: P1
        [0, 0, 1, 0, 0, 1, 1],    # B01: P3, P6, P7
        [1, 1, 0, -1, 0, 1, 1],   # B10: P1, P2, P4, P6, P7
        [1, 0, -1, 1, 1, 0, 1],   # B11: P1, P3, P4, P5, P7
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 (k=0,i=0): P1+P4-P5+P7
        [0, 1, 0, 1, 0, 0, 0],    # C01 (k=0,i=1): P2+P4
        [0, 0, 1, 0, 1, 0, 0],    # C10 (k=1,i=0): P3+P5
        [1, -1, 1, 0, 0, 1, 0],   # C11 (k=1,i=1): P1-P2+P3+P6
    ]
    return U, V, W


def _construct_rank39():
    """Construct rank-39 decomposition: split l into {0} and {1,2}.
    Part 1: l=0, all k=0..6: 2*1*7 = 14 terms (schoolbook)
    Part 2: l in {1,2}, k split into (0,1),(2,3),(4,5),(6):
      3 Strassen 2x2x2 blocks (rank 7 each) + 1 schoolbook 2x2x1 (rank 4) = 25
    Total: 14 + 25 = 39
    """
    n, m, p = 2, 3, 7
    rank = 39
    U = [[0] * rank for _ in range(n * m)]   # 6 x 39
    V = [[0] * rank for _ in range(m * p)]   # 21 x 39
    W = [[0] * rank for _ in range(p * n)]   # 14 x 39

    col = 0

    # Part 1: l=0, k=0..6 (schoolbook, 14 terms)
    for i in range(2):
        for k in range(7):
            U[i * 3 + 0][col] = 1
            V[0 * 7 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1

    # Part 2: l in {1,2}
    sU, sV, sW = _strassen_222()

    def add_strassen_block(col_off, k0, l_base):
        """Add Strassen 2x2x2 block for l in {l_base, l_base+1}, k in {k0, k0+1}."""
        for c in range(7):
            for ii in range(2):
                for jj in range(2):
                    val = sU[ii * 2 + jj][c]
                    if val != 0:
                        U[ii * 3 + (l_base + jj)][col_off + c] = val
            for jj in range(2):
                for kk in range(2):
                    val = sV[jj * 2 + kk][c]
                    if val != 0:
                        V[(l_base + jj) * 7 + (k0 + kk)][col_off + c] = val
            for kk in range(2):
                for ii in range(2):
                    val = sW[kk * 2 + ii][c]
                    if val != 0:
                        W[(k0 + kk) * 2 + ii][col_off + c] = val
        return col_off + 7

    def add_schoolbook_221(col_off, k_val, l_base):
        """Add schoolbook 2x2x1 block for l in {l_base, l_base+1}, k=k_val."""
        for ii in range(2):
            for jj in range(2):
                U[ii * 3 + (l_base + jj)][col_off] = 1
                V[(l_base + jj) * 7 + k_val][col_off] = 1
                W[k_val * 2 + ii][col_off] = 1
                col_off += 1
        return col_off

    # Three Strassen blocks for k-pairs (0,1), (2,3), (4,5)
    col = add_strassen_block(col, 0, 1)
    col = add_strassen_block(col, 2, 1)
    col = add_strassen_block(col, 4, 1)
    # One schoolbook block for k=6
    col = add_schoolbook_221(col, 6, 1)

    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}


def construct():
    global _START_TIME
    n, m, p = 2, 3, 7
    elapsed = time.time() - _START_TIME

    # Try rank 39 construction
    try:
        result39 = _construct_rank39()
        if _check_validity(result39["U"], result39["V"], result39["W"], n, m, p, max_sample=500):
            elapsed = time.time() - _START_TIME
            if elapsed < TIME_LIMIT:
                return result39
    except Exception:
        pass

    # Fall back to schoolbook rank 42
    return _schoolbook(n, m, p)


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
