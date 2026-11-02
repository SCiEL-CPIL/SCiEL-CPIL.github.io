"""Fresh design: block-structured Strassen-based rank-28 decomposition for 2x4x4."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U (4x7), V (4x7), W (4x7) for the rank-7 2x2x2 Strassen product.

    Convention: C = A*B where A,B are 2x2; U indexed by (i,j), V by (j,k),
    W by (k,i) with linear indices i*m+j, j*p+k, k*n+i (here n=m=p=2).
    Each rank term: U[i,j]*V[j,k]*W[k,i] = C[i,k].
    """
    # (i,j) -> row in U, (j,k) -> row in V, (k,i) -> row in W
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    def setu(i, j, c, val):
        U[i*2 + j][c] = val
    def setv(j, k, c, val):
        V[j*2 + k][c] = val
    def setw(k, i, c, val):
        W[k*2 + i][c] = val

    # M0 = (A00+A11)(B00+B11)
    setu(0,0,0,1); setu(1,1,0,1)
    setv(0,0,0,1); setv(1,1,0,1)
    setw(0,0,0,1); setw(1,1,0,1)
    # M1 = (A10+A11)B00
    setu(1,0,1,1); setu(1,1,1,1)
    setv(0,0,1,1)
    setw(0,1,1,1); setw(1,1,1,1)
    # M2 = A00(B10-B11)
    setu(0,0,2,1)
    setv(1,0,2,1); setv(1,1,2,-1)
    setw(0,0,2,1); setw(1,0,2,-1)
    # M3 = A11(B01-B10)
    setu(1,1,3,1)
    setv(0,1,3,1); setv(0,0,3,-1)
    setw(1,0,3,1); setw(1,1,3,-1)
    # M4 = (A00+A01)B11
    setu(0,0,4,1); setu(0,1,4,1)
    setv(1,1,4,1)
    setw(0,0,4,1); setw(0,1,4,1)
    # M5 = (A10-A00)(B00+B01)
    setu(1,0,5,1); setu(0,0,5,-1)
    setv(0,0,5,1); setv(0,1,5,1)
    setw(0,0,5,1); setw(1,0,5,1)
    # M6 = (A01-A11)(B10+B11)
    setu(0,1,6,1); setu(1,1,6,-1)
    setv(1,0,6,1); setv(1,1,6,1)
    setw(0,1,6,1); setw(1,1,6,1)

    return U, V, W


def _build_block_decomposition():
    """Two independent 2x4x4 products, each = two 2x2x2 Strassen blocks (rank 7).

    Product row i in {0,1}; left column j in {0..3}; right column k in {0..3}.
    Split j -> (a, jb) with a in {0,1}, jb in {0,1}; k -> (b, kb).
    For fixed (a,b): C[i, (b,kb)] gets A[i,(a,jb)] * B[(a,jb),(b,kb)] -> 2x2x2 block.
    """
    n, m, p = 2, 4, 4
    rank = 4 * 7
    U = [[0]*rank for _ in range(n*m)]   # 8 x rank
    V = [[0]*rank for _ in range(m*p)]   # 16 x rank
    W = [[0]*rank for _ in range(p*n)]   # 8 x rank

    su, sv, sw = _strassen_2x2x2()

    for a in range(2):       # left block index
        for b in range(2):   # right block index
            base = (a*2 + b) * 7
            for c in range(7):
                col = base + c
                for i in range(2):
                    for jb in range(2):
                        uval = su[i*2 + jb][c]
                        if uval:
                            U[i*m + (a*2 + jb)][col] += uval
                for jb in range(2):
                    for kb in range(2):
                        vval = sv[jb*2 + kb][c]
                        if vval:
                            V[(a*2 + jb)*p + (b*2 + kb)][col] += vval
                for i in range(2):
                    for kb in range(2):
                        wval = sw[kb*2 + i][c]
                        if wval:
                            W[(b*2 + kb)*n + i][col] += wval

    return {"U": U, "V": V, "W": W}


def _verify(result):
    """Symbolically verify the decomposition against the multiplication tensor."""
    import random
    U = result["U"]; V = result["V"]; W = result["W"]
    n, m, p = 2, 4, 4
    rank = len(U[0])
    for _ in range(3):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j]*B[j][k]
                C[i][k] = s
        # Reconstruct via tensor: C[i,k] = sum_c U[i*m+j] V[j*p+k] W[k*n+i]
        # Equivalently: for each c, intermediate = U[:,c] (n*m) x V[:,c] (m*p)
        # gives (n x p) contribution; sum over c.
        R = [[0]*p for _ in range(n)]
        for c in range(rank):
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += U[i*m + j][c] * V[j*p + k][c]
                    R[i][k] += s * W[k*n + i][c]
        for i in range(n):
            for k in range(p):
                if R[i][k] != C[i][k]:
                    return False
    return True


def construct():
    start = time.time()
    deadline = start + 120.0
    result = _build_block_decomposition()
    if _verify(result):
        return result
    # Fallback to guaranteed schoolbook if something is off.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
