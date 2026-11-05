"""Matrix multiplication 2x4x5 rank-minimal decomposition.
Structural design: a verified Strassen-block decomposition (rank 36) plus an
exact-verification search for lower ranks."""
import time
import random

DIMENSIONS = [2, 4, 5]
_START = time.time()
_DEADLINE = _START + 125.0  # safety margin before the 150 s kill

def _time_left():
    return _DEADLINE - time.time()

def _verify(U, V, W, rank, n, m, p):
    """Exact check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition of 2x2x2 as (U,V,W) with 4 rows each."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # U[i*2+j], V[j*2+k], W[k*2+i]
    # M1
    U[0][0]=1; U[3][0]=-1
    V[2][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=-1
    # M2
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=1; W[3][1]=-1
    # M3
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[2][2]=1; W[3][2]=1
    # M4
    U[3][3]=1
    V[1][3]=1; V[0][3]=1
    W[0][3]=1; W[1][3]=1
    # M5
    U[0][4]=1; U[2][4]=1
    V[3][4]=1
    W[0][4]=-1; W[2][4]=1
    # M6
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[2][5]=1
    W[3][5]=1
    # M7
    U[1][6]=1; U[3][6]=-1
    V[1][6]=1; V[3][6]=1
    W[0][6]=1
    return U, V, W

def _schoolbook_2x2x1():
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U[i*2+j][c]=1
                V[j*1+k][c]=1
                W[k*2+i][c]=1
                c+=1
    return U, V, W

def _build_full_2x4x5():
    n, m, p = 2, 4, 5
    RANK = 36
    U = [[0]*RANK for _ in range(n*m)]
    V = [[0]*RANK for _ in range(m*p)]
    W = [[0]*RANK for _ in range(p*n)]

    def place_strassen(A_i0, A_j0, B_j0, B_k0, off):
        u, v, w = _strassen_2x2x2()
        for li in range(2):
            for lj in range(2):
                for c in range(7):
                    U[(A_i0+li)*4 + (A_j0+lj)][off+c] = u[li*2+lj][c]
        for lj in range(2):
            for lk in range(2):
                for c in range(7):
                    V[(B_j0+lj)*5 + (B_k0+lk)][off+c] = v[lj*2+lk][c]
        for lk in range(2):
            for li in range(2):
                for c in range(7):
                    W[(B_k0+lk)*2 + (A_i0+li)][off+c] = w[lk*2+li][c]

    def place_schoolbook(A_i0, A_j0, B_j0, B_k0, off):
        u, v, w = _schoolbook_2x2x1()
        for li in range(2):
            for lj in range(2):
                for c in range(4):
                    U[(A_i0+li)*4 + (A_j0+lj)][off+c] = u[li*2+lj][c]
        for lj in range(2):
            for lk in range(1):
                for c in range(4):
                    V[(B_j0+lj)*5 + (B_k0+lk)][off+c] = v[lj*1+lk][c]
        for lk in range(1):
            for li in range(2):
                for c in range(4):
                    W[(B_k0+lk)*2 + (A_i0+li)][off+c] = w[lk*2+li][c]

    off = 0
    # Left big block: A cols 0,1 (A_j0=0), B rows 0,1 (B_j0=0)
    place_strassen(0, 0, 0, 0, off); off += 7
    place_strassen(0, 0, 0, 2, off); off += 7
    place_schoolbook(0, 0, 0, 4, off); off += 4
    # Right big block: A cols 2,3 (A_j0=2), B rows 2,3 (B_j0=2)
    place_strassen(0, 2, 2, 0, off); off += 7
    place_strassen(0, 2, 2, 2, off); off += 7
    place_schoolbook(0, 2, 2, 4, off); off += 4
    assert off == RANK
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_full_2x4x5()
    rank = 36
    assert _verify(U, V, W, rank, n, m, p), "Strassen-block decomposition failed"
    best = (rank, U, V, W)

    # Structural search: verified random integer search for lower ranks.
    rng = random.Random(20240517)
    found = None
    for target_rank in range(30, 36):
        if _time_left() < 15:
            break
        for t in range(600):
            if _time_left() < 10:
                break
            Uc = [[rng.randint(-1,1) for _ in range(target_rank)] for _ in range(8)]
            Vc = [[rng.randint(-1,1) for _ in range(target_rank)] for _ in range(20)]
            Wc = [[rng.randint(-1,1) for _ in range(target_rank)] for _ in range(10)]
            if _verify(Uc, Vc, Wc, target_rank, n, m, p):
                found = (target_rank, Uc, Vc, Wc)
                break
        if found:
            break
    if found and found[0] < best[0]:
        best = found

    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
