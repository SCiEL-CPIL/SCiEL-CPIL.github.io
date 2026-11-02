"""Block-structured low-rank decomposition for 4x5x6 matrix multiplication."""
import time
import sys

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0


def remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _verify(U, V, W, rank):
    # Exact check of decomposition: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]
    # i.e. W tensor equals the multiplication tensor. Check on random integer matrices.
    import random
    n, m, p = DIMENSIONS
    for _ in range(3):
        if remaining() <= 0:
            return False
        A = [[random.randint(-2, 2) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-2, 2) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # Compute C via U,V,W: C[i,k] = sum_j sum_c U[i*m+j,c]*W[k*n+i,c]*V[j*p+k,c]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u = U[i * m + j][c]
                        if u == 0:
                            continue
                        acc += u * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True


def _block_winograd_2x2x2():
    # Returns 7 rank-1 terms for 2x2x2 matmul: each term is (U2x2, V2x2, W2x2) as flat 4-vecs
    # Winograd F^T (2,2,2) rank 7. Using standard identity.
    # a,b;c,d  * e,f;g,h  -> use 7 products.
    # m1=(a+d)(e+h), m2=(a+b)g? we use a known valid 7-set.
    # Known 7 products (from Winograd):
    # p1=(a+d)(e+h); p2=(c+d)e; p3=a(f-h); p4=d(g-e); p5=(a+b)g; p6=(c-a)(e+f); p7=(b-d)(h+f)
    # C00=p1-p4+p5-p7; C01=p3+p4+p5+p7; C10=p2+p4-p6+p7; C11=p1-p2+p6-p7
    Uterms = []
    Vterms = []
    Wterms = []
    def add(u, v, w, sgn):
        Uterms.append(sgn * u)
        Vterms.append(sgn * v)
        Wterms.append(sgn * w)
    # A entries indexed as (i,j)->A[i][j]; flat [A00,A01,A10,A11]
    # p1=(a+d)(e+h): A uses a+d -> (1,0,0,1)? a=A00,d=A11 => (1,0,0,1); B: e+h -> e=B00,h=B11 => (1,0,0,1); C: p1 adds + to C00,C11; - to none. W for p1: C00 +1, C11 +1 => flat C [C00,C01,C10,C11] -> (1,0,0,1)
    add([1,0,0,1], [1,0,0,1], [1,0,0,1], +1)   # p1 into C00,C11
    # p2=(c+d)e: A: c+d -> A10+A11 => (0,0,1,1); B: e=B00 => (1,0,0,0); C: +p2 at C10, -p2 at C11 => (0,0,1,-1)
    add([0,0,1,1], [1,0,0,0], [0,0,1,-1], +1)   # p2
    # p3=a(f-h): A: a=A00 => (1,0,0,0); B: f-h => B01 - B11 => (0,1,0,-1); C: +p3 at C01 => (0,1,0,0)
    add([1,0,0,0], [0,1,0,-1], [0,1,0,0], +1)   # p3
    # p4=d(g-e): A: d=A11 => (0,0,0,1); B: g-e => B10-B00 => (-1,0,1,0); C: -p4 at C00,+ at C01,+ at C10 => (-1,1,1,0)
    add([0,0,0,1], [-1,0,1,0], [-1,1,1,0], +1)  # p4
    # p5=(a+b)g: A: a+b => A00+A01 => (1,1,0,0); B: g=B10 => (0,0,1,0); C: +p5 at C00,+ at C01 => (1,1,0,0)
    add([1,1,0,0], [0,0,1,0], [1,1,0,0], +1)    # p5
    # p6=(c-a)(e+f): A: c-a => A10-A00 => (-1,0,1,0); B: e+f => B00+B01 => (1,1,0,0); C: -p6 at C10,+ at C11 => (0,0,-1,1)
    add([-1,0,1,0], [1,1,0,0], [0,0,-1,1], +1)  # p6
    # p7=(b-d)(h+f): A: b-d => A01-A11 => (0,1,0,-1); B: h+f => B11+B01 => (0,1,0,1); C: -p7 at C00,+ at C01,+ at C10,-? C11 no. => (-1,1,1,0)
    add([0,1,0,-1], [0,1,0,1], [-1,1,1,0], +1)  # p7
    return Uterms, Vterms, Wterms


def _block_decomposition():
    # Build 4x5x6 as sum of 2x2x2 Winograd blocks (rank 7) plus schoolbook remainders.
    n, m, p = DIMENSIONS
    terms_U = []
    terms_V = []
    terms_W = []
    Wt, Vt, Ut = _block_winograd_2x2x2()
    # Cover all 2x2x2 blocks: i in {0,1,2,3} -> 2 blocks, j in {0..4} -> 2 blocks (0,1),(2,3), k in {0..5} -> 2 blocks (0,1),(2,3)
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                for (u, v, w) in zip(Ut, Vt, Wt):
                    ufull = [0] * (n * m)
                    vfull = [0] * (m * p)
                    wfull = [0] * (p * n)
                    for a in range(2):
                        for b in range(2):
                            ui = (2 * ib + a) * m + (2 * jb + b)
                            ufull[ui] += u[a * 2 + b]
                    for b in range(2):
                        for c in range(2):
                            vi = (2 * jb + b) * p + (2 * kb + c)
                            vfull[vi] += v[b * 2 + c]
                    for c in range(2):
                        for a in range(2):
                            wi = (2 * kb + c) * n + (2 * ib + a)
                            wfull[wi] += w[c * 2 + a]
                    terms_U.append(ufull)
                    terms_V.append(vfull)
                    terms_W.append(wfull)
    # Covered: i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}
    # Missing: j=4 (all i,k), and k in {4,5} (all i,j)
    covered = set()
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            covered.add((2*ib+a, 2*jb+b, 2*kb+c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                ufull = [0] * (n * m); vfull = [0] * (m * p); wfull = [0] * (p * n)
                ufull[i * m + j] = 1
                vfull[j * p + k] = 1
                wfull[k * n + i] = 1
                terms_U.append(ufull)
                terms_V.append(vfull)
                terms_W.append(wfull)
    rank = len(terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    return {"U": U, "V": V, "W": W, "rank": rank}


def construct():
    n, m, p = DIMENSIONS
    best = None
    # Try block decomposition (expect rank 108).
    try:
        if remaining() > 10:
            cand = _block_decomposition()
            if remaining() > 5 and _verify(cand["U"], cand["V"], cand["W"], cand["rank"]):
                best = cand
    except Exception:
        pass
    # Fallback to schoolbook (rank 120) always valid.
    if best is None:
        best = _schoolbook()
    del best["rank"]
    return best


if __name__ == "__main__":
    r = construct()
    print("U shape:", len(r["U"]), "x", len(r["U"][0]))
    print("rank:", len(r["U"][0]))
