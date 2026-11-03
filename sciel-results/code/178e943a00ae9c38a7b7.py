"""Matrix multiplication 3,5,7 rank-minimal search."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_exact(U, V, W, n, m, p, trials=6):
    """Check exact decomposition using rational arithmetic on random integer inputs."""
    for _ in range(trials):
        A = np.array([[random.randint(-4, 4) for _ in range(m)] for _ in range(n)])
        B = np.array([[random.randint(-4, 4) for _ in range(p)] for _ in range(m)])
        C_true = A @ B
        rank = len(U[0])
        # C[i,k] = sum_c U[i*m+j?] ... we need to reconstruct via the tensor.
        # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        C = np.zeros((n, p), dtype=object)
        for i in range(n):
            for k in range(p):
                s = Fraction(0)
                for j in range(m):
                    t = Fraction(0)
                    for c in range(rank):
                        t += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    s += Fraction(A[i][j]) * t * Fraction(B[j][k])
                C[i][k] = s
        for i in range(n):
            for k in range(p):
                if C[i][k] != Fraction(C_true[i][k]):
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _as_fraction_matrix(rows):
    return [[Fraction(x) for x in row] for row in rows]

def _rank_over_Q(rows):
    """Gaussian rank over rationals."""
    M = [list(r) for r in rows]
    if not M or not M[0]:
        return 0
    nrow = len(M)
    ncol = len(M[0])
    r = 0
    for col in range(ncol):
        # find pivot
        piv = -1
        for i in range(r, nrow):
            if M[i][col] != 0:
                piv = i
                break
        if piv < 0:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][col]
        # normalize
        for j in range(col, ncol):
            M[r][j] = M[r][j] / pv
        for i in range(nrow):
            if i == r:
                continue
            f = M[i][col]
            if f != 0:
                for j in range(col, ncol):
                    M[i][j] = M[i][j] - f * M[r][j]
        r += 1
    return r

def _try_reduce(U, V, W, n, m, p, deadline, seed):
    """Try to drop one component by expressing it as combination of others."""
    rank = len(U[0])
    rng = random.Random(seed)
    # Build the tensor as a 3D array of size n x m x p
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    # We want to find a linear dependency among the rank-1 tensors.
    # Flatten each rank-1 tensor into a vector of length n*m*p.
    N = n*m*p
    vecs = []
    for c in range(rank):
        v = np.zeros(N)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    v[i*m*p + j*p + k] = U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        vecs.append(v)
    # Try random linear combinations: pick a target c0, see if it's in span of others.
    # Build matrix of other vectors and check if target is in span.
    best = None
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        c0 = rng.randrange(rank)
        others = [c for c in range(rank) if c != c0]
        # random subset to keep matrix small
        sub = rng.sample(others, min(len(others), 60))
        rows = []
        for c in sub:
            rows.append(vecs[c].tolist())
        # augmented with target
        target = vecs[c0].tolist()
        A = rows + [target]
        A = _as_fraction_matrix(A)
        rankA = _rank_over_Q(A)
        rankB = _rank_over_Q(rows)
        if rankA == rankB:
            # dependency found; need coefficients. Solve.
            # Build matrix with target as last column, solve for coefficients.
            # M = [rows | target], find x s.t. sum x_c rows_c = target
            # Solve via fractions: set up system.
            # We'll do a quick solve: treat as linear system R x = target
            # R is (len(sub)) x N; we need x (len(sub)). Use least-rank solve.
            # Build augmented and solve symbolically via elimination.
            # Simpler: use numpy float solve to get coefficients, then round to fractions.
            Rf = np.array(rows, dtype=float)
            tf = np.array(target, dtype=float)
            try:
                sol, res, rank_s, sv = np.linalg.lstsq(Rf, tf, rcond=None)
            except Exception:
                sol = None
            if sol is not None:
                # check residual
                if np.linalg.norm(Rf @ sol - tf) < 1e-6:
                    # build new U,V,W without c0
                    new_U = [row for idx, row in enumerate(U) for _ in [0]]  # placeholder
                    new_U = []
                    new_V = []
                    new_W = []
                    for r in range(len(U)):
                        new_U.append(U[r][:c0] + U[r][c0+1:])
                    for r in range(len(V)):
                        new_V.append(V[r][:c0] + V[r][c0+1:])
                    for r in range(len(W)):
                        new_W.append(W[r][:c0] + W[r][c0+1:])
                    # add combination
                    for c in sub:
                        idx = c if c < c0 else c-1
                        coef = Fraction(str(round(sol[sub.index(c)], 6)))
                        # better: use exact fraction from solution; approximate
                        for r in range(len(new_U)):
                            new_U[r][idx] += coef * U[r][c]
                        for r in range(len(new_V)):
                            new_V[r][idx] += coef * V[r][c]
                        for r in range(len(new_W)):
                            new_W[r][idx] += coef * W[r][c]
                    if _verify_exact(new_U, new_V, new_W, n, m, p, trials=3):
                        return new_U, new_V, new_W
    return None

def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    best = (U, V, W)
    best_rank = len(U[0])
    random.seed(12345)
    np.random.seed(12345)
    seed = 0
    while time.time() < deadline - 10:
        seed += 1
        res = _try_reduce(U, V, W, n, m, p, deadline, seed)
        if res is not None:
            U, V, W = res
            best = (U, V, W)
            best_rank = len(U[0])
            print("reduced to", best_rank, flush=True)
    return {"U": best[0], "V": best[1], "W": best[2]}
