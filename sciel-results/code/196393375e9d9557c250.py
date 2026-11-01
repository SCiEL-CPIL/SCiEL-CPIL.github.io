"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before the 150s wall-clock limit
    rng = random.Random(12345)

    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    nm, mp, pn = n * m, m * p, p * n

    # Build the full tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    T = [[[0] * pn for _ in range(mp)] for _ in range(nm)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j][j * p + k][k * n + i] = 1

    # Schoolbook decomposition of rank n*m*p (guaranteed valid baseline).
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return U, V, W

    def verify(U, V, W, rank):
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != T[a][b][c]:
                        return False
        return True

    # Current best state (start from schoolbook).
    U, V, W = schoolbook()
    rank = len(U[0])
    best_rank = rank

    # Index of each schoolbook term: (i,j,k) -> term id in [0, n*m*p).
    term_id = {}
    tid = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                term_id[(i, j, k)] = tid
                tid += 1
    all_terms = list(term_id.keys())

    # Local-search merge: try to replace two terms (a,b,c) and (a2,b2,c2) by a
    # single rank-1 tensor (u,v,w) so that the sum of rank-1 tensors is unchanged.
    # Solves a 4-variable, 4-equation system using the (non-zero) tensor entries.
    def try_merge(i, j, k, i2, j2, k2):
        nonlocal U, V, W, rank
        a, b, c = i * m + j, j * p + k, k * n + i
        a2, b2, c2 = i2 * m + j2, j2 * p + k2, k2 * n + i2
        x1, y1, z1 = U[a][rank - 1], V[b][rank - 1], W[c][rank - 1]
        x2, y2, z2 = U[a2][rank - 1], V[b2][rank - 1], W[c2][rank - 1]
        # unknowns (u,v,w,t); equations from the four 1-entries:
        # a:  u*y1 + v*x1 + w*t = x1*y1
        # b:  x1*v + u*y2 + w*t = x1*y2
        # a2: u*y1 + v*x2 + w*t = x2*y1
        # b2: x2*v + u*y2 + w*t = x2*y2
        A = [[y1, x1, 1.0, 0.0],
             [x1, y2, 1.0, 0.0],
             [y1, x2, 1.0, 0.0],
             [x2, y2, 1.0, 0.0]]
        rhs = [x1 * y1, x1 * y2, x2 * y1, x2 * y2]
        # Gaussian elimination with partial pivoting.
        M = [row[:] + [rhs[i]] for i, row in enumerate(A)]
        sol = None
        for col in range(4):
            piv = max(range(col, 4), key=lambda r: abs(M[r][col]))
            if abs(M[piv][col]) < 1e-9:
                return False
            if piv != col:
                M[col], M[piv] = M[piv], M[col]
            d = M[col][col]
            M[col] = [x / d for x in M[col]]
            for r in range(4):
                if r != col and abs(M[r][col]) > 1e-12:
                    f = M[r][col]
                    M[r] = [M[r][c] - f * M[col][c] for c in range(5)]
        sol = [M[r][4] for r in range(4)]
        # Apply the merge in place.
        U[a][rank - 1] = sol[0]; V[b][rank - 1] = sol[1]; W[c][rank - 1] = sol[2]
        U[a2][rank - 1] = sol[0]; V[b2][rank - 1] = sol[1]; W[c2][rank - 1] = sol[2]
        if not verify(U, V, W, rank):
            # revert
            U[a][rank - 1] = x1; V[b][rank - 1] = y1; W[c][rank - 1] = z1
            U[a2][rank - 1] = x2; V[b2][rank - 1] = y2; W[c2][rank - 1] = z2
            return False
        # drop the merged column
        del U[rank - 1]; del V[rank - 1]; del W[rank - 1]
        for row in U:
            del row[rank - 1]
        for row in V:
            del row[rank - 1]
        for row in W:
            del row[rank - 1]
        rank -= 1
        if rank < best_rank:
            best_rank = rank
        return True

    # Run a time-budgeted local search, always keeping the best verified rank.
    while time.time() < deadline:
        merged_this_pass = False
        for (i, j, k) in all_terms:
            if time.time() > deadline:
                break
            for (i2, j2, k2) in all_terms:
                if (i, j, k) == (i2, j2, k2):
                    continue
                if try_merge(i, j, k, i2, j2, k2):
                    merged_this_pass = True
                    break
            if merged_this_pass:
                break
        if not merged_this_pass:
            break  # local optimum reached

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
