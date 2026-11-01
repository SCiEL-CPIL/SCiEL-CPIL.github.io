import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    start = time.time()
    DEADLINE = 110  # safety margin before 120s (evaluator kills at 150s)
    n, m, p = DIMENSIONS
    # schoolbook fallback
    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # list of constrained (a,b,d) triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                d = k*n + i
                triples.append((a, b, d, i, j, k))
    
    def is_half_int(x):
        # x is Fraction; check 2*x is integer
        return (2*x).denominator == 1
    
    def verify(U, V, W, r):
        # U: list of lists (n*m x r), etc. entries are Fraction
        for (a, b, d, i, j, k) in triples:
            s = Fraction(0)
            for c in range(r):
                s += U[a][c] * V[b][c] * W[d][c]
            if s != 1:
                return False
        return True
    
    def to_list_frac(M):
        # M is list of lists of Fraction
        return [[float(x) if False else x for x in row] for row in M]
    
    # Try increasing rank
    for r in [8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 30]:
        if time.time() > DEADLINE - 20:
            break
        max_trials = 2000
        for trial in range(max_trials):
            if time.time() > DEADLINE - 10:
                break
            # random U, V with small half-integer entries
            entries = [Fraction(-1), Fraction(-1,2), Fraction(0), Fraction(1,2), Fraction(1)]
            U = [[entries[np.random.randint(len(entries))] for _ in range(r)] for _ in range(n*m)]
            V = [[entries[np.random.randint(len(entries))] for _ in range(r)] for _ in range(m*p)]
            # Build linear system A W = 1, A is 30 x (p*n*r)
            num_eq = len(triples)
            num_unk = p*n*r
            A = [[Fraction(0) for _ in range(num_unk)] for _ in range(num_eq)]
            b = [Fraction(1) for _ in range(num_eq)]
            for t, (a, bb, d, i, j, k) in enumerate(triples):
                for c in range(r):
                    coef = U[a][c] * V[bb][c]
                    if coef != 0:
                        A[t][d*r + c] += coef
            # Solve A W = b (underdetermined). Find a solution.
            # Use Gaussian elimination over fractions.
            # Build augmented matrix
            aug = [row[:] + [b[i]] for i, row in enumerate(A)]
            # Row reduce
            rank_row = 0
            cols = num_unk
            pivot_cols = []
            for col in range(cols):
                # find pivot
                piv = -1
                for row in range(rank_row, num_eq):
                    if aug[row][col] != 0:
                        piv = row
                        break
                if piv == -1:
                    continue
                aug[rank_row], aug[piv] = aug[piv], aug[rank_row]
                # normalize
                pv = aug[rank_row][col]
                aug[rank_row] = [x / pv for x in aug[rank_row]]
                for row in range(num_eq):
                    if row != rank_row and aug[row][col] != 0:
                        factor = aug[row][col]
                        aug[row] = [aug[row][t2] - factor * aug[rank_row][t2] for t2 in range(num_unk+1)]
                pivot_cols.append(col)
                rank_row += 1
                if rank_row == num_eq:
                    break
            # Check consistency
            for row in range(rank_row, num_eq):
                if aug[row][num_unk] != 0:
                    break  # inconsistent
            else:
                # consistent; set free variables to 0, get particular solution
                Wsol = [Fraction(0) for _ in range(num_unk)]
                for row in range(rank_row):
                    col = pivot_cols[row]
                    Wsol[col] = aug[row][num_unk]
                # Check all Wsol entries are half-integers
                ok = all(is_half_int(x) for x in Wsol)
                if ok:
                    # Build U, V, W as lists
                    Wmat = [[Wsol[d*r + c] for c in range(r)] for d in range(p*n)]
                    if verify(U, V, Wmat, r):
                        # Convert to float/int representation
                        def conv(M):
                            out = []
                            for row in M:
                                nr = []
                                for x in row:
                                    # represent as int or half-int
                                    v = float(x)
                                    if v == int(v):
                                        nr.append(int(v))
                                    else:
                                        nr.append(v)
                                out.append(nr)
                            return out
                        return {"U": conv(U), "V": conv(V), "W": conv(Wmat)}
        # end trials
    # fallback
    return schoolbook()
