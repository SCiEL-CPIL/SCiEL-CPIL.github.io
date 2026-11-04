import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k."""
    # Precompute U and V as lists of Fractions (rows)
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    for i in range(n):
        for j in range(m):
            ui = Uf[i * m + j]
            for k in range(p):
                vj = Vf[j * p + k]
                s = Frac(0)
                for c in range(rank):
                    s += ui[c] * vj[c]
                if s != Frac(W[k * n + i]):
                    return False
    return True

def _search_rank(rank, n, m, p, deadline, rng, max_iter):
    """Try to find an exact rank-`rank` decomposition. Returns (U,V,W) or None."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        # Random small integer U,V
        # Use small integers to keep fractions small
        U = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(mp)]
        
        # Solve for W
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                # Build system: for each (j,l) in [0,m)x[0,p): sum_c U[i,j,c]*V[l,k,c]*W[k,i,c] = delta_{j,l}
                # Unknowns: W[k,i,0..rank-1]
                # Equations: m*p
                rows = []
                rhs = []
                for j in range(m):
                    for l in range(p):
                        row = [Frac(U[i * m + j][c] * V[l * p + k][c]) for c in range(rank)]
                        rows.append(row)
                        rhs.append(Frac(1 if j == l else 0))
                
                # Solve rows x W_vec = rhs (m*p x rank system, rank unknowns)
                # Use exact Gaussian elimination on augmented matrix (m*p) x (rank+1)
                aug = [row[:] + [rhs[r]] for r, row in enumerate(rows)]
                R = len(aug)
                C = rank
                pivots = []
                col = 0
                for r in range(R):
                    if col >= C:
                        break
                    # find pivot
                    pr = None
                    for rr in range(r, R):
                        if aug[rr][col] != 0:
                            pr = rr
                            break
                    if pr is None:
                        col += 1
                        continue
                    if pr != r:
                        aug[r], aug[pr] = aug[pr], aug[r]
                    pivots.append((r, col))
                    # normalize
                    pv = aug[r][col]
                    if pv != 1:
                        aug[r] = [x / pv for x in aug[r]]
                    for rr in range(R):
                        if rr != r and aug[rr][col] != 0:
                            f = aug[rr][col]
                            aug[rr] = [a - f * b for a, b in zip(aug[rr], aug[r])]
                    col += 1
                # Check consistency
                for rr in range(R):
                    coeff_zero = all(aug[rr][c] == 0 for c in range(C))
                    if coeff_zero and aug[rr][C] != 0:
                        ok = False
                        break
                if not ok:
                    break
                # Particular solution: free columns -> 0
                Wvec = [Frac(0)] * C
                for (r, c) in pivots:
                    Wvec[c] = aug[r][C]
                
                # Check if W entries are integer or half-integer
                valid_half = True
                for c in range(C):
                    if Wvec[c].denominator not in (1, 2):
                        valid_half = False
                        break
                if not valid_half:
                    ok = False
                    break
                
                W[k * n + i] = [int(Wvec[c]) if Wvec[c].denominator == 1 else Wvec[c] for c in range(C)]
        
        if not ok:
            continue
        
        # Full exact verification
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    return None

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    # Try to find a lower rank decomposition
    # We know schoolbook is 36.
    # Try ranks 32, 34, 35
    for rank in [32, 34, 35]:
        rng = random.Random(rank) # Deterministic seed for reproducibility
        result = _search_rank(rank, n, m, p, deadline, rng, max_iter=500)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Check time
        if time.time() > deadline:
            break

    # Fallback to schoolbook
    return _schoolbook(n, m, p)
