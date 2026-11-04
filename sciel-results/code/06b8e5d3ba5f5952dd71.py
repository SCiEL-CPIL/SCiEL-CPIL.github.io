import time
import random
import fractions
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True

def _check_half_integer(W, rank):
    """Check all entries are integer or half-integer."""
    for row in W:
        for x in row:
            if isinstance(x, Frac):
                if x.denominator not in (1, 2):
                    return False
            elif isinstance(x, (int, float)):
                if isinstance(x, float) and x != int(x) and 2*x != int(2*x):
                    return False
    return True

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

def _solve_w_system_exact(U, V, i, k, n, m, p, rank):
    """Solve for W[k*n+i] given U, V. Returns list of Fractions or None if inconsistent."""
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [Frac(U[i * m + j][c]) * Frac(V[l * p + k][c]) for c in range(rank)]
            rows.append(row)
            rhs.append(Frac(1) if j == l else Frac(0))
    
    R = len(rows)
    C = rank
    aug = [rows[r][:] + [rhs[r]] for r in range(R)]
    
    pivots = []
    col = 0
    for r in range(R):
        if col >= C:
            break
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
        pv = aug[r][col]
        if pv != 1:
            aug[r] = [x / pv for x in aug[r]]
        for rr in range(R):
            if rr != r and aug[rr][col] != 0:
                f = aug[rr][col]
                aug[rr] = [a - f * b for a, b in zip(aug[rr], aug[r])]
        col += 1
    
    for rr in range(R):
        coeff_zero = all(aug[rr][c] == 0 for c in range(C))
        if coeff_zero and aug[rr][C] != 0:
            return None
    
    Wvec = [Frac(0)] * C
    for (r, c) in pivots:
        Wvec[c] = aug[r][C]
    
    return Wvec

def _fast_rank_check(U, V, i, k, n, m, p, rank):
    """Quick numerical check if the system for W[k*n+i] is consistent."""
    # Build the coefficient matrix A where A @ W = b
    # A[j*m + l, c] = U[i*m+j, c] * V[l*p+k, c]
    # b[j*m + l] = 1 if j==l else 0
    A = np.zeros((m*m, rank))
    for j in range(m):
        for l in range(m):
            for c in range(rank):
                A[j*m + l, c] = float(U[i*m + j][c]) * float(V[l*p + k][c])
    b = np.zeros(m*m)
    for j in range(m):
        b[j*m + j] = 1.0
    
    # Check consistency: rank(A) == rank([A|b])
    try:
        rank_A = np.linalg.matrix_rank(A, tol=1e-10)
        Ab = np.column_stack([A, b])
        rank_Ab = np.linalg.matrix_rank(Ab, tol=1e-10)
        if rank_A != rank_Ab:
            return False
    except np.linalg.LinAlgError:
        return False
    return True

def _try_structured_search(rank, n, m, p, deadline, rng, max_iter=10000):
    """Try structured U,V and check if W can be solved."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Generate random U and V with small integers, but ensure some structure
        # Use a mix of strategies
        strategy = rng.randint(0, 2)
        
        if strategy == 0:
            # Dense random with {-1, 0, 1}
            U = [[rng.randint(-1, 1) for c in range(rank)] for _ in range(nm)]
            V = [[rng.randint(-1, 1) for c in range(rank)] for _ in range(mp)]
        elif strategy == 1:
            # Block-structured: each row has a few nonzeros
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            for i in range(nm):
                nnz = rng.randint(2, min(5, rank))
                cols = rng.sample(range(rank), nnz)
                for c in cols:
                    U[i][c] = rng.choice([-1, 1])
            for j in range(mp):
                nnz = rng.randint(2, min(5, rank))
                cols = rng.sample(range(rank), nnz)
                for c in cols:
                    V[j][c] = rng.choice([-1, 1])
        else:
            # Cyclic/structured pattern
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            for i in range(nm):
                base = (i * 3) % rank
                for offset in range(min(4, rank)):
                    c = (base + offset) % rank
                    U[i][c] = rng.choice([-1, 1])
            for j in range(mp):
                base = (j * 5) % rank
                for offset in range(min(4, rank)):
                    c = (base + offset) % rank
                    V[j][c] = rng.choice([-1, 1])
        
        # Quick check: no all-zero rows
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # Fast pre-filter using NumPy
        feasible = True
        for i in range(n):
            for k in range(p):
                if not _fast_rank_check(U, V, i, k, n, m, p, rank):
                    feasible = False
                    break
            if not feasible:
                break
        if not feasible:
            continue
        
        # Try to solve for all W[k*n+i]
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break
        
        if not ok:
            continue
        
        if not _check_half_integer(W, rank):
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def _try_low_rank_construction(n, m, p, deadline):
    """Try known low-rank constructions for small matrix products."""
    # For 3x3 * 3x4, try ranks from 18 upward with structured patterns
    for rank in range(18, 27):
        if time.time() > deadline:
            break
        
        # Try a specific structured approach: 
        # Use the fact that we can decompose into blocks
        # U has shape (9, rank), V has shape (12, rank), W has shape (12, rank)
        
        rng = random.Random(rank * 7 + 13)
        result = _try_structured_search(rank, n, m, p, deadline, rng, max_iter=3000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_systematic_ranks(n, m, p, deadline):
    """Try ranks from low to high with increasing time allocation."""
    ranks_to_try = list(range(18, 37))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate more time to lower ranks
        if rank <= 22:
            max_iter = 8000
        elif rank <= 27:
            max_iter = 5000
        else:
            max_iter = 2000
        
        # Strategy: Structured search
        rng = random.Random(rank * 1000 + 42)
        result = _try_structured_search(rank, n, m, p, deadline, rng, max_iter=max_iter)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    
    # Try low-rank constructions first
    try:
        result = _try_low_rank_construction(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    # Try systematic search for lower rank
    try:
        result = _try_systematic_ranks(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    return best_result
