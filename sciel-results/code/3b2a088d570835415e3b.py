"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _schoolbook_decomp(n, m, p):
    """Return (U, V, W) for schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

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

def _check_half_integer(U, V, W, rank):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for x in row:
                if isinstance(x, Frac):
                    if x.denominator not in (1, 2):
                        return False
                elif isinstance(x, (int, float)):
                    if isinstance(x, float) and x != int(x) and 2*x != int(2*x):
                        return False
    return True

def _build_w_system_matrix(U, V, i, k, n, m, p, rank):
    """Build the coefficient matrix and RHS for the W system using numpy."""
    # M has shape (m*m, rank), b has shape (m*m,)
    M = np.zeros((m * m, rank), dtype=float)
    b = np.zeros(m * m, dtype=float)
    for j in range(m):
        for l in range(m):
            idx = j * m + l
            for c in range(rank):
                M[idx, c] = float(U[i * m + j][c]) * float(V[l * p + k][c])
            if j == l:
                b[idx] = 1.0
    return M, b

def _solve_w_system_fast(U, V, i, k, n, m, p, rank):
    """
    Fast solve for W[k*n+i] given U, V.
    Uses numpy to find a solution, checks if it's half-integer.
    Falls back to exact solve only if needed.
    Returns list of Fractions or None if inconsistent.
    """
    M, b = _build_w_system_matrix(U, V, i, k, n, m, p, rank)
    
    # Quick consistency check via rank
    rank_M = np.linalg.matrix_rank(M)
    M_aug = np.column_stack([M, b])
    rank_aug = np.linalg.matrix_rank(M_aug)
    if rank_M != rank_aug:
        return None
    
    # Get a minimum-norm solution via lstsq
    sol, residuals, rank_sol, sv = np.linalg.lstsq(M, b, rcond=None)
    
    # Check if solution is close to half-integers
    scaled = 2.0 * sol
    rounded = np.round(scaled)
    if np.max(np.abs(scaled - rounded)) < 1e-8:
        # Solution is already half-integer (within tolerance)
        # Actually let's be more careful
        Wvec = []
        for c in range(rank):
            val = rounded[c]
            if abs(val) < 1e-9:
                Wvec.append(Frac(0))
            else:
                Wvec.append(Frac(int(val)))
        return Wvec
    
    # Not half-integer from lstsq. The null space may contain a half-integer solution.
    # Try: the general solution is sol + t*null_vec for t in null space.
    # For rank 9 system with rank columns, if rank > 9, there's a null space.
    # Let's try a few specific null space directions.
    
    # Alternative: do exact solve but only for a subset of columns (the pivot columns)
    # Build exact system using integer arithmetic where possible
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [U[i * m + j][c] * V[l * p + k][c] for c in range(rank)]
            rows.append(row)
            rhs.append(1 if j == l else 0)
    
    R = len(rows)
    C = rank
    
    # Gaussian elimination with Fractions, but track only pivot columns
    aug = [[Frac(x) for x in rows[r]] + [Frac(rhs[r])] for r in range(R)]
    
    pivots = []
    col = 0
    for r in range(R):
        if col >= C:
            break
        # Find pivot
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
                aug[rr] = [a - f * b_val for a, b_val in zip(aug[rr], aug[r])]
        col += 1
    
    # Check consistency
    for rr in range(R):
        coeff_zero = all(aug[rr][c] == 0 for c in range(C))
        if coeff_zero and aug[rr][C] != 0:
            return None
    
    # Particular solution: free variables = 0
    Wvec = [Frac(0)] * C
    for (r, c) in pivots:
        Wvec[c] = aug[r][C]
    
    return Wvec

def _try_random_search(rank, n, m, p, deadline, rng, max_iter, use_half_int=False):
    """Random search for a valid decomposition at given rank."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        if it % 20 == 0 and time.time() > deadline:
            break
        
        # Generate U, V - use denser matrices to make systems more likely solvable
        if use_half_int:
            vals = [Frac(1), Frac(-1), Frac(1, 2), Frac(-1, 2)]
            U = [[rng.choice(vals) if rng.random() < 0.7 else Frac(0) for c in range(rank)] for _ in range(nm)]
            V = [[rng.choice(vals) if rng.random() < 0.7 else Frac(0) for c in range(rank)] for _ in range(mp)]
        else:
            # Use -1, 0, 1 but with higher probability of non-zero
            U = [[rng.choice([-1, 0, 1]) for c in range(rank)] for _ in range(nm)]
            V = [[rng.choice([-1, 0, 1]) for c in range(rank)] for _ in range(mp)]
        
        # Skip if any row is all zeros
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # Try to solve for W
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system_fast(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = Wvec
            if not ok:
                break
        
        if not ok:
            continue
        
        # Check if W is half-integer
        if not _check_half_integer(U, V, W, rank):
            continue
        
        # Final validation
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def _try_structured_patterns(rank, n, m, p, deadline, rng):
    """
    Try structured patterns based on symmetry.
    For 3x3x4, exploit that the tensor has structure.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Pattern 1: Block diagonal structure
    # Try to construct U, V with block structure
    for attempt in range(30):
        if time.time() > deadline:
            return None
        
        # Random but structured: ensure each (i,j) row of U has at least one nonzero
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        
        # Assign random values but ensure coverage
        for c in range(rank):
            # Pick random rows to activate
            if rng.random() < 0.85:
                row = rng.randint(0, nm - 1)
                U[row][c] = rng.choice([-1, 1])
            if rng.random() < 0.85:
                row = rng.randint(0, mp - 1)
                V[row][c] = rng.choice([-1, 1])
        
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system_fast(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = Wvec
            if not ok:
                break
        
        if not ok:
            continue
        if not _check_half_integer(U, V, W, rank):
            continue
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def _search_rank(rank, n, m, p, deadline, rng, max_iter):
    """Try to find an exact rank-`rank` decomposition. Returns (U,V,W) or None."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try structured patterns first
    result = _try_structured_patterns(rank, n, m, p, deadline, rng)
    if result is not None:
        return result
    
    # Random search with integers
    result = _try_random_search(rank, n, m, p, deadline, rng, max_iter, use_half_int=False)
    if result is not None:
        return result
    
    # Random search with half-integers
    remaining_iter = max(50, max_iter // 3)
    result = _try_random_search(rank, n, m, p, deadline, rng, remaining_iter, use_half_int=True)
    if result is not None:
        return result
    
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

def _try_structured_search(n, m, p, deadline):
    """Try structured patterns that might yield lower rank."""
    # Try ranks from 27 downward
    for rank in range(27, 20, -1):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        max_iter = int(300 * (36 - rank))
        rng = random.Random(rank * 1000 + 42)
        result = _search_rank(rank, n, m, p, deadline, rng, max_iter=max_iter)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First try to find a better decomposition
    best_result = _schoolbook(n, m, p)
    
    # Try structured/random search for lower rank
    try:
        result = _try_structured_search(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    # If we still have time, try more aggressive searches at lower ranks
    remaining = deadline - time.time()
    if remaining > 10:
        for rank in range(24, 27):
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 5:
                break
            max_iter = int(remaining * 200)
            rng = random.Random(rank * 7777 + 123)
            try:
                result = _search_rank(rank, n, m, p, deadline, rng, max_iter=max_iter)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
    
    return best_result
