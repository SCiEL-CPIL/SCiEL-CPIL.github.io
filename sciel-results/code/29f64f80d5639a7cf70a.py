"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np

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
                if isinstance(x, float) and x != int(x) and 2 * x != int(2 * x):
                    return False
    return True


def _solve_w_system_numpy(U, V, i, k, n, m, p, rank):
    """Solve for W[k*n+i] using numpy. Returns list of floats or None."""
    M = np.zeros((m * m, rank), dtype=float)
    for j in range(m):
        for l in range(m):
            for c in range(rank):
                M[j * m + l, c] = float(U[i * m + j][c]) * float(V[l * p + k][c])
    b = np.zeros(m * m, dtype=float)
    for j in range(m):
        b[j * m + j] = 1.0
    
    try:
        # Use least squares to handle potential over-determined systems
        Wvec, residuals, rank_M, sv = np.linalg.lstsq(M, b, rcond=1e-12)
        # Check if the solution is consistent
        if not np.allclose(M @ Wvec, b, atol=1e-9):
            return None
        return Wvec.tolist()
    except np.linalg.LinAlgError:
        return None


def _try_solve_all_W_fast(U, V, n, m, p, rank):
    """Try to solve for all W vectors using numpy. Returns W or None."""
    pn = p * n
    W = [[0] * rank for _ in range(pn)]
    for i in range(n):
        for k in range(p):
            Wvec = _solve_w_system_numpy(U, V, i, k, n, m, p, rank)
            if Wvec is None:
                return None
            # Check if entries are close to half-integers
            valid = True
            for x in Wvec:
                # Check if x is close to a half-integer
                if abs(x * 2 - round(x * 2)) > 1e-6:
                    valid = False
                    break
            if not valid:
                return None
            W[k * n + i] = Wvec
    return W


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


def _try_random_search(rank, n, m, p, deadline, rng, density_range=(0.2, 0.5), values=(-1, 1)):
    """Generic random search with numpy-based W solving."""
    nm = n * m
    mp = m * p
    pn = p * n
    max_iter = 5000  # More iterations possible due to speedup
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        
        d_low, d_high = density_range
        density = rng.uniform(d_low, d_high)
        
        for idx in range(nm):
            for c in range(rank):
                if rng.random() < density:
                    U[idx][c] = rng.choice(values)
        
        for idx in range(mp):
            for c in range(rank):
                if rng.random() < density:
                    V[idx][c] = rng.choice(values)
        
        # Quick check: ensure no zero rows
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        W_float = _try_solve_all_W_fast(U, V, n, m, p, rank)
        if W_float is None:
            continue
        
        # Convert to Fractions and verify exactly
        W_frac = [[Frac(int(x * 2 + 0.5) if abs(x * 2 - round(x * 2)) < 1e-6 else 0) / 2 for x in row] for row in W_float]
        # Actually, simpler: just use the float values converted to Fraction(2*x)/2
        W_frac = [[Frac(int(round(x * 2))) / 2 for x in row] for row in W_float]
        
        if _validate(U, V, W_frac, n, m, p, rank):
            # Ensure output format is clean (int if possible)
            W_out = [[int(x) if x.denominator == 1 else x for x in row] for row in W_frac]
            return U, V, W_out
    
    return None


def _try_systematic_ranks(n, m, p, deadline):
    """Try ranks from low to high with time allocation."""
    # We focus on ranks that are plausible for 3x3x3x4
    # Schoolbook is 36. We try down to 20.
    ranks_to_try = list(range(20, 36))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate time based on rank
        # Lower ranks are harder, give them more time? Or just uniform?
        # Let's give 5 seconds per rank if possible, but respect deadline.
        time_budget = min(5.0, max(1.0, (deadline - time.time()) / 10.0))
        sub_deadline = time.time() + time_budget
        
        # Strategy 1: Sparse binary
        rng = random.Random(rank * 1000 + 42)
        result = _try_random_search(rank, n, m, p, sub_deadline, rng, density_range=(0.1, 0.3), values=(-1, 1))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() >= deadline:
            break
        
        # Strategy 2: Dense binary
        rng = random.Random(rank * 7777 + 123)
        result = _try_random_search(rank, n, m, p, sub_deadline, rng, density_range=(0.4, 0.7), values=(-1, 1))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() >= deadline:
            break
            
        # Strategy 3: Small integers {-2, -1, 1, 2}
        rng = random.Random(rank * 5555 + 789)
        result = _try_random_search(rank, n, m, p, sub_deadline, rng, density_range=(0.2, 0.5), values=(-2, -1, 1, 2))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    
    # Try systematic search for lower rank
    try:
        result = _try_systematic_ranks(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    return best_result
