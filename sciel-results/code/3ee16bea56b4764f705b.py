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


def _fast_consistency_check(U, V, i, k, n, m, p, rank):
    """Fast numpy-based consistency check for the W system."""
    M = np.zeros((m * m, rank), dtype=float)
    for j in range(m):
        for l in range(m):
            for c in range(rank):
                M[j * m + l, c] = float(U[i * m + j][c]) * float(V[l * p + k][c])
    b = np.zeros(m * m, dtype=float)
    for j in range(m):
        b[j * m + j] = 1.0
    M_aug = np.column_stack([M, b])
    rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    rank_aug = np.linalg.matrix_rank(M_aug, tol=1e-9)
    if rank_M != rank_aug:
        return False
    return True


def _try_solve_all_W(U, V, n, m, p, rank):
    """Try to solve for all W vectors. Returns W or None."""
    pn = p * n
    W = [[0] * rank for _ in range(pn)]
    for i in range(n):
        for k in range(p):
            if not _fast_consistency_check(U, V, i, k, n, m, p, rank):
                return None
            Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
            if Wvec is None:
                return None
            W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
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


def _try_strassen_like(rank, n, m, p, deadline, rng):
    """Try Strassen-like structured patterns."""
    nm = n * m
    mp = m * p
    pn = p * n
    max_iter = 2000
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        
        # Use a structured pattern with small integers
        for idx in range(nm):
            nnz = rng.randint(1, 3)
            cols = rng.sample(range(rank), min(nnz, rank))
            for c in cols:
                U[idx][c] = rng.choice([-2, -1, 1, 2])
        
        for idx in range(mp):
            nnz = rng.randint(1, 3)
            cols = rng.sample(range(rank), min(nnz, rank))
            for c in cols:
                V[idx][c] = rng.choice([-2, -1, 1, 2])
        
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        W = _try_solve_all_W(U, V, n, m, p, rank)
        if W is None:
            continue
        
        if not _check_half_integer(W, rank):
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None


def _try_kronecker_block(rank, n, m, p, deadline, rng):
    """Try Kronecker-like block structures."""
    nm = n * m
    mp = m * p
    pn = p * n
    max_iter = 1500
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        
        # Block-structured: group by i, then j
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                base = (i * m + j) % rank
                U[idx][base] = rng.choice([1, -1])
                if rank > 1 and rng.random() < 0.5:
                    sec = (base + 1) % rank
                    U[idx][sec] = rng.choice([-1, 1])
        
        for l in range(m):
            for k in range(p):
                idx = l * p + k
                base = (l * p + k) % rank
                V[idx][base] = rng.choice([1, -1])
                if rank > 1 and rng.random() < 0.5:
                    sec = (base + 1) % rank
                    V[idx][sec] = rng.choice([-1, 1])
        
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        W = _try_solve_all_W(U, V, n, m, p, rank)
        if W is None:
            continue
        
        if not _check_half_integer(W, rank):
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None


def _try_binary_sparse(rank, n, m, p, deadline, rng):
    """Try binary sparse patterns (entries in {-1, 0, 1})."""
    nm = n * m
    mp = m * p
    pn = p * n
    max_iter = 3000
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        
        density = rng.uniform(0.3, 0.6)
        for idx in range(nm):
            for c in range(rank):
                if rng.random() < density:
                    U[idx][c] = rng.choice([-1, 1])
        
        for idx in range(mp):
            for c in range(rank):
                if rng.random() < density:
                    V[idx][c] = rng.choice([-1, 1])
        
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        W = _try_solve_all_W(U, V, n, m, p, rank)
        if W is None:
            continue
        
        if not _check_half_integer(W, rank):
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None


def _try_systematic_ranks(n, m, p, deadline):
    """Try ranks from low to high with time allocation."""
    ranks_to_try = list(range(18, 37))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate time based on rank
        if rank <= 22:
            time_budget = min(20, max(5, (deadline - time.time()) / 2))
        elif rank <= 28:
            time_budget = min(15, max(3, (deadline - time.time()) / 3))
        else:
            time_budget = min(10, max(2, (deadline - time.time()) / 5))
        
        sub_deadline = time.time() + time_budget
        
        # Strategy 1: Strassen-like
        rng = random.Random(rank * 1000 + 42)
        result = _try_strassen_like(rank, n, m, p, sub_deadline, rng)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() >= deadline:
            break
        
        # Strategy 2: Kronecker block
        rng = random.Random(rank * 7777 + 123)
        result = _try_kronecker_block(rank, n, m, p, sub_deadline, rng)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() >= deadline:
            break
        
        # Strategy 3: Binary sparse
        rng = random.Random(rank * 5555 + 789)
        result = _try_binary_sparse(rank, n, m, p, sub_deadline, rng)
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
