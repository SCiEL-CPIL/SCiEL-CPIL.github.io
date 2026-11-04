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


def _strassen_3x3x3_base():
    """
    Return a known rank-23 (or lower) decomposition for 3x3x3 matrix multiplication.
    Uses a Strassen-like construction with 7 multiplications per 2x2 block,
    extended to 3x3 via partitioning.
    
    For 3x3x3, we partition into 2x2 blocks with overlap, giving a structured
    low-rank decomposition. We'll use a simplified construction.
    """
    # A simpler approach: use the fact that 3x3x3 can be done with rank <= 23
    # We'll construct a valid decomposition using a known pattern.
    # For now, return None to indicate we need to search, but use this as a seed.
    return None


def _construct_strassen_seed(n, m, p):
    """
    Construct a seed decomposition based on Strassen-like ideas for 3x3x3,
    then extend to 3x3x4 by duplicating the last column structure.
    
    Strategy: 
    - For 3x3x3, use a rank-23 construction
    - For the 4th column, reuse the structure of column 3 (or a combination)
    """
    # We'll build a structured U, V for 3x3x4
    # The key insight: V has 12 rows (3*4), U has 9 rows (3*3), W has 12 rows (4*3)
    
    # Start with a block structure where columns 0,1,2 of V share a common pattern
    # and column 3 is a linear combination or slight perturbation
    
    rank = 21  # Start with a reasonable target
    
    # U: 9 x rank, structured by (i,j)
    # V: 12 x rank, structured by (l,k)
    # W: 12 x rank, structured by (k,i)
    
    # Use a Kronecker-like structure:
    # U[i*m+j, c] = a[i,j,c] for c in some set
    # V[l*p+k, c] = b[l,k,c] for c in some set
    
    # For the 4th column (k=3), make V[l*4+3, c] related to V[l*4+2, c]
    
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    # Assign columns in a structured way
    # First 18 columns: for 3x3x3 part
    # Last 3 columns: for the 4th column extension
    
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            for c in range(rank):
                # Structured pattern based on (i,j)
                if c < 9:
                    # Block 1: identity-like
                    U[idx][c] = 1 if (c // 3 == i and c % 3 == j) else 0
                elif c < 18:
                    # Block 2: Strassen-like combinations
                    # Use sums/differences
                    row_block = c // 3
                    col_block = c % 3
                    if row_block == i:
                        U[idx][c] = 1
                    elif row_block == (i + 1) % 3:
                        U[idx][c] = -1
                else:
                    # Block 3: for 4th column
                    U[idx][c] = 1 if (c - 18) % 3 == j else 0
    
    for l in range(3):
        for k in range(4):
            idx = l * 4 + k
            for c in range(rank):
                if c < 9:
                    V[idx][c] = 1 if (c // 3 == l and c % 3 == k % 3) else 0
                elif c < 18:
                    row_block = c // 3
                    col_block = c % 3
                    if row_block == l:
                        V[idx][c] = 1
                    elif row_block == (l + 1) % 3:
                        V[idx][c] = -1
                else:
                    V[idx][c] = 1 if (c - 18) % 3 == k % 3 else 0
    
    # Now try to solve for W
    W = [[0] * rank for _ in range(12)]
    ok = True
    for i in range(3):
        for k in range(4):
            Wvec = _solve_w_system_exact(U, V, i, k, 3, 3, 4, rank)
            if Wvec is None:
                ok = False
                break
            W[k * 3 + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
        if not ok:
            break
    
    if ok and _check_half_integer(W, rank) and _validate(U, V, W, 3, 3, 4, rank):
        return {"U": U, "V": V, "W": W}
    
    return None


def _try_strassen_extension(rank, n, m, p, deadline, rng, max_iter=5000):
    """
    Try Strassen-like structured decompositions with perturbations.
    Start from a base pattern and add small random perturbations.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Start with a structured base
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        
        # Base pattern: assign primary roles
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # Primary assignment
                primary = (i * m + j) % rank
                U[idx][primary] = rng.choice([1, -1])
                
                # Secondary assignments (Strassen-like combinations)
                if rank > 1 and rng.random() < 0.5:
                    secondary = (primary + rng.randint(1, min(3, rank - 1))) % rank
                    if secondary != primary:
                        U[idx][secondary] = rng.choice([-1, 1, 2])
                
                # Tertiary for more structure
                if rank > 2 and rng.random() < 0.3:
                    tertiary = (primary + rng.randint(4, min(6, rank - 1))) % rank
                    if tertiary != primary and tertiary != (primary + 1) % rank:
                        U[idx][tertiary] = rng.choice([-1, 1])
        
        for l in range(m):
            for k in range(p):
                idx = l * p + k
                primary = (l * p + k) % rank
                V[idx][primary] = rng.choice([1, -1])
                
                if rank > 1 and rng.random() < 0.5:
                    secondary = (primary + rng.randint(1, min(3, rank - 1))) % rank
                    if secondary != primary:
                        V[idx][secondary] = rng.choice([-1, 1, 2])
                
                if rank > 2 and rng.random() < 0.3:
                    tertiary = (primary + rng.randint(4, min(6, rank - 1))) % rank
                    if tertiary != primary and tertiary != (primary + 1) % rank:
                        V[idx][tertiary] = rng.choice([-1, 1])
        
        # Check no all-zero rows
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # Solve for W
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


def _try_kronecker_structure(rank, n, m, p, deadline, rng, max_iter=3000):
    """
    Try Kronecker product structures where U and V have block-diagonal
    or block-constant patterns.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        
        # Divide rank into blocks
        block_size = max(1, rank // 3)
        
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # Assign to blocks based on (i,j)
                block = (i + j) % 3
                for offset in range(block_size):
                    c = block * block_size + offset
                    if c < rank:
                        U[idx][c] = rng.choice([-1, 0, 1])
        
        for l in range(m):
            for k in range(p):
                idx = l * p + k
                block = (l + k) % 3
                for offset in range(block_size):
                    c = block * block_size + offset
                    if c < rank:
                        V[idx][c] = rng.choice([-1, 0, 1])
        
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
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


def _try_local_refinement(U, V, W, n, m, p, rank, deadline, rng, max_iter=1000):
    """
    Given a near-valid decomposition, try small local perturbations to fix inconsistencies.
    This is a polish step.
    """
    # Check if we're close to valid
    # For now, just try random single-entry modifications
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Pick a random entry to modify
        target = rng.choice(['U', 'V', 'W'])
        if target == 'U':
            idx = rng.randrange(n * m)
            c = rng.randrange(rank)
            U[idx][c] += rng.choice([-1, 1])
        elif target == 'V':
            idx = rng.randrange(m * p)
            c = rng.randrange(rank)
            V[idx][c] += rng.choice([-1, 1])
        else:
            idx = rng.randrange(p * n)
            c = rng.randrange(rank)
            W[idx][c] += rng.choice([-1, 1])
        
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


def _try_systematic_ranks(n, m, p, deadline):
    """Try ranks from low to high with increasing time allocation."""
    ranks_to_try = list(range(18, 37))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate time based on rank
        if rank <= 22:
            max_iter1 = 6000
            max_iter2 = 4000
            max_iter3 = 2000
        elif rank <= 28:
            max_iter1 = 4000
            max_iter2 = 2500
            max_iter3 = 1500
        else:
            max_iter1 = 1500
            max_iter2 = 1000
            max_iter3 = 800
        
        # Strategy 1: Strassen extension
        rng = random.Random(rank * 1000 + 42)
        result = _try_strassen_extension(rank, n, m, p, deadline, rng, max_iter=max_iter1)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Strategy 2: Kronecker structure
        rng = random.Random(rank * 7777 + 123)
        result = _try_kronecker_structure(rank, n, m, p, deadline, rng, max_iter=max_iter2)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Strategy 3: Random search (original)
        rng = random.Random(rank * 5555 + 789)
        U = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(n * m)]
        V = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(m * p)]
        
        W = [[0] * rank for _ in range(p * n)]
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
        
        if ok and _check_half_integer(W, rank) and _validate(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    
    # Try Strassen seed
    try:
        seed = _construct_strassen_seed(n, m, p)
        if seed is not None:
            best_result = seed
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
