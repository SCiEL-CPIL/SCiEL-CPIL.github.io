<<<<<<< SEARCH
"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np
import time

DIMENSIONS = [2, 2, 2]
TIME_LIMIT = 150

# EVOLVE-BLOCK-START

def construct():
    """
    Return the verified rank-7 Strassen decomposition for 2x2 matrix multiplication.
    The decomposition is exact with integer entries, matching the official trace
    convention W[k*n+i, r]. A lightweight sanity check is performed to confirm
    correctness before returning.
    """
    U, V, W = _strassen_decomposition()
    
    # Lightweight verification to ensure correctness
    if not _verify_decomposition(U, V, W):
        # Should never happen, but if it does, fall back to a known-good copy
        U, V, W = _strassen_decomposition()
    
    return {"U": U, "V": V, "W": W}
=======
"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 2]
TIME_LIMIT = 150
SAFETY_MARGIN = 5.0  # seconds before TIME_LIMIT to stop searching

# EVOLVE-BLOCK-START

def construct():
    """
    Attempt to find a rank-6 decomposition via exhaustive search over finite fields.
    Falls back to the verified rank-7 Strassen decomposition if no rank-6 solution
    is found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN

    # Try to find a rank-6 decomposition over small finite fields
    # A rank-6 solution over F_q that works for all elements implies
    # the tensor has rank <= 6 over the reals (by polynomial identity).
    for q in [2, 3]:
        if time.time() > deadline:
            break
        result = _search_rank_6_Fq(q, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}

    # Fall back to rank-7 Strassen decomposition
    U, V, W = _strassen_decomposition()
    return {"U": U, "V": V, "W": W}

def _search_rank_6_Fq(q, deadline):
    """
    Search for a rank-6 decomposition over F_q.
    
    The matrix multiplication tensor T has entries T[i,j,k] in {0,1}.
    We seek U, V, W with integer entries (lifted from F_q) such that
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c in range(4).
    
    For rank 6, we fix the structure: each column r of U, V, W is a vector
    in F_q^4. We use the fact that we can gauge-fix:
    - Scale each column: U[:,r] * s, V[:,r] * t, W[:,r] * (st)^{-1}
      so that sum of products is invariant.
    - We fix W[0,r] = 1 for all r (if nonzero) by absorbing into U,V.
    
    Actually, let's use a more direct approach: enumerate all possible
    rank-6 decompositions by fixing a gauge and searching.
    
    For F_2: each vector is in {0,1}^4, so 16 possibilities per column.
    For F_3: each vector is in {0,1,2}^4, so 81 possibilities per column.
    
    With gauge fixing (W[0,r]=1), for F_2: 8 per column, 8^6 = 262144 total.
    For F_3: 27 per column, 27^6 = 387,420,489 -- too large.
    
    Better approach: fix first column, then use the equations to constrain others.
    Or: use the structure that T is fixed, so we need:
    sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all (a,b,c).
    
    For rank 6, there are 6*12 = 72 unknowns (before gauge fixing).
    The tensor has 64 entries but many are zero. The nonzero entries are:
    T[i, j, (i,j)] = 1 where output index encodes (i,k) with k=j... 
    Wait, let me re-examine.
    
    T[a,b,c] where a indexes A (4 entries), b indexes B (4 entries), c indexes C (4 entries).
    C[i,k] = sum_j A[i,j] B[j,k]
    So T[a,b,c] = 1 iff a = i*2+j, b = j*2+k, c = k*2+i for some i,j,k.
    
    Let me enumerate:
    i=0,j=0,k=0: a=0, b=0, c=0
    i=0,j=0,k=1: a=0, b=1, c=2
    i=0,j=1,k=0: a=1, b=2, c=1
    i=0,j=1,k=1: a=1, b=3, c=3
    i=1,j=0,k=0: a=2, b=0, c=0
    i=1,j=0,k=1: a=2, b=1, c=2
    i=1,j=1,k=0: a=3, b=2, c=1
    i=1,j=1,k=1: a=3, b=3, c=3
    
    So T has exactly 8 nonzero entries, each equal to 1.
    
    For rank 6, we need 6 rank-1 tensors whose sum equals T.
    Each rank-1 tensor is u_r ⊗ v_r ⊗ w_r.
    
    Over F_2, let's try a structured search. Fix gauge: W[0,r] = 1 for r=0..5.
    Then U[:,r] and V[:,r] are free in F_2^4 (but W[:,r] has W[0,r]=1).
    
    Actually, a cleaner approach: since the problem is small, let's just do a
    bounded brute-force over the space of possible decompositions.
    
    For F_2 with gauge W[0,r]=1:
    - U columns: 16 possibilities each, 16^6 = 16M
    - V columns: 16 possibilities each
    - W columns: 8 possibilities each (W[0,r]=1, other 3 bits free)
    Total: 16^6 * 16^6 * 8^6 -- way too large.
    
    Better: use the 8 equations from the nonzero entries of T.
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] = 1 for the 8 nonzero (a,b,c).
    T[a,b,c] = 0 for the other 56 entries.
    
    This is a system of 64 polynomial equations in 72 unknowns (over F_q).
    
    Let me try a different strategy: fix the first 2 columns and solve for the rest.
    Or use a meet-in-the-middle approach.
    
    Given time constraints, let's do a limited search:
    Fix gauge W[0,r] = 1 for all r.
    For F_2: enumerate U columns and V columns, check if W can be solved.
    
    Actually, the simplest viable approach within time:
    For F_2, the space is small enough that we can try random restarts
    with local search, or a structured enumeration with pruning.
    
    Let me try: fix the first column, then use backtracking on subsequent columns.
    """
    n, m, p = 2, 2, 2
    rank = 6
    
    # Build the tensor T over F_q
    T = np.zeros((4, 4, 4), dtype=int)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i * 2 + j
                b = j * 2 + k
                c = k * 2 + i
                T[a, b, c] = 1
    
    if q == 2:
        return _search_rank_6_F2(T, deadline)
    elif q == 3:
        return _search_rank_6_F3(T, deadline)
    return None

def _search_rank_6_F2(T, deadline):
    """
    Search for rank-6 decomposition over F_2.
    
    Gauge fix: W[0,r] = 1 for r = 0, ..., 5.
    Unknowns: U (4x6), V (4x6), W (4x6) with W[0,r]=1.
    
    Equations: for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] mod 2.
    
    Strategy: Enumerate U columns (16 each) and for each, try to find V and W.
    With W[0,r]=1, W has 8 possibilities per column.
    
    Total U: 16^6 = 16,777,216 -- too many to enumerate fully.
    
    Better: fix U, then the equations become linear in V and W (bilinear).
    Actually, for fixed U and W, the equations are linear in V.
    For fixed U, the equations are:
    sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    This is still bilinear in V and W.
    
    Alternative: fix U and V, then solve for W (linear).
    For fixed U and V:
    sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    Let M_r = U[:,r] * V[:,r] (outer product, 4x4 matrix).
    Then sum_r M_r[a,b] * W[c,r] = T[a,b,c]
    
    For each (a,b), this is a linear system in W[:,0..5]:
    sum_r M_r[a,b] * W[c,r] = T[a,b,c] for c = 0,1,2,3.
    
    So for each (a,b), we have a 4x6 system: [M_0[a,b], M_1[a,b], ..., M_5[a,b]] * W[:,r] = T[a,b,:]
    
    We need this to be consistent for all (a,b).
    
    Strategy: Enumerate U (16^6 = 16M) -- too slow.
    Instead, enumerate U column by column with pruning.
    
    Let's try: fix first 3 columns of U, enumerate them (16^3 = 4096),
    then for each, try to complete the remaining 3 columns.
    
    Actually, let's just do a bounded random search with verification.
    """
    rng = np.random.default_rng(12345)
    start = time.time()
    
    # Number of random trials we can afford
    # Each trial: pick random U, V, check if W exists
    max_trials = 200000
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        # Random U and V over F_2
        U = rng.integers(0, 2, size=(4, 6))
        V = rng.integers(0, 2, size=(4, 6))
        
        # Check if W exists (linear system)
        W = _solve_W_F2(U, V, T)
        if W is not None:
            # Verify
            if _verify_over_F2(U, V, W, T):
                # Lift to integers
                U_int = U.astype(int)
                V_int = V.astype(int)
                W_int = W.astype(int)
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def _solve_W_F2(U, V, T):
    """
    Given U (4x6) and V (4x6) over F_2, solve for W (4x6) over F_2
    such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c.
    
    For each (a,b), the equation is:
    sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c] for c=0,1,2,3.
    
    This is a 4x6 linear system over F_2.
    """
    # Build the system: for each (a,b), we have 4 equations in 6 unknowns (W[0,r]..W[3,r] for r=0..5)
    # Actually, W is 4x6, so 24 unknowns total.
    # For each (a,b), we get 4 equations: one for each c.
    # Total: 16*4 = 64 equations in 24 unknowns.
    
    # Let's build the full system
    # Unknowns: W[0,0], W[0,1], ..., W[3,5] (24 unknowns)
    # Equation for (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    A_mat = np.zeros((64, 24), dtype=int)
    b_vec = np.zeros(64, dtype=int)
    
    eq_idx = 0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for r in range(6):
                    coeff = (U[a, r] * V[b, r]) % 2
                    if coeff != 0:
                        # W[c, r] is at position c*6 + r
                        A_mat[eq_idx, c * 6 + r] = coeff
                b_vec[eq_idx] = T[a, b, c] % 2
                eq_idx += 1
    
    # Solve over F_2 using Gaussian elimination
    W_flat = _gauss_F2(A_mat, b_vec)
    if W_flat is None:
        return None
    
    W = W_flat.reshape(4, 6)
    return W

def _gauss_F2(A, b):
    """
    Solve Ax = b over F_2. Returns x (length len(b's columns)) or None if inconsistent.
    A is (m, n), b is (m,).
    """
    m, n = A.shape
    A_aug = np.hstack([A, b.reshape(-1, 1)]).astype(int) % 2
    
    rank = 0
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(rank, m):
            if A_aug[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        
        # Swap
        A_aug[[rank, pivot]] = A_aug[[pivot, rank]]
        
        # Eliminate
        for row in range(m):
            if row != rank and A_aug[row, col] == 1:
                A_aug[row] = (A_aug[row] + A_aug[rank]) % 2
        
        rank += 1
    
    # Check for inconsistency
    for row in range(rank, m):
        if A_aug[row, n] == 1:
            return None
    
    # Extract solution (set free variables to 0)
    x = np.zeros(n, dtype=int)
    for i in range(rank):
        # Find the leading column in row i
        for col in range(n):
            if A_aug[i, col] == 1:
                x[col] = A_aug[i, n]
                break
    
    return x

def _verify_over_F2(U, V, W, T):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] over F_2."""
    for a in range(4):
        for b in range(4):
            for c in range(4):
                s = 0
                for r in range(6):
                    s = (s + U[a, r] * V[b, r] * W[c, r]) % 2
                if s != T[a, b, c] % 2:
                    return False
    return True

def _search_rank_6_F3(T, deadline):
    """
    Search for rank-6 decomposition over F_3.
    
    This space is much larger. Use random restarts with local search.
    """
    rng = np.random.default_rng(54321)
    start = time.time()
    
    # For F_3, each entry is in {0, 1, 2}.
    # Gauge fix: W[0,r] = 1 for r = 0, ..., 5.
    # Then W has 3^3 = 27 possibilities per column.
    # U and V have 3^4 = 81 possibilities per column.
    
    # Total space is enormous. Use random sampling with verification.
    max_trials = 50000
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        U = rng.integers(0, 3, size=(4, 6))
        V = rng.integers(0, 3, size=(4, 6))
        
        # Solve for W over F_3
        W = _solve_W_F3(U, V, T)
        if W is not None:
            if _verify_over_F3(U, V, W, T):
                U_int = U.astype(int)
                V_int = V.astype(int)
                W_int = W.astype(int)
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def _solve_W_F3(U, V, T):
    """
    Given U (4x6) and V (4x6) over F_3, solve for W (4x6) over F_3
    such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c mod 3.
    """
    A_mat = np.zeros((64, 24), dtype=int)
    b_vec = np.zeros(64, dtype=int)
    
    eq_idx = 0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for r in range(6):
                    coeff = (U[a, r] * V[b, r]) % 3
                    if coeff != 0:
                        A_mat[eq_idx, c * 6 + r] = coeff
                b_vec[eq_idx] = T[a, b, c] % 3
                eq_idx += 1
    
    W_flat = _gauss_F3(A_mat, b_vec)
    if W_flat is None:
        return None
    
    W = W_flat.reshape(4, 6)
    return W

def _gauss_F3(A, b):
    """
    Solve Ax = b over F_3. Returns x or None if inconsistent.
    """
    m, n = A.shape
    A_aug = np.hstack([A, b.reshape(-1, 1)]).astype(int) % 3
    
    rank = 0
    for col in range(n):
        pivot = -1
        for row in range(rank, m):
            if A_aug[row, col] % 3 != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        
        A_aug[[rank, pivot]] = A_aug[[pivot, rank]]
        
        # Normalize pivot row
        pivot_val = A_aug[rank, col] % 3
        inv = _inv_F3(pivot_val)
        A_aug[rank] = (A_aug[rank] * inv) % 3
        
        # Eliminate
        for row in range(m):
            if row != rank and A_aug[row, col] % 3 != 0:
                factor = A_aug[row, col] % 3
                A_aug[row] = (A_aug[row] - factor * A_aug[rank]) % 3
        
        rank += 1
    
    # Check inconsistency
    for row in range(rank, m):
        if A_aug[row, n] % 3 != 0:
            return None
    
    x = np.zeros(n, dtype=int)
    for i in range(rank):
        for col in range(n):
            if A_aug[i, col] % 3 == 1:
                x[col] = A_aug[i, n] % 3
                break
    
    return x

def _inv_F3(x):
    """Inverse of x in F_3."""
    x = x % 3
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x == 2:
        return 2  # 2*2 = 4 = 1 mod 3
    return 0

def _verify_over_F3(U, V, W, T):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] over F_3."""
    for a in range(4):
        for b in range(4):
            for c in range(4):
                s = 0
                for r in range(6):
                    s = (s + U[a, r] * V[b, r] * W[c, r]) % 3
                if s != T[a, b, c] % 3:
                    return False
    return True
>>>>>>> REPLACE

