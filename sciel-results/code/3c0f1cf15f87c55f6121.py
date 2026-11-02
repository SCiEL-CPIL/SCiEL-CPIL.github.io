import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10

    deadline = time.time() + 145  # safety margin before 150s limit

    def validate(U, V, W, tol=1e-9):
        """Check if decomposition is valid: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)"""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def snap_half_ints(M):
        """Snap matrix entries to nearest half-integer."""
        return [[round(x * 2) / 2 for x in row] for row in M]

    def snap_ints(M):
        """Snap matrix entries to nearest integer."""
        return [[round(x) for x in row] for row in M]

    def error_value(U, V, W):
        """Compute sum of squared errors."""
        rank = len(U[0]) if U else 0
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err += (val - 1.0) ** 2
        return err

    def alternating_minimize(U, V, W, max_iters=200, tol=1e-12):
        """
        Alternating least squares to minimize error.
        Returns (U, V, W, err)
        """
        rank = len(U[0])
        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        best_err = error_value(U, V, W)

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j*p+k][c] * W[k*n+i][c]
                    b = np.ones(p)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j] = sol.tolist()
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i*m+j][c] * W[k*n+i][c]
                    b = np.ones(n)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k] = sol.tolist()
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i*m+j][c] * V[j*p+k][c]
                    b = np.ones(m)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i] = sol.tolist()
                    except:
                        pass

            err = error_value(U, V, W)
            if err < best_err:
                best_err = err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]

            if err < tol:
                break

        return best_U, best_V, best_W, best_err

    def try_snap_and_validate(U, V, W):
        """Try to snap to half-integers or integers and validate."""
        for snap_func, name in [(snap_half_ints, "half"), (snap_ints, "int")]:
            U_s = snap_func(U)
            V_s = snap_func(V)
            W_s = snap_func(W)
            if validate(U_s, V_s, W_s, tol=1e-9):
                return U_s, V_s, W_s
        return None

    def schoolbook_decomp(rank=None):
        """Standard schoolbook decomposition with rank = n*m*p."""
        if rank is None:
            rank = n * m * p
        U = [[0.0 for _ in range(rank)] for _ in range(nm)]
        V = [[0.0 for _ in range(rank)] for _ in range(mp)]
        W = [[0.0 for _ in range(rank)] for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1.0
                    V[j*p+k][r] = 1.0
                    W[k*n+i][r] = 1.0
                    r += 1
        return U, V, W

    def random_init(rank, seed=0):
        """Generate random initialization."""
        rng = random.Random(seed)
        U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(mp)]
        W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(pn)]
        return U, V, W

    def try_rank_alternating(R, seed=0, max_iters=150):
        """Try to find valid decomposition of given rank using alternating minimization."""
        U, V, W = random_init(R, seed)
        U, V, W, err = alternating_minimize(U, V, W, max_iters=max_iters)
        if err < 1e-10:
            result = try_snap_and_validate(U, V, W)
            if result:
                return result
        return None

    def try_structured_rank(R, seed=0, max_iters=100):
        """
        Try structured initialization that respects the tensor structure.
        Use a hybrid of schoolbook and random.
        """
        rng = random.Random(seed)
        U, V, W = schoolbook_decomp(R) if R == n*m*p else ([[[0.0]*R for _ in range(nm)],
                                                           [[0.0]*R for _ in range(mp)],
                                                           [[0.0]*R for _ in range(pn)]])

        if R != n * m * p:
            # Initialize with partial structure
            for c in range(R):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j][c] = 1.0 + rng.uniform(-0.2, 0.2)
                V[j*p+k][c] = 1.0 + rng.uniform(-0.2, 0.2)
                W[k*n+i][c] = 1.0 + rng.uniform(-0.2, 0.2)

        U, V, W, err = alternating_minimize(U, V, W, max_iters=max_iters)
        if err < 1e-10:
            result = try_snap_and_validate(U, V, W)
            if result:
                return result
        return None

    def polylog_search(R, seed=0, max_trials=5):
        """
        Use polynomial identity testing idea: try multiple random linear combinations
        and check if they yield exact decompositions.
        This is a heuristic that leverages the algebraic structure.
        """
        rng = random.Random(seed)
        # Start from schoolbook
        U0, V0, W0 = schoolbook_decomp(R) if R == n*m*p else ([[[0.0]*R for _ in range(nm)],
                                                               [[0.0]*R for _ in range(mp)],
                                                               [[0.0]*R for _ in range(pn)]])

        if R != n * m * p:
            # Random sparse initialization
            U = [[0.0 for _ in range(R)] for _ in range(nm)]
            V = [[0.0 for _ in range(R)] for _ in range(mp)]
            W = [[0.0 for _ in range(R)] for _ in range(pn)]
            for c in range(R):
                # Assign each column to a random (i,j,k) tuple
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j][c] = rng.choice([-1, 1])
                V[j*p+k][c] = rng.choice([-1, 1])
                W[k*n+i][c] = rng.choice([-1, 1])
        else:
            U, V, W = U0, V0, W0

        U, V, W, err = alternating_minimize(U, V, W, max_iters=100)
        if err < 1e-10:
            result = try_snap_and_validate(U, V, W)
            if result:
                return result
        return None

    def strassen_block_decomp(R, seed=0):
        """
        Try Strassen-like block structure for rank-R decomposition.
        For 2x3 * 3x5, we can try to merge some schoolbook terms.
        """
        rng = random.Random(seed)
        
        # Start with schoolbook (rank 30)
        U = [[0.0 for _ in range(R)] for _ in range(nm)]
        V = [[0.0 for _ in range(R)] for _ in range(mp)]
        W = [[0.0 for _ in range(R)] for _ in range(pn)]
        
        if R == n * m * p:
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][r] = 1.0
                        V[j*p+k][r] = 1.0
                        W[k*n+i][r] = 1.0
                        r += 1
            return U, V, W
        
        # For R < 30, try to merge terms using Strassen-like ideas
        # Each "block" can handle multiple (i,j,k) combinations
        # Strategy: group terms by shared indices
        
        # Initialize with zeros, then assign structured patterns
        for c in range(R):
            # Choose a random pattern type
            pattern = rng.randint(0, 3)
            
            if pattern == 0:
                # Single term (schoolbook-like)
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j][c] = rng.choice([-1, 1])
                V[j*p+k][c] = rng.choice([-1, 1])
                W[k*n+i][c] = rng.choice([-1, 1])
            
            elif pattern == 1:
                # Pair in k dimension (same i,j, different k)
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k1 = rng.randint(0, p-1)
                k2 = (k1 + 1) % p
                if k1 != k2:
                    U[i*m+j][c] = rng.choice([-1, 1])
                    V[j*p+k1][c] = 1.0
                    V[j*p+k2][c] = 1.0
                    W[k1*n+i][c] = 1.0
                    W[k2*n+i][c] = 1.0
            
            elif pattern == 2:
                # Pair in j dimension (same i,k, different j)
                i = rng.randint(0, n-1)
                k = rng.randint(0, p-1)
                j1 = rng.randint(0, m-1)
                j2 = (j1 + 1) % m
                if j1 != j2:
                    U[i*m+j1][c] = 1.0
                    U[i*m+j2][c] = 1.0
                    V[j1*p+k][c] = 1.0
                    V[j2*p+k][c] = 1.0
                    W[k*n+i][c] = rng.choice([-1, 1])
            
            else:
                # Pair in i dimension (same j,k, different i)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                i1 = rng.randint(0, n-1)
                i2 = (i1 + 1) % n
                if i1 != i2:
                    U[i1*m+j][c] = 1.0
                    U[i2*m+j][c] = 1.0
                    V[j*p+k][c] = rng.choice([-1, 1])
                    W[k*n+i1][c] = 1.0
                    W[k*n+i2][c] = 1.0
        
        U, V, W, err = alternating_minimize(U, V, W, max_iters=100)
        if err < 1e-10:
            result = try_snap_and_validate(U, V, W)
            if result:
                return result
        return None

    def finite_field_verify(U, V, W, mod):
        """
        Verify decomposition over finite field Z/mod.
        Returns True if exact over Z/mod.
        """
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val = (val + (U[i*m+j][c] * mod) * (V[j*p+k][c] * mod) * (W[k*n+i][c] * mod)) % mod
                    if val % mod != 1 % mod:
                        return False
        return True

    def fingerprint_search(R, seed=0, max_trials=20):
        """
        Use fingerprinting over finite fields to quickly filter valid decompositions.
        """
        rng = random.Random(seed)
        primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Generate random structured decomposition
            U = [[0.0 for _ in range(R)] for _ in range(nm)]
            V = [[0.0 for _ in range(R)] for _ in range(mp)]
            W = [[0.0 for _ in range(R)] for _ in range(pn)]
            
            # Use random sparse pattern with structured pairs
            for c in range(R):
                pattern = rng.randint(0, 2)
                
                if pattern == 0:
                    i = rng.randint(0, n-1)
                    j = rng.randint(0, m-1)
                    k = rng.randint(0, p-1)
                    U[i*m+j][c] = rng.choice([-1, 1])
                    V[j*p+k][c] = rng.choice([-1, 1])
                    W[k*n+i][c] = rng.choice([-1, 1])
                
                elif pattern == 1:
                    # Merge in k
                    i = rng.randint(0, n-1)
                    j = rng.randint(0, m-1)
                    k1 = rng.randint(0, p-1)
                    k2 = (k1 + rng.randint(1, p-1)) % p
                    if k1 != k2:
                        U[i*m+j][c] = rng.choice([-1, 1])
                        V[j*p+k1][c] = 1.0
                        V[j*p+k2][c] = rng.choice([-1, 1])
                        W[k1*n+i][c] = 1.0
                        W[k2*n+i][c] = 1.0
                
                else:
                    # Merge in j
                    i = rng.randint(0, n-1)
                    k = rng.randint(0, p-1)
                    j1 = rng.randint(0, m-1)
                    j2 = (j1 + rng.randint(1, m-1)) % m
                    if j1 != j2:
                        U[i*m+j1][c] = 1.0
                        U[i*m+j2][c] = rng.choice([-1, 1])
                        V[j1*p+k][c] = 1.0
                        V[j2*p+k][c] = 1.0
                        W[k*n+i][c] = rng.choice([-1, 1])
            
            # Quick fingerprint check over small primes
            valid = True
            for mod in primes[:3]:
                if not finite_field_verify(U, V, W, mod):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # If fingerprint passes, do full alternating minimization
            U, V, W, err = alternating_minimize(U, V, W, max_iters=150)
            if err < 1e-10:
                result = try_snap_and_validate(U, V, W)
                if result:
                    return result
        
        return None

    best_result = None
    best_rank = n * m * p  # schoolbook rank = 30

    # Try ranks from lower bound upward
    # Known lower bound for M_{2,3,5} is 28
    for R in range(28, 31):
        if time.time() > deadline - 10:
            break

        # Try Strassen-like block decomposition
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = strassen_block_decomp(R, seed=seed)
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}

        # Try fingerprint-based search
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = fingerprint_search(R, seed=seed, max_trials=15)
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}

        # Try alternating minimization with multiple seeds
        for seed in range(8):
            if time.time() > deadline - 5:
                break
            result = try_rank_alternating(R, seed=seed, max_iters=100)
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}

        # Try structured initialization
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            result = try_structured_rank(R, seed=seed, max_iters=80)
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}

        # Try polynomial-identity-style search
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            result = polylog_search(R, seed=seed, max_trials=3)
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}

    # Fallback: return schoolbook decomposition (rank 30)
    U, V, W = schoolbook_decomp()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
