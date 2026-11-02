"""Combined approach: structured search for rank-minimal 3x5x6 matmul decomposition."""
import random
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Schoolbook baseline (always valid, rank 90)
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p, best_rank)

    # Strategy: try structured decompositions
    # 1. Try to find a decomposition by solving the tensor equation
    #    For rank r, we need U (nm x r), V (mp x r), W (pn x r)
    #    such that sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all (i,j,k)

    # 2. Use a "split" approach: decompose into blocks and use known small-matrix decompositions
    # 3. Try random initialization + gradient-free search for specific ranks

    # First, try a few known structured approaches
    # Approach A: Use 2x2 Strassen for sub-blocks
    # Approach B: Direct search with verification

    # Let's try ranks from a reasonable lower bound upward
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But that's not tight. A better bound: rank >= (n*m*p) / max(n,m,p) = 90/6 = 15
    # Even better: use the "slice rank" or other bounds
    # For 3x5x6, let's try ranks from 25 to 89

    # Try a few specific structured decompositions first
    # Then do random search for lower ranks

    # Try rank 45 (half of schoolbook) with structured patterns
    # Try rank 60, 75, etc.

    # Actually, let's try a smarter approach:
    # Use the fact that the tensor is "all ones" to find sparse decompositions

    # Method: For a given rank r, try to find U, V, W by:
    # 1. Initialize randomly
    # 2. Use alternating minimization or direct solve
    # 3. Verify

    # Let's try a few ranks with different strategies

    # Strategy 1: Try to use Kronecker product structure
    # If we can write the 3x5x6 tensor as a sum of rank-1 tensors with structure

    # Strategy 2: Try small rank decompositions via linear algebra
    # For rank r, the condition is:
    # For each (i,k): the vector [sum_j U[i*m+j,c]*V[j*p+k,c]]_c must satisfy
    # sum_c W[k*n+i,c] * (that vector)_c = 1 for all j... no wait.

    # The correct condition is:
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for ALL (i,j,k)

    # This is equivalent to: for each c, define matrices A_c (n x m), B_c (m x p), C_c (p x n)
    # where A_c[i,j] = U[i*m+j,c], B_c[j,k] = V[j*p+k,c], C_c[k,i] = W[k*n+i,c]
    # Then sum_c C_c[k,i] * A_c[i,j] * B_c[j,k] = 1 for all (i,j,k)

    # Let's try random search for a few ranks with a time budget

    ranks_to_try = list(range(30, 90, 2))  # Try even ranks from 30 to 88
    random.seed(42)

    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            break

        # Try a few random trials for this rank
        success = _try_rank_random(n, m, p, r, max_trials=200, time_budget=deadline - 5)
        if success:
            best_rank = r
            # Keep searching for lower ranks
            continue

    # Also try some specific structured ranks
    for r in [30, 32, 35, 40, 45, 50, 55, 60]:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        success = _try_rank_structured(n, m, p, r, time_budget=deadline - 5)
        if success:
            best_rank = r

    return best_result


def _schoolbook(n, m, p, rank):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _check_decomp(n, m, p, U, V, W, r, tol=1e-9):
    """Check if U,V,W of rank r exactly decompose the matmul tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += W[k * n + i][c] * U[i * m + j][c] * V[j * p + k][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank_random(n, m, p, r, max_trials=100, time_budget=None):
    """Try to find a decomposition of rank r via random search with half-integer entries."""
    global best_rank, best_result

    if time_budget is None:
        time_budget = time.time() + 30

    # Use a small set of half-integer values
    vals = [0, 0.5, 1, -0.5, -1, 0.25, -0.25, 2, -2]

    for trial in range(max_trials):
        if time.time() > time_budget:
            break

        # Generate random factors with sparse structure
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))

        # Make each factor somewhat sparse
        density = 0.3
        for row in range(n * m):
            for col in range(r):
                if random.random() < density:
                    U[row, col] = random.choice(vals)
        for row in range(m * p):
            for col in range(r):
                if random.random() < density:
                    V[row, col] = random.choice(vals)
        for row in range(p * n):
            for col in range(r):
                if random.random() < density:
                    W[row, col] = random.choice(vals)

        # Check
        if _check_decomp(n, m, p, U.tolist(), V.tolist(), W.tolist(), r):
            result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if r < best_rank:
                best_rank = r
                best_result = result
            return True

    return False


def _try_rank_structured(n, m, p, r, time_budget=None):
    """Try structured decompositions using block patterns."""
    global best_rank, best_result

    if time_budget is None:
        time_budget = time.time() + 30

    # Try a few structured patterns
    # Pattern 1: Block diagonal with 2x2 Strassen-like blocks
    # Pattern 2: Cyclic patterns

    # For now, try a simple structured approach:
    # Divide into blocks and use known small decompositions

    # This is a placeholder for more sophisticated structured search
    # In practice, we'd try:
    # 1. 2x2x2 Strassen (rank 7) for sub-blocks
    # 2. Combine blocks

    # For 3x5x6, one approach:
    # Split 3 = 1+2, 5 = 2+3, 6 = 2+4
    # Use Strassen for 2x2x2 blocks

    # This is complex, so let's just return False for now
    # and rely on the random search

    return False
# EVOLVE-BLOCK-END
