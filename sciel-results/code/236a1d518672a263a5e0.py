"""Matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns: {'U', 'V', 'W'} with shapes (20, rank), (25, rank), (20, rank)
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    # Deadline guard: 150s wall-clock limit, use 115s for safety
    start_time = time.time()
    deadline = start_time + 115

    # Build schoolbook decomposition (rank = 100)
    schoolbook_rank = n * m * p  # 100
    U_school = [[0] * schoolbook_rank for _ in range(nm)]
    V_school = [[0] * schoolbook_rank for _ in range(mp)]
    W_school = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_rational(U, V, W):
        """Verify exact decomposition using rational arithmetic."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0, 1)
                    for r_idx in range(r):
                        s += Fraction(U[i * m + j][r_idx]) * Fraction(V[j * p + k][r_idx]) * Fraction(W[k * n + i][r_idx])
                    if s != Fraction(1, 1):
                        return False
        return True

    def try_random_decomposition(rank, num_trials=50):
        """Try to find a rank-r decomposition using random initialization."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None

            # Random integer/half-integer entries
            U = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(rank)] for _ in range(pn)]

            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}

        return None

    # Try structured decompositions first
    # One idea: use the fact that n=4, m=5, p=5
    # Try rank values that are factors or near factors

    # Let's try a greedy rank reduction approach:
    # Start with schoolbook, try to merge terms

    # Alternative: use known results or try specific structured decompositions

    # For now, let's try random search for ranks slightly below 100
    # This is unlikely to succeed for large reductions, but may find small improvements

    # Try ranks 99 down to 90
    for rank in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if time.time() > deadline:
            break

        result = try_random_decomposition(rank, num_trials=10)
        if result is not None:
            best_rank = rank
            best_result = result
            break

    # If no improvement found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
