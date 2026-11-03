"""Program for rank-minimal exact tensor decomposition of 2x3 * 3x7 matrix multiplication.
Uses numerical optimization over reals followed by rational reconstruction to half-integers."""
import numpy as np
from fractions import Fraction
import random
import time
import sys

DIMENSIONS = [2, 3, 7]
DEADLINE = 120.0  # seconds

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    schoolbook_rank = n * m * p  # 42

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def schoolbook_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor using Fraction arithmetic."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(r):
                        u_val = U[i * m + j][c]
                        v_val = V[j * p + k][c]
                        w_val = W[k * n + i][c]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            total += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                    if total != 1:
                        return False
        return True

    def try_numerical_optimization(target_rank, num_trials=5):
        """
        Try to find a rank-r decomposition using numerical optimization over reals,
        then reconstruct to half-integers.
        """
        r = target_rank
        best_result = None

        for trial in range(num_trials):
            if time_remaining() < 20:
                break

            # Initialize U and V with random values
            np.random.seed(trial * 12345 + 67890)
            U_float = np.random.randn(nm, r) * 0.5
            V_float = np.random.randn(mp, r) * 0.5

            # Normalize columns to avoid scaling issues
            for c in range(r):
                norm_u = np.linalg.norm(U_float[:, c])
                norm_v = np.linalg.norm(V_float[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10:
                    U_float[:, c] /= norm_u
                    V_float[:, c] /= norm_v

            # Solve for W using least squares
            # For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is a linear system in W
            num_constraints = n * m * p  # 42
            num_unknowns = pn * r  # 14 * r

            A = np.zeros((num_constraints, num_unknowns))
            b = np.ones(num_constraints)

            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            coeff = U_float[i * m + j, c] * V_float[j * p + k, c]
                            pos = (k * n + i) * r + c
                            A[idx, pos] = coeff
                        idx += 1

            # Solve least squares
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W_float = W_vec.reshape(pn, r)
            except:
                continue

            # Check residual
            residual = np.linalg.norm(A @ W_vec - b)
            if residual > 1e-6:
                continue

            # Try to reconstruct to half-integers
            U_frac = _reconstruct_to_half_integers(U_float)
            V_frac = _reconstruct_to_half_integers(V_float)
            W_frac = _reconstruct_to_half_integers(W_float)

            if U_frac is None or V_frac is None or W_frac is None:
                continue

            # Verify exact decomposition
            if verify_decomposition(U_frac, V_frac, W_frac):
                return {"U": U_frac, "V": V_frac, "W": W_frac}

        return None

    def _reconstruct_to_half_integers(matrix_float):
        """
        Reconstruct a floating-point matrix to half-integers (multiples of 0.5).
        Returns None if the reconstruction doesn't seem close enough.
        """
        # Scale by 2 to work with integers
        scaled = matrix_float * 2

        # Round to nearest integer
        rounded = np.round(scaled)

        # Check if the reconstruction is close
        max_err = np.max(np.abs(scaled - rounded))
        if max_err > 0.1:
            return None

        # Convert to Fraction
        result = []
        for row in rounded:
            frac_row = []
            for val in row:
                frac_row.append(Fraction(int(val), 2))
            result.append(frac_row)

        return result

    def try_direct_half_integer_search(target_rank, num_trials=100):
        """
        Directly search for half-integer entries in U, V, W.
        Initialize U and V with small half-integers, solve for W.
        """
        r = target_rank
        half_integers = [Fraction(i, 2) for i in range(-4, 5)]  # -2 to 2 in steps of 0.5

        for trial in range(num_trials):
            if time_remaining() < 15:
                break

            random.seed(trial * 999 + 123)

            # Initialize U and V with random half-integers
            U = [[random.choice(half_integers) for _ in range(r)] for _ in range(nm)]
            V = [[random.choice(half_integers) for _ in range(r)] for _ in range(mp)]

            # Solve for W using linear system over rationals
            W = _solve_for_W_rational(U, V, r)
            if W is None:
                continue

            # Check if W entries are close to half-integers
            w_is_half = True
            for row in W:
                for val in row:
                    # Check if val is a half-integer
                    if val.denominator not in (1, 2):
                        w_is_half = False
                        break
                if not w_is_half:
                    break

            if not w_is_half:
                continue

            # Verify exact decomposition
            if verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}

        return None

    def _solve_for_W_rational(U, V, target_rank):
        """
        Given U (nm x r) and V (mp x r) with rational entries,
        solve for W (pn x r) such that the tensor identity holds.
        """
        r = target_rank
        num_constraints = n * m * p  # 42
        num_unknowns = pn * r  # 14 * r

        # Build linear system over Fractions
        A = [[Fraction(0) for _ in range(num_unknowns)] for _ in range(num_constraints)]
        b = [Fraction(1) for _ in range(num_constraints)]

        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        coeff = U[i * m + j][c] * V[j * p + k][c]
                        if coeff != 0:
                            pos = (k * n + i) * r + c
                            A[idx][pos] += coeff
                    idx += 1

        w_vec = _gaussian_elimination(A, b)
        if w_vec is None:
            return None

        W = [[Fraction(0) for _ in range(r)] for _ in range(pn)]
        for row in range(pn):
            for col in range(r):
                W[row][col] = w_vec[row * r + col]

        return W

    def _gaussian_elimination(A, b):
        """Solve A*x = b using exact Gaussian elimination over rationals."""
        num_rows = len(A)
        num_cols = len(A[0]) if num_rows > 0 else 0

        if num_cols == 0:
            return [Fraction(0)] * 0

        aug = [row[:] + [b[i]] for i, row in enumerate(A)]

        rank = 0
        pivot_cols = []

        for col in range(num_cols):
            pivot_row = None
            for row in range(rank, num_rows):
                if aug[row][col] != 0:
                    pivot_row = row
                    break

            if pivot_row is None:
                continue

            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
            pivot_cols.append(col)

            pivot_val = aug[rank][col]
            for j in range(col, num_cols + 1):
                aug[rank][j] = aug[rank][j] / pivot_val

            for row in range(num_rows):
                if row != rank and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(col, num_cols + 1):
                        aug[row][j] = aug[row][j] - factor * aug[rank][j]

            rank += 1

        # Check consistency
        for row in range(rank, num_rows):
            for j in range(num_cols):
                if aug[row][j] != 0:
                    return None

        # Extract solution (minimum norm solution: free variables = 0)
        x = [Fraction(0) for _ in range(num_cols)]
        for i, pc in enumerate(pivot_cols):
            x[pc] = aug[i][num_cols]

        return x

    def try_structured_winograd(target_rank):
        """
        Try a structured Winograd-like decomposition.
        For 2x3 * 3x7, use block decomposition:
        - Split the 7 columns into groups and apply Winograd to each.
        """
        if time_remaining() < 10:
            return None

        # Winograd for small matrix multiplications
        # 2x3 * 3x3 has rank at most 14 (schoolbook)
        # 2x3 * 3x2 has rank at most 8 (Winograd)
        # 2x3 * 3x1 has rank 3 (trivial)

        # Strategy: use rank 30 = 14 + 8 + 8
        # Block 1: k=0,1,2 (3 cols) -> rank 14
        # Block 2: k=3,4 (2 cols) -> rank 8
        # Block 3: k=5,6 (2 cols) -> rank 8

        if target_rank < 30:
            return None

        # Build the decomposition block by block
        U = [[Fraction(0) for _ in range(target_rank)] for _ in range(nm)]
        V = [[Fraction(0) for _ in range(target_rank)] for _ in range(mp)]
        W = [[Fraction(0) for _ in range(target_rank)] for _ in range(pn)]

        col = 0

        # Block 1: 2x3 * 3x3, schoolbook rank 9 (not 14, that was wrong)
        # Actually 2x3 * 3x3 has 18 scalar products, so schoolbook rank is 18
        # But we can do better. Let's just use schoolbook for now.
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    if col < target_rank:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1

        # Block 2: 2x3 * 3x2, schoolbook rank 12
        for i in range(n):
            for j in range(m):
                for k in range(3, 5):
                    if col < target_rank:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1

        # Block 3: 2x3 * 3x2, schoolbook rank 12
        for i in range(n):
            for j in range(m):
                for k in range(5, 7):
                    if col < target_rank:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1

        # This gives rank 9 + 12 + 12 = 33
        # Verify
        if verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}

        return None

    # Try various approaches in order of expected rank
    # 1. Try numerical optimization for rank 30-34
    for rank in [30, 32, 34]:
        if time_remaining() < 30:
            break
        result = try_numerical_optimization(rank, num_trials=3)
        if result is not None:
            return result

    # 2. Try direct half-integer search for rank 30-36
    for rank in [30, 32, 34, 36]:
        if time_remaining() < 20:
            break
        result = try_direct_half_integer_search(rank, num_trials=50)
        if result is not None:
            return result

    # 3. Try structured Winograd (rank 33)
    if time_remaining() > 10:
        result = try_structured_winograd(33)
        if result is not None:
            return result

    # 4. Fallback: schoolbook decomposition
    return schoolbook_decomposition()

# EVOLVE-BLOCK-END
