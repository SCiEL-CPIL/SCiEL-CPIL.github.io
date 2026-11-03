import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook fallback (rank = n*m*p = 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    def is_half_integer(x, tol=1e-6):
        """Check if x is close to an integer or half-integer."""
        int_part = round(x)
        half_int_part = round(2 * x) / 2.0
        return abs(x - int_part) < tol or abs(x - half_int_part) < tol

    def to_exact_fraction(x):
        """Convert a float that should be integer or half-integer to a Fraction."""
        if abs(x - round(x)) < 1e-6:
            return Fraction(int(round(x)))
        if abs(2 * x - round(2 * x)) < 1e-6:
            return Fraction(int(round(2 * x)), 2)
        return Fraction(x).limit_denominator(1000)

    def verify_decomposition(U, V, W, r, tol=1e-6):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c] for c in range(r))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_exact(U_frac, V_frac, W_frac, r):
        """Verify the decomposition exactly using Fractions."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c] for c in range(r))
                    if val != 1:
                        return False
        return True

    def try_random_rank(r, max_attempts=500, entry_range=(-1, 2)):
        """Try to find a rank-r decomposition using random integer U, V and solving for W."""
        if time.time() > deadline - 5:
            return None

        lo, hi = entry_range

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None

            # Random U and V with small integer entries
            U = np.random.randint(lo, hi + 1, size=(nm, r))
            V = np.random.randint(lo, hi + 1, size=(mp, r))

            W = np.zeros((pn, r), dtype=float)
            valid = True

            for i in range(n):
                for k in range(p):
                    # Build m x r matrix A_sys where A_sys[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A_sys = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_sys[j, c] = U[i * m + j, c] * V[j * p + k, c]

                    b_sys = np.ones(m)

                    if r == m:
                        det = np.linalg.det(A_sys)
                        if abs(det) < 1e-9:
                            valid = False
                            break
                        try:
                            W_row = np.linalg.solve(A_sys, b_sys)
                        except:
                            valid = False
                            break
                    elif r > m:
                        # Use least squares
                        try:
                            W_row = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                        except:
                            valid = False
                            break
                    else:
                        # r < m: overdetermined, check consistency
                        try:
                            W_row, residuals, rank, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                            if len(residuals) > 0 and residuals[0] > 1e-6:
                                valid = False
                                break
                        except:
                            valid = False
                            break

                    W[k * n + i, :] = W_row

                if not valid:
                    break

            if not valid:
                continue

            # Check if W entries are close to integer or half-integer
            entries_ok = True
            for row in W:
                for x in row:
                    if not is_half_integer(x, 1e-5):
                        entries_ok = False
                        break
                if not entries_ok:
                    break

            if not entries_ok:
                continue

            # Verify the decomposition
            if not verify_decomposition(U.tolist(), V.tolist(), W.tolist(), r, 1e-5):
                continue

            # Convert to exact fractions
            U_frac = [[to_exact_fraction(x) for x in row] for row in U]
            V_frac = [[to_exact_fraction(x) for x in row] for row in V]
            W_frac = [[to_exact_fraction(x) for x in row] for row in W]

            # Verify exactly
            if verify_exact(U_frac, V_frac, W_frac, r):
                return {"U": U_frac, "V": V_frac, "W": W_frac}

        return None

    def try_winograd_rank(r, max_attempts=200):
        """
        Try Winograd-style structured decomposition.
        For 2x3x7, we can try to exploit structure.
        This is a heuristic approach using small structured matrices.
        """
        if time.time() > deadline - 5:
            return None
        
        # Try a few specific structured patterns
        # For simplicity, we'll try random small integer matrices with some structure
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Create sparse random matrices
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            
            for c in range(r):
                # Each column has a few non-zero entries
                num_nonzero = np.random.randint(1, 4)
                U_indices = np.random.choice(nm, num_nonzero, replace=False)
                U[U_indices, c] = np.random.choice([-1, 1, 2], num_nonzero)
                
                V_indices = np.random.choice(mp, num_nonzero, replace=False)
                V[V_indices, c] = np.random.choice([-1, 1, 2], num_nonzero)
            
            W = np.zeros((pn, r), dtype=float)
            valid = True

            for i in range(n):
                for k in range(p):
                    A_sys = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_sys[j, c] = U[i * m + j, c] * V[j * p + k, c]

                    b_sys = np.ones(m)

                    if r == m:
                        det = np.linalg.det(A_sys)
                        if abs(det) < 1e-9:
                            valid = False
                            break
                        try:
                            W_row = np.linalg.solve(A_sys, b_sys)
                        except:
                            valid = False
                            break
                    elif r > m:
                        try:
                            W_row = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                        except:
                            valid = False
                            break
                    else:
                        try:
                            W_row, residuals, rank, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                            if len(residuals) > 0 and residuals[0] > 1e-6:
                                valid = False
                                break
                        except:
                            valid = False
                            break

                    W[k * n + i, :] = W_row

                if not valid:
                    break

            if not valid:
                continue

            entries_ok = True
            for row in W:
                for x in row:
                    if not is_half_integer(x, 1e-5):
                        entries_ok = False
                        break
                if not entries_ok:
                    break

            if not entries_ok:
                continue

            if not verify_decomposition(U.tolist(), V.tolist(), W.tolist(), r, 1e-5):
                continue

            U_frac = [[to_exact_fraction(x) for x in row] for row in U]
            V_frac = [[to_exact_fraction(x) for x in row] for row in V]
            W_frac = [[to_exact_fraction(x) for x in row] for row in W]

            if verify_exact(U_frac, V_frac, W_frac, r):
                return {"U": U_frac, "V": V_frac, "W": W_frac}

        return None

    best_result = None
    best_rank = rank_sb

    # Try ranks from lower to upper
    # The theoretical lower bound for 2x3x7 is not well established, but let's try to find something better than 42.
    # Start from a reasonable lower bound and go up.
    
    # First, try some structured approaches with higher ranks to see if we can get below 42
    # Then try random search for lower ranks
    
    # Try random search for ranks 30-41
    for r in range(30, best_rank):
        if time.time() > deadline - 10:
            break

        # Adjust attempts based on rank
        if r <= 35:
            max_attempts = 300
        elif r <= 40:
            max_attempts = 150
        else:
            max_attempts = 50

        result = try_random_rank(r, max_attempts=max_attempts, entry_range=(-1, 2))
        
        if result is not None:
            best_rank = r
            best_result = result
            # If we found a good rank, we can try to go lower
            continue
    
    # Also try Winograd-style structured search for ranks 35-41
    for r in range(35, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_winograd_rank(r, max_attempts=100)
        if result is not None:
            best_rank = r
            best_result = result

    if best_result is not None:
        U_best = best_result["U"]
        V_best = best_result["V"]
        W_best = best_result["W"]
        # Convert to lists for the return interface
        return {"U": U_best, "V": V_best, "W": W_best}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
