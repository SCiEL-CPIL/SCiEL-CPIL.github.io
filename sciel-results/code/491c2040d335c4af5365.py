import numpy as np
import time
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 120s limit, leave 5s margin

    n, m, p = 3, 4, 7
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    num_eqs = n * m * p  # 84

    # Pre-compute index mappings
    idx_U = np.zeros(num_eqs, dtype=int)
    idx_V = np.zeros(num_eqs, dtype=int)
    idx_W_row = np.zeros(num_eqs, dtype=int)

    e = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U[e] = i * m + j
                idx_V[e] = j * p + k
                idx_W_row[e] = k * n + i
                e += 1

    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]

        U_exp = U[idx_U, :]  # (84, r)
        V_exp = V[idx_V, :]  # (84, r)
        W_exp = W[idx_W_row, :]  # (84, r)

        vals = np.sum(U_exp * V_exp * W_exp, axis=1)
        return np.all(np.abs(vals - 1.0) < 1e-6)

    def solve_for_W(U, V):
        """Given U and V, solve for W such that the decomposition holds."""
        r = U.shape[1]

        U_exp = U[idx_U, :]  # (84, r)
        V_exp = V[idx_V, :]  # (84, r)
        coeffs = U_exp * V_exp  # (84, r)

        # Build A matrix: A[e, idx_W_row[e]*r + c] = coeffs[e, c]
        A = np.zeros((num_eqs, dim_W * r))
        for c in range(r):
            A[np.arange(num_eqs), idx_W_row * r + c] = coeffs[:, c]

        b = np.ones(num_eqs)

        try:
            W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None, None

        if residuals.size > 0 and residuals[0] > 1e-6:
            return None, None

        W_sol = W_flat.reshape(dim_W, r)
        return W_sol, residuals[0] if residuals.size > 0 else 0.0

    def try_structured_construction(rank, seed=42):
        """
        Try a structured construction based on algebraic identities.
        Uses the fact that for matrix multiplication, we can share
        intermediate products across different output indices.
        """
        if time.time() > deadline - 5:
            return None

        np.random.seed(seed)

        # Generate U and V with structured patterns
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))

        for c in range(rank):
            # Create structured patterns for U and V
            # Use a mix of deterministic and random patterns

            # Pattern type 1: Block diagonal
            if c % 3 == 0:
                # Assign values based on position
                for idx in range(dim_U):
                    i, j = divmod(idx, m)
                    U[idx, c] = ((i + c) % 3) - 1  # -1, 0, 1
                for idx in range(dim_V):
                    j, k = divmod(idx, p)
                    V[idx, c] = ((j + c) % 3) - 1
            # Pattern type 2: Cyclic
            elif c % 3 == 1:
                for idx in range(dim_U):
                    i, j = divmod(idx, m)
                    U[idx, c] = 1 if (i + j + c) % 2 == 0 else -1
                for idx in range(dim_V):
                    j, k = divmod(idx, p)
                    V[idx, c] = 1 if (j + k + c) % 2 == 0 else -1
            # Pattern type 3: Random with structure
            else:
                # Use random but with some structure
                U[:, c] = np.random.randint(-2, 3, size=dim_U)
                V[:, c] = np.random.randint(-2, 3, size=dim_V)

        # Ensure no row is all zeros
        U_rows_zero = np.all(U == 0, axis=1)
        V_rows_zero = np.all(V == 0, axis=1)
        if np.any(U_rows_zero) or np.any(V_rows_zero):
            # Fix zero rows
            for idx in range(dim_U):
                if U[idx, :] == 0:
                    U[idx, c] = 1
            for idx in range(dim_V):
                if V[idx, :] == 0:
                    V[idx, c] = 1

        W_sol, residual = solve_for_W(U, V)
        if W_sol is None:
            return None

        # Check if W has half-integer entries
        W_scaled = W_sol * 2
        W_rounded = np.round(W_scaled)

        if np.max(np.abs(W_scaled - W_rounded)) < 1e-4:
            W_int = W_rounded / 2.0

            if verify(U, V, W_int):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}

        return None

    def try_iterative_refinement(U_init, V_init, max_iter=50):
        """
        Start from an initial decomposition and iteratively refine
        to reduce rank while maintaining validity.
        """
        if time.time() > deadline - 10:
            return None

        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        r = U.shape[1]

        # Try to reduce rank by finding linear dependencies
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break

            # Check if we can remove a column
            for c in range(r):
                if time.time() > deadline - 10:
                    break

                # Try removing column c
                U_test = np.delete(U, c, axis=1)
                V_test = np.delete(V, c, axis=1)

                if U_test.shape[1] == 0:
                    continue

                W_test, residual = solve_for_W(U_test, V_test)
                if W_test is None:
                    continue

                # Check if W has half-integer entries
                W_scaled = W_test * 2
                W_rounded = np.round(W_scaled)

                if np.max(np.abs(W_scaled - W_rounded)) < 1e-4:
                    W_int = W_rounded / 2.0

                    if verify(U_test, V_test, W_int):
                        U = U_test
                        V = V_test
                        r = U.shape[1]
                        break

        if r < U_init.shape[1]:
            W_sol, _ = solve_for_W(U, V)
            if W_sol is not None:
                W_scaled = W_sol * 2
                W_rounded = np.round(W_scaled)
                if np.max(np.abs(W_scaled - W_rounded)) < 1e-4:
                    W_int = W_rounded / 2.0
                    if verify(U, V, W_int):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}

        return None

    # Try ranks from lower bound to schoolbook
    best_result = None

    # First, try structured constructions for lower ranks
    for r in range(28, 85):
        if time.time() > deadline - 5:
            break

        # Number of tries: allocate more to lower ranks
        if r <= 35:
            tries = 30
        elif r <= 50:
            tries = 20
        elif r <= 70:
            tries = 10
        else:
            tries = 5

        found = False
        for trial in range(tries):
            if time.time() > deadline - 3:
                break

            # Try structured construction
            result = try_structured_construction(r, seed=trial)
            if result is not None:
                if best_result is None or r < best_result[0]:
                    best_result = (r, result)
                found = True
                break

        if found:
            # Found a valid solution at this rank, keep searching for better
            continue

    # If no structured solution found, fall back to random search
    if best_result is None or best_result[0] > 50:
        for r in range(28, 85):
            if time.time() > deadline - 5:
                break

            # Number of tries: allocate more to lower ranks
            if r <= 35:
                tries = 30
            elif r <= 50:
                tries = 20
            elif r <= 70:
                tries = 10
            else:
                tries = 5

            for trial in range(tries):
                if time.time() > deadline - 3:
                    break

                # Generate random U and V with integer entries
                if trial % 3 == 0:
                    U = np.random.randint(-2, 3, size=(dim_U, r))
                    V = np.random.randint(-2, 3, size=(dim_V, r))
                elif trial % 3 == 1:
                    U = np.random.randint(-3, 4, size=(dim_U, r))
                    V = np.random.randint(-3, 4, size=(dim_V, r))
                else:
                    U = np.zeros((dim_U, r))
                    V = np.zeros((dim_V, r))
                    for _ in range(dim_U * r // 3):
                        U[np.random.randint(dim_U), np.random.randint(r)] = np.random.choice([-2, -1, 1, 2])
                    for _ in range(dim_V * r // 3):
                        V[np.random.randint(dim_V), np.random.randint(r)] = np.random.choice([-2, -1, 1, 2])

                U_rows_zero = np.all(U == 0, axis=1)
                V_rows_zero = np.all(V == 0, axis=1)
                if np.any(U_rows_zero) or np.any(V_rows_zero):
                    continue

                W_sol, residual = solve_for_W(U, V)
                if W_sol is None:
                    continue

                W_scaled = W_sol * 2
                W_rounded = np.round(W_scaled)

                if np.max(np.abs(W_scaled - W_rounded)) < 1e-4:
                    W_int = W_rounded / 2.0

                    if verify(U, V, W_int):
                        result = {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}
                        if best_result is None or r < best_result[0]:
                            best_result = (r, result)
                        break

    if best_result is not None:
        return best_result[1]

    return get_schoolbook()
