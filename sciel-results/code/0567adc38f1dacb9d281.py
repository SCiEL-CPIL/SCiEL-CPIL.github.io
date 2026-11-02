import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin before 150s limit

    n, m, p = 2, 3, 3
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)
    # W: (p*n, rank) = (6, rank)

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        n, m, p = 2, 3, 3
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        for i in range(n):
            for k in range(p):
                for j in range(m):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                    target = 1.0 if i == k else 0.0
                    if abs(s - target) > 1e-9:
                        return False
        return True

    def error(U, V, W):
        """Calculate the squared error."""
        n, m, p = 2, 3, 3
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        total_err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                    target = 1.0 if i == k else 0.0
                    total_err += (s - target) ** 2
        return total_err

    def schoolbook():
        rank = n * m * p  # 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    fallback = schoolbook()

    # Values to consider in local search
    VALUES = [-1, -0.5, 0, 0.5, 1]

    def random_decomp(rank, density=0.5, seed=None):
        if seed is not None:
            random.seed(seed)
        def gen_matrix(rows):
            mat = [[0] * rank for _ in range(rows)]
            for r in range(rows):
                for c in range(rank):
                    if random.random() < density:
                        mat[r][c] = random.choice([v for v in VALUES if v != 0])
            return mat

        U = gen_matrix(n * m)
        V = gen_matrix(m * p)
        W = gen_matrix(p * n)
        return U, V, W

    def local_search(U, V, W, max_iter=500, deadline=None):
        """
        Hill climbing with row-wise updates.
        """
        rank = len(U[0])
        err = error(U, V, W)
        if err < 1e-9:
            return U, V, W, True

        for it in range(max_iter):
            if deadline and time.time() > deadline:
                break

            # Pick a random matrix and row
            which = random.randint(0, 2)
            if which == 0:
                idx = random.randint(0, n * m - 1)
                mat = U
            elif which == 1:
                idx = random.randint(0, m * p - 1)
                mat = V
            else:
                idx = random.randint(0, p * n - 1)
                mat = W

            # Try to improve the row by changing one entry at a time
            # We iterate through columns and try all values
            best_val = mat[idx][:]
            best_err = err

            for c in range(rank):
                old_val = mat[idx][c]
                for new_val in VALUES:
                    if new_val == old_val:
                        continue
                    mat[idx][c] = new_val
                    new_err = error(U, V, W)
                    if new_err < best_err:
                        best_err = new_err
                        best_val[c] = new_val
                mat[idx][c] = old_val # Restore

            # Apply the best improvement found for this row
            if best_err < err:
                for c in range(rank):
                    mat[idx][c] = best_val[c]
                err = best_err
                if err < 1e-9:
                    return U, V, W, True
            else:
                # If no improvement, try a random move to escape local minima
                c = random.randint(0, rank - 1)
                old_val = mat[idx][c]
                mat[idx][c] = random.choice(VALUES)
                new_err = error(U, V, W)
                # Accept with some probability if worse (simulated annealing lite)
                if new_err < err or (err > 0.1 and random.random() < 0.1):
                    err = new_err
                    if err < 1e-9:
                        return U, V, W, True
                else:
                    # Revert
                    mat[idx][c] = old_val

        return U, V, W, False

    def structured_search(rank):
        """
        Try to find a structured solution using the fact that the tensor has some symmetry.
        We try to exploit the fact that n=2, m=3, p=3.
        """
        # This is a heuristic search for structured solutions
        # We try different patterns of non-zero entries
        for seed in range(20):
            if time.time() > DEADLINE - 10:
                return None
            random.seed(seed + 1000)
            U, V, W = random_decomp(rank, density=0.6, seed=seed)
            U, V, W, success = local_search(U, V, W, max_iter=300, deadline=DEADLINE - 10)
            if success and verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        return None

    # Try to find rank 15
    best_result = None

    for target_rank in [15, 16, 17]:
        if time.time() > DEADLINE - 10:
            break

        found = False
        max_restarts = 100 if target_rank == 15 else 50
        restarts = 0

        while restarts < max_restarts and not found:
            if time.time() > DEADLINE - 10:
                break

            # Vary density
            density = 0.3 + random.random() * 0.4
            U, V, W = random_decomp(target_rank, density=density, seed=restarts)

            # Run local search
            U, V, W, success = local_search(U, V, W, max_iter=100, deadline=DEADLINE - 10)

            if success:
                if verify(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    found = True
                    break
            restarts += 1

        if found:
            break

    if best_result is None:
        # Try structured search
        for target_rank in [15, 16]:
            if time.time() > DEADLINE - 5:
                break
            res = structured_search(target_rank)
            if res is not None:
                best_result = res
                break

    if best_result is not None:
        return best_result

    return fallback
