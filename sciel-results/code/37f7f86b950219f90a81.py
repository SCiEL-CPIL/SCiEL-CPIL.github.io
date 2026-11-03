import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Precompute target slices: target_slices[(i_prime, j)] is (n*p) x (m*p)
    target_slices = {}
    for i_prime in range(n):
        for j in range(m):
            mat = np.zeros((n * p, m * p))
            for i in range(n):
                if i == i_prime:
                    for k in range(p):
                        for j_prime in range(m):
                            if j_prime == j:
                                mat[i * p + k, j_prime * p + k] = 1.0
            target_slices[(i_prime, j)] = mat

    def compute_error(U, V, W, rank):
        """Compute squared Frobenius norm of the difference."""
        err = 0.0
        for i_prime in range(n):
            for j in range(m):
                computed = np.zeros((n * p, m * p))
                for t in range(rank):
                    val = V[i_prime * m + j, t]
                    if val != 0:
                        computed += val * np.outer(U[:, t], W[:, t])
                target = target_slices[(i_prime, j)]
                diff = computed - target
                err += np.sum(diff * diff)
        return err

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * p, rank_school))
    V_school = np.zeros((n * m, rank_school))
    W_school = np.zeros((m * p, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * p + k, col] = 1
                V_school[i * m + j, col] = 1
                W_school[j * p + k, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find lower rank decompositions using local search
    # Start from rank 30 (a reasonable lower bound)
    for rank in range(30, best_rank + 1):
        if time.time() > deadline:
            break

        found = False
        max_attempts = 50 if rank < 50 else 20

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Initialize with random half-integer values
            U = np.random.randint(-2, 3, size=(n * p, rank)) / 2.0
            V = np.random.randint(-2, 3, size=(n * m, rank)) / 2.0
            W = np.random.randint(-2, 3, size=(m * p, rank)) / 2.0

            current_error = compute_error(U, V, W, rank)

            # Local search: perturb one entry at a time
            max_steps = 300
            for step in range(max_steps):
                if time.time() > deadline:
                    break

                # Pick a random matrix and entry
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = U
                    rows = n * p
                elif mat_idx == 1:
                    mat = V
                    rows = n * m
                else:
                    mat = W
                    rows = m * p

                r_idx = np.random.randint(rows)
                c_idx = np.random.randint(rank)

                old_val = mat[r_idx, c_idx]
                improved = False
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    new_val = old_val + delta
                    new_val = round(new_val * 2) / 2
                    mat[r_idx, c_idx] = new_val
                    new_error = compute_error(U, V, W, rank)
                    if new_error < current_error:
                        current_error = new_error
                        improved = True
                        break
                    mat[r_idx, c_idx] = old_val

                if not improved:
                    # Try a few more random entries before giving up
                    if step % 50 == 49:
                        # Reset to a new random initialization
                        U = np.random.randint(-2, 3, size=(n * p, rank)) / 2.0
                        V = np.random.randint(-2, 3, size=(n * m, rank)) / 2.0
                        W = np.random.randint(-2, 3, size=(m * p, rank)) / 2.0
                        current_error = compute_error(U, V, W, rank)

            if current_error < 1e-10:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break

        if found:
            break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
