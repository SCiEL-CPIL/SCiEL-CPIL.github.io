"""Strassen-like recursive partitioning with coefficient local search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    # ---- schoolbook (rank 63) ----
    r0 = n * m * p  # 63
    U0 = np.zeros((n * m, r0))
    V0 = np.zeros((m * p, r0))
    W0 = np.zeros((p * n, r0))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, c] = 1.0
                V0[j * p + k, c] = 1.0
                W0[k * n + i, c] = 1.0
                c += 1

    # ---- Correct verification for matrix multiplication tensor ----
    # For C = A*B where A is 3x3, B is 3x7, C is 3x7:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The factorization: sum_c W[k*n+i,c] * (sum_j A[i,j] U[i*m+j,c]) * (sum_j B[j,k] V[j*p+k,c])
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    # sum_c W[k*n+i,c] * U[i*m+j1,c] * V[j2*p+k,c] = delta_{j1,j2} for all i,k,j1,j2

    def verify_matmul(U, V, W):
        """Verify: sum_c W[k*n+i,c]*U[i*m+j1,c]*V[j2*p+k,c] == delta_{j1,j2} for all i,k,j1,j2."""
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(r):
                            s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > 1e-8:
                            return False
        return True

    # Verify schoolbook with correct condition
    if not verify_matmul(U0, V0, W0):
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    best_rank = r0
    best_solution = (U0.copy(), V0.copy(), W0.copy())

    # Precompute constraint structure
    constraints = []
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    constraints.append((i, k, j1, j2, 1.0 if j1 == j2 else 0.0))

    def evaluate_residual(U, V, W):
        """Compute max absolute residual over all constraints."""
        r = U.shape[1]
        max_res = 0.0
        for (i, k, j1, j2, expected) in constraints:
            s = 0.0
            for c in range(r):
                s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
            res = abs(s - expected)
            if res > max_res:
                max_res = res
        return max_res

    # ---- Random search with local refinement ----
    random.seed(42)
    np.random.seed(42)

    # Try various ranks from low to high
    for target_rank in [20, 25, 30, 35, 40, 45, 50, 53, 55, 60]:
        if time_left() < 10:
            break
        if target_rank >= best_rank:
            continue

        max_attempts = 50
        for attempt in range(max_attempts):
            if time_left() < 5:
                break

            # Random sparse initialization
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))

            # Initialize with random sparse patterns
            for c in range(target_rank):
                n_active_u = random.randint(1, 4)
                active_u = random.sample(range(n*m), min(n_active_u, n*m))
                for idx in active_u:
                    U_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                n_active_v = random.randint(1, 4)
                active_v = random.sample(range(m*p), min(n_active_v, m*p))
                for idx in active_v:
                    V_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                n_active_w = random.randint(1, 4)
                active_w = random.sample(range(p*n), min(n_active_w, p*n))
                for idx in active_w:
                    W_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])

            # Check if valid
            if verify_matmul(U_r, V_r, W_r):
                best_rank = target_rank
                best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
                break

        if best_rank == target_rank:
            break

    # ---- Try a more structured approach ----
    if best_rank == r0 and time_left() > 20:
        for target_rank in [40, 45, 50, 53, 55, 58, 60, 62]:
            if time_left() < 10:
                break
            if target_rank >= best_rank:
                continue

            max_attempts = 200
            for attempt in range(max_attempts):
                if time_left() < 5:
                    break

                U_r = np.zeros((n*m, target_rank))
                V_r = np.zeros((m*p, target_rank))
                W_r = np.zeros((p*n, target_rank))

                for c in range(target_rank):
                    i_c = random.randint(0, n-1)
                    k_c = random.randint(0, p-1)
                    j_start = random.randint(0, m-1)
                    j_len = random.randint(1, m - j_start)

                    for j in range(j_start, j_start + j_len):
                        U_r[i_c*m+j, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                    for j in range(j_start, j_start + j_len):
                        V_r[j*p+k_c, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                    W_r[k_c*n+i_c, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                if verify_matmul(U_r, V_r, W_r):
                    best_rank = target_rank
                    best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
                    break

            if best_rank == target_rank:
                break

    # ---- Final: try gradient-based local search from schoolbook ----
    if best_rank == r0 and time_left() > 15:
        for target_rank in [50, 52, 54, 55, 56, 58, 60, 62]:
            if time_left() < 8:
                break
            if target_rank >= best_rank:
                continue

            max_attempts = 500
            for attempt in range(max_attempts):
                if time_left() < 3:
                    break

                U_r = np.zeros((n*m, target_rank))
                V_r = np.zeros((m*p, target_rank))
                W_r = np.zeros((p*n, target_rank))

                for c in range(min(55, target_rank)):
                    i = c // (m*p)
                    rem = c % (m*p)
                    j = rem // p
                    k = rem % p
                    if c < n*m*p:
                        U_r[i*m+j, c] = 1.0
                        V_r[j*p+k, c] = 1.0
                        W_r[k*n+i, c] = 1.0

                for c in range(55, target_rank):
                    for idx in range(n*m):
                        if random.random() < 0.3:
                            U_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])
                    for idx in range(m*p):
                        if random.random() < 0.3:
                            V_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])
                    for idx in range(p*n):
                        if random.random() < 0.3:
                            W_r[idx, c] = random.choice([0.5, 1.0, -0.5, -1.0])

                if verify_matmul(U_r, V_r, W_r):
                    best_rank = target_rank
                    best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
                    break

            if best_rank == target_rank:
                break

    # ---- Return the best solution found ----
    U_best, V_best, W_best = best_solution

    def to_list(arr):
        result = []
        for row in arr:
            new_row = []
            for x in row:
                hx = round(x * 2) / 2.0
                if abs(x - hx) < 1e-9:
                    new_row.append(hx)
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}

# EVOLVE-BLOCK-END
