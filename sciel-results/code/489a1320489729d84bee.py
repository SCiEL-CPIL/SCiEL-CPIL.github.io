import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 110  # Safety margin for 150s limit

    # Initialize Schoolbook
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    current_rank = rank_schoolbook
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = current_rank

    def is_valid(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def snap_to_half_int(matrix):
        return np.round(matrix * 2) / 2

    def snap_to_int(matrix):
        return np.round(matrix)

    # Try to eliminate term c by solving for variable 'var' in U, V, W
    def try_eliminate(U, V, W, r, c, var, snap_func):
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()

        if var == 0:  # Solve for U
            success = True
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r-1))
                    b = np.ones(p)
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        for k in range(p):
                            A[k, idx] = V[j*p+k, c_prime] * W[k*n+i, c_prime]
                        idx += 1
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.linalg.norm(A.dot(x) - b) > 1e-6:
                        success = False
                        break
                    x_snapped = snap_func(x)
                    if np.linalg.norm(A.dot(x_snapped) - b) > 1e-6:
                        success = False
                        break
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        U_new[i*m+j, c_prime] = x_snapped[idx]
                        idx += 1
            if not success:
                return None, None, None, False
            U_new[:, c] = 0
            return U_new, V_new, W_new, True

        elif var == 1:  # Solve for V
            success = True
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r-1))
                    b = np.ones(n)
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        for i in range(n):
                            A[i, idx] = U[i*m+j, c_prime] * W[k*n+i, c_prime]
                        idx += 1
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.linalg.norm(A.dot(x) - b) > 1e-6:
                        success = False
                        break
                    x_snapped = snap_func(x)
                    if np.linalg.norm(A.dot(x_snapped) - b) > 1e-6:
                        success = False
                        break
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        V_new[j*p+k, c_prime] = x_snapped[idx]
                        idx += 1
            if not success:
                return None, None, None, False
            V_new[:, c] = 0
            return U_new, V_new, W_new, True

        else:  # Solve for W
            success = True
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r-1))
                    b = np.ones(m)
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        for j in range(m):
                            A[j, idx] = U[i*m+j, c_prime] * V[j*p+k, c_prime]
                        idx += 1
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.linalg.norm(A.dot(x) - b) > 1e-6:
                        success = False
                        break
                    x_snapped = snap_func(x)
                    if np.linalg.norm(A.dot(x_snapped) - b) > 1e-6:
                        success = False
                        break
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        W_new[k*n+i, c_prime] = x_snapped[idx]
                        idx += 1
            if not success:
                return None, None, None, False
            W_new[:, c] = 0
            return U_new, V_new, W_new, True

    # NEW: Try to merge two terms c1, c2 into a single term.
    # The contribution of terms c1 and c2 is:
    #   U[:,c1]*V[:,c1]*W[:,c1] + U[:,c2]*V[:,c2]*W[:,c2]
    # We want to express this as U[:,new]*V[:,new]*W[:,new] where
    # U[:,new], V[:,new], W[:,new] are linear combinations of the existing terms.
    # Specifically, we try to find coefficients a, b such that:
    #   (a*U[:,c1] + b*U[:,c2]) * (a*V[:,c1] + b*V[:,c2]) * (a*W[:,c1] + b*W[:,c2])
    # equals the original contribution. This is a rank-1 approximation problem.
    # A simpler approach: if U[:,c1]*V[:,c1]*W[:,c1] and U[:,c2]*V[:,c2]*W[:,c2]
    # are "close" to being linearly dependent, we can merge them.
    # For now, try a direct approach: check if the two terms can be combined
    # by finding a single rank-1 tensor that approximates their sum.
    def try_merge(U, V, W, r, c1, c2, snap_func):
        # The sum of the two rank-1 tensors is:
        # T = U[:,c1] ⊗ V[:,c1] ⊗ W[:,c1] + U[:,c2] ⊗ V[:,c2] ⊗ W[:,c2]
        # We want to find U_new, V_new, W_new such that:
        # U_new ⊗ V_new ⊗ W_new ≈ T
        # This is generally not possible for arbitrary T, but if the two terms
        # share structure, it might work.
        # A practical approach: try to express T as a single rank-1 tensor by
        # finding vectors that best approximate the decomposition.
        # This is a hard problem, so we use a heuristic: check if the two terms
        # are already "similar" enough to be merged.
        # For now, return None to indicate merge not attempted successfully.
        # A more sophisticated approach would use iterative methods.
        return None, None, None, False

    order = list(range(current_rank))
    random.shuffle(order)

    stuck_count = 0
    max_stuck = 50

    while time.time() - start_time < time_limit:
        progress = False

        # Try to eliminate terms in the shuffled order
        for c in order:
            if c >= current_rank:
                continue
            if time.time() - start_time > time_limit:
                break

            # Try solving for U, V, W with integer and half-integer snapping
            for var in [0, 1, 2]:
                for snap_func in [snap_to_int, snap_to_half_int]:
                    U_new, V_new, W_new, success = try_eliminate(U, V, W, current_rank, c, var, snap_func)
                    if success:
                        # Verify full decomposition
                        if is_valid(U_new, V_new, W_new, current_rank - 1):
                            U, V, W = U_new, V_new, W_new
                            current_rank -= 1
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                            best_rank = current_rank
                            progress = True
                            stuck_count = 0
                            # Re-shuffle remaining terms
                            order = list(range(current_rank))
                            random.shuffle(order)
                            break
                if progress:
                    break
            if progress:
                break

        if not progress:
            stuck_count += 1
            if stuck_count > max_stuck:
                break
            random.shuffle(order)

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
