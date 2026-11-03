import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    dim_U, dim_V, dim_W = n * m, m * p, p * n

    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit

    # Build target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Baseline: rank-17 solution
    U17, V17, W17 = _build_rank17()

    best_rank = 17
    best_U, best_V, best_W = U17, V17, W17

    # Try to find rank-16 solution
    if time.time() < deadline - 30:
        r16 = _search_rank16(T, U17, V17, W17, deadline)
        if r16 is not None:
            best_rank = 16
            best_U, best_V, best_W = r16

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _build_rank17():
    n, m, p = 2, 3, 3
    dim_U, dim_V, dim_W = 6, 9, 6
    rank = 17
    U = np.zeros((dim_U, rank), dtype=np.int64)
    V = np.zeros((dim_V, rank), dtype=np.int64)
    W = np.zeros((dim_W, rank), dtype=np.int64)

    # Strassen for A[0:2,0:2] * B[0:2,0:2]
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)*B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11*(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1

    # M6 = (A10-A00)*(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Schoolbook: A[:,2]*B[2,:] -> C[:,0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # Schoolbook: A[:,0:2]*B[:,2] -> C[:,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1

    # Schoolbook: A[:,2]*B[2,2] -> C[:,2]
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return U, V, W


def _search_rank16(T, U17, V17, W17, deadline):
    """Search for a rank-16 solution using local search."""
    dim_U, dim_V, dim_W = 6, 9, 6
    rank = 16

    # Precompute T as flat array for speed
    T_flat = T.flatten()

    def compute_S(U, V, W):
        """Compute the sum of rank-1 terms."""
        S = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
        for r in range(rank):
            S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return S

    def penalty(U, V, W):
        S = compute_S(U, V, W)
        return np.sum((S - T) ** 2)

    def is_valid(U, V, W):
        S = compute_S(U, V, W)
        return np.allclose(S, T, atol=1e-9)

    # Start from rank-17, drop the last column (index 16)
    # The dropped column is A12*B22 -> C12
    U = U17[:, :rank].copy().astype(np.float64)
    V = V17[:, :rank].copy().astype(np.float64)
    W = W17[:, :rank].copy().astype(np.float64)

    # Also try dropping other columns
    best_penalty = np.inf
    best_U, best_V, best_W = None, None, None

    for drop_col in range(17):
        U_try = U17[:, :drop_col].copy().astype(np.float64)
        V_try = V17[:, :drop_col].copy().astype(np.float64)
        W_try = W17[:, :drop_col].copy().astype(np.float64)
        if drop_col < 16:
            U_try = np.column_stack([U_try, U17[:, drop_col+1:17].copy().astype(np.float64)])
            V_try = np.column_stack([V_try, V17[:, drop_col+1:17].copy().astype(np.float64)])
            W_try = np.column_stack([W_try, W17[:, drop_col+1:17].copy().astype(np.float64)])

        p = penalty(U_try, V_try, W_try)
        if p < best_penalty:
            best_penalty = p
            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()

        if time.time() > deadline - 10:
            break

    if best_U is None:
        return None

    # Local search: perturb entries and accept improvements
    U = best_U
    V = best_V
    W = best_W
    current_penalty = best_penalty

    np.random.seed(12345)
    step_size = 1.0
    iterations = 0

    while time.time() < deadline - 10:
        iterations += 1

        # Randomly choose which matrix and entry to perturb
        mat = np.random.choice(3)
        if mat == 0:
            idx = np.random.randint(dim_U * rank)
            a, r = divmod(idx, rank)
            delta = np.random.choice([-2, -1, 1, 2]) * step_size
            old_val = U[a, r]
            U[a, r] += delta
            new_p = penalty(U, V, W)
            if new_p < current_penalty:
                current_penalty = new_p
            else:
                U[a, r] = old_val
        elif mat == 1:
            idx = np.random.randint(dim_V * rank)
            b, r = divmod(idx, rank)
            delta = np.random.choice([-2, -1, 1, 2]) * step_size
            old_val = V[b, r]
            V[b, r] += delta
            new_p = penalty(U, V, W)
            if new_p < current_penalty:
                current_penalty = new_p
            else:
                V[b, r] = old_val
        else:
            idx = np.random.randint(dim_W * rank)
            c, r = divmod(idx, rank)
            delta = np.random.choice([-2, -1, 1, 2]) * step_size
            old_val = W[c, r]
            W[c, r] += delta
            new_p = penalty(U, V, W)
            if new_p < current_penalty:
                current_penalty = new_p
            else:
                W[c, r] = old_val

        # Check for exact solution periodically
        if iterations % 100 == 0 and current_penalty < 1e-6:
            if is_valid(U, V, W):
                # Round to integers
                U_int = np.round(U).astype(np.int64)
                V_int = np.round(V).astype(np.int64)
                W_int = np.round(W).astype(np.int64)
                if is_valid(U_int.astype(np.float64), V_int.astype(np.float64), W_int.astype(np.float64)):
                    return U_int, V_int, W_int

        # Occasionally try larger moves
        if iterations % 1000 == 0:
            # Try to set a whole column to a new random pattern
            r = np.random.randint(rank)
            for _ in range(50):
                U_new_col = np.random.randint(-2, 3, size=dim_U).astype(np.float64)
                V_new_col = np.random.randint(-2, 3, size=dim_V).astype(np.float64)
                W_new_col = np.random.randint(-2, 3, size=dim_W).astype(np.float64)
                U[:, r] = U_new_col
                V[:, r] = V_new_col
                W[:, r] = W_new_col
                new_p = penalty(U, V, W)
                if new_p < current_penalty:
                    current_penalty = new_p
                    break
                else:
                    U[:, r] = U17[:, r].astype(np.float64)
                    V[:, r] = V17[:, r].astype(np.float64)
                    W[:, r] = W17[:, r].astype(np.float64)

    # Final check
    if current_penalty < 1e-6:
        U_int = np.round(U).astype(np.int64)
        V_int = np.round(V).astype(np.int64)
        W_int = np.round(W).astype(np.int64)
        if is_valid(U_int.astype(np.float64), V_int.astype(np.float64), W_int.astype(np.float64)):
            return U_int, V_int, W_int

    return None
