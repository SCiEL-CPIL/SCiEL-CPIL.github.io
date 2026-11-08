import numpy as np
import time

def _verify(U, V, W, n=2, m=3, p=3):
    num_inputs_a = n * m
    num_inputs_b = m * p
    num_outputs = p * n
    T = np.zeros((num_inputs_a, num_inputs_b, num_outputs))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i * m + j
                ib = j * p + k
                out_idx = k * n + i
                T[ia, ib, out_idx] = 1.0
    computed = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.max(np.abs(computed - T)) < 1e-10

def _rank17_solution():
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen for A[:,:2] * B[:2,:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1

    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    U[5, 13] = 1; V[6, 13] = 1; W[1, 13] = 1
    U[5, 14] = 1; V[7, 14] = 1; W[3, 14] = 1

    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return U, V, W

def _compute_sse(U, V, W, T):
    """Compute sum of squared errors (smooth surrogate)."""
    computed = np.einsum('ir,jr,kr->ijk', U, V, W)
    diff = computed - T
    return np.sum(diff * diff)

def _try_rank16_search(deadline):
    """Attempt to find a rank-16 decomposition via smooth gradient-based search."""
    n, m, p = 2, 3, 3
    num_a, num_b, num_c = n*m, m*p, p*n
    rank = 16

    T = np.zeros((num_a, num_b, num_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    U17, V17, W17 = _rank17_solution()
    best_err = np.inf
    best_U, best_V, best_W = None, None, None

    rng = np.random.default_rng(42)

    # Try zeroing each different column from rank-17
    for zero_col in range(17):
        if time.time() > deadline - 10:
            break

        # Initialize from rank-17 with one column zeroed
        U = U17.copy()
        V = V17.copy()
        W = W17.copy()
        U[:, zero_col] = 0
        V[:, zero_col] = 0
        W[:, zero_col] = 0

        # Keep only first 16 columns
        U = U[:, :rank].copy()
        V = V[:, :rank].copy()
        W = W[:, :rank].copy()

        # Random perturbation
        scale = 0.05
        U += rng.normal(0, scale, U.shape)
        V += rng.normal(0, scale, V.shape)
        W += rng.normal(0, scale, W.shape)

        # Gradient-based optimization with adaptive learning rate
        lr = 0.01
        prev_err = _compute_sse(U, V, W, T)

        for step in range(300):
            if time.time() > deadline - 10:
                break

            computed = np.einsum('ir,jr,kr->ijk', U, V, W)
            max_err = np.max(np.abs(computed - T))
            sse = _compute_sse(U, V, W, T)

            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if max_err < 1e-10:
                break

            # Compute gradients w.r.t. each matrix
            # grad_U[ia, r] = sum_{ib, oc} V[ib, r] * W[oc, r] * (computed[ia, ib, oc] - T[ia, ib, oc])
            residual = computed - T
            grad_U = np.einsum('jrk,ijk->ir', V, W, residual)
            grad_V = np.einsum('irk,ijk->jr', U, W, residual)
            grad_W = np.einsum('ijr,ijk->kr', U, V, residual)

            # Adaptive learning rate
            if sse < prev_err * 0.99:
                lr = min(lr * 1.1, 0.1)
            elif sse > prev_err * 1.01:
                lr = max(lr * 0.7, 0.001)
            prev_err = sse

            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

        # Final polish: if close, try high-precision refinement
        if best_err < 0.1 and best_U is not None:
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
            lr = 0.001
            for step in range(100):
                if time.time() > deadline - 10:
                    break
                computed = np.einsum('ir,jr,kr->ijk', U, V, W)
                max_err = np.max(np.abs(computed - T))
                if max_err < 1e-10:
                    break
                residual = computed - T
                grad_U = np.einsum('jrk,ijk->ir', V, W, residual)
                grad_V = np.einsum('irk,ijk->jr', U, W, residual)
                grad_W = np.einsum('ijr,ijk->kr', U, V, residual)
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                if max_err < best_err:
                    best_err = max_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if best_err < 1e-10 and best_U is not None:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    n, m, p = 2, 3, 3
    num_a, num_b, num_c = n*m, m*p, p*n

    # Try rank 16 first
    result = _try_rank16_search(deadline)
    if result is not None:
        U, V, W = result
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to rank 17
    U, V, W = _rank17_solution()
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: rank 18 naive
    rank = 18
    U = np.zeros((num_a, rank))
    V = np.zeros((num_b, rank))
    W = np.zeros((num_c, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
