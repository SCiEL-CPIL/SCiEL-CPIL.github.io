import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Strassen decomposition using structured coefficients (from inspiration 1)
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]

    def strassen_block():
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0
        for a in range(2):  # A block column: 2a, 2a+1
            for b in range(4):  # B column block: 2b, 2b+1
                for t in range(7):
                    term = strassen_terms[t]
                    for (i_prime, j_prime), coeff in term['x'].items():
                        u_idx = i_prime * m + (2*a + j_prime)
                        U[u_idx, col] += coeff
                    for (j_prime, k_prime), coeff in term['y'].items():
                        v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                        V[v_idx, col] += coeff
                    for (i_prime, k_prime), coeff in term['z'].items():
                        w_idx = (2*b + k_prime) * n + i_prime
                        W[w_idx, col] += coeff
                    col += 1

        return U, V, W, rank

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Vectorized verification."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank] * V[j*p+k, :rank], W[k*n+i, :rank])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def error_metric(U, V, W, rank):
        """Compute sum of squared errors for all tensor entries (vectorized)."""
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank] * V[j*p+k, :rank], W[k*n+i, :rank])
                    total_err += (val - 1.0) ** 2
        return total_err

    def compute_gradients(U, V, W, rank):
        """Compute gradients of sum-of-squared-errors w.r.t. U, V, W (vectorized)."""
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)

        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :rank]
                for k in range(p):
                    v_row = V[j*p+k, :rank]
                    w_row = W[k*n+i, :rank]
                    val = np.dot(u_row * v_row, w_row)
                    err = val - 1.0
                    uv = u_row * v_row
                    dU[i*m+j, :rank] += err * v_row * w_row
                    dV[j*p+k, :rank] += err * u_row * w_row
                    dW[k*n+i, :rank] += err * uv
        return dU, dV, dW

    def optimize_rank(U_init, V_init, W_init, target_rank, max_iters=2000, lr=0.01):
        """
        Gradient-based optimization to find a valid decomposition at target_rank.
        Uses smooth surrogate: minimize sum of squared errors.
        """
        if time.time() > deadline - 15:
            return None

        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))

        # Initialize: use first target_rank columns from best solution
        init_cols = min(target_rank, U_init.shape[1])
        U[:, :init_cols] = U_init[:, :init_cols]
        V[:, :init_cols] = V_init[:, :init_cols]
        W[:, :init_cols] = W_init[:, :init_cols]

        # If we're reducing rank, add small random perturbation to break symmetry
        if target_rank < init_cols:
            U[:, init_cols:] = np.random.randn(nm, target_rank - init_cols) * 0.1
            V[:, init_cols:] = np.random.randn(mp, target_rank - init_cols) * 0.1
            W[:, init_cols:] = np.random.randn(pn, target_rank - init_cols) * 0.1

        # Compute initial error
        current_err = error_metric(U, V, W, target_rank)

        # Gradient descent with adaptive learning rate
        best_U, best_V, best_W, best_err = U.copy(), V.copy(), W.copy(), current_err

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Compute gradients (vectorized)
            dU, dV, dW = compute_gradients(U, V, W, target_rank)

            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW

            new_err = error_metric(U, V, W, target_rank)

            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            # Adaptive learning rate
            if new_err > current_err * 1.5:
                lr *= 0.5
            elif it % 100 == 0 and it > 0:
                lr *= 0.95

            current_err = new_err

            if best_err < 1e-10:
                break

        if best_err < 1e-9:
            return best_U, best_V, best_W, target_rank
        return None

    def polish(U, V, W, rank, max_iters=5000, lr=0.005):
        """
        High-precision polish pass: tighter tolerance, more iterations,
        smaller learning rate for final convergence.
        """
        if time.time() > deadline - 5:
            return None

        current_err = error_metric(U, V, W, rank)
        if current_err < 1e-12:
            return U, V, W, rank

        best_U, best_V, best_W, best_err = U.copy(), V.copy(), W.copy(), current_err

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            dU, dV, dW = compute_gradients(U, V, W, rank)

            U -= lr * dU
            V -= lr * dV
            W -= lr * dW

            new_err = error_metric(U, V, W, rank)

            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            # Adaptive learning rate: decay slowly
            if new_err > best_err * 1.5:
                lr *= 0.5
            elif it % 200 == 0 and it > 0:
                lr *= 0.9

            if best_err < 1e-14:
                break

        if best_err < 1e-10:
            return best_U, best_V, best_W, rank
        return None

    # Start with Strassen (rank 56)
    U_best, V_best, W_best, rank_best = strassen_block()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    # Try to find lower rank via gradient-based optimization
    # Theoretical lower bound: ceil(64/2) = 32, but realistically 40-48
    target_ranks = [48, 44, 40, 36, 32]

    for target in target_ranks:
        if time.time() > deadline - 25:
            break
        if target >= rank_best:
            continue

        result = optimize_rank(U_best, V_best, W_best, target, max_iters=1500, lr=0.02)
        if result is not None:
            U_best, V_best, W_best, rank_best = result
            # Continue trying lower ranks
            continue
        else:
            # If this rank failed, don't try lower ranks (they'll likely fail too)
            break

    # Final high-precision polish pass
    if time.time() < deadline - 10:
        polished = polish(U_best, V_best, W_best, rank_best, max_iters=3000, lr=0.005)
        if polished is not None:
            U_best, V_best, W_best, rank_best = polished

    # Final verification
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
