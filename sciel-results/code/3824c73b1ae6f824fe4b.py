"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
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

    best_result = schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for vectorized operations
    # For each (i,j,k) triple, we need:
    #   u_idx = i*m + j
    #   v_idx = j*p + k
    #   w_idx = k*n + i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    triples = np.array(triples, dtype=np.int64)
    ntrip = len(triples)

    def compute_sum(U, V, W, rank):
        """Compute the tensor contraction sum for all (i,j,k) triples."""
        U_sub = U[triples[:, 0], :]
        V_sub = V[triples[:, 1], :]
        W_sub = W[triples[:, 2], :]
        return np.sum(U_sub * V_sub * W_sub, axis=1)

    def compute_error(sums):
        """Compute max absolute error from sums."""
        return np.max(np.abs(sums - 1.0))

    def gradient_step(U, V, W, rank, lr, beta=100.0):
        """One gradient step on the smooth objective."""
        sums = compute_sum(U, V, W, rank)
        d = sums - 1.0
        abs_d = np.abs(d)
        max_abs = np.max(abs_d)
        if max_abs < 1e-15:
            return U, V, W, 0.0

        # Soft weights: exp(beta * (|d| - max_abs)) / sum
        shifted = abs_d - max_abs
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)

        U_sub = U[triples[:, 0], :]
        V_sub = V[triples[:, 1], :]
        W_sub = W[triples[:, 2], :]

        sign_d = np.sign(d)
        wgt = weights * sign_d

        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)

        contrib_U = wgt[:, None] * V_sub * W_sub
        contrib_V = wgt[:, None] * U_sub * W_sub
        contrib_W = wgt[:, None] * U_sub * V_sub

        np.add.at(dU, triples[:, 0], contrib_U)
        np.add.at(dV, triples[:, 1], contrib_V)
        np.add.at(dW, triples[:, 2], contrib_W)

        U = U - lr * dU
        V = V - lr * dV
        W = W - lr * dW

        return U, V, W, compute_error(sums)

    def try_rank(rank, max_trials=50):
        """Attempt to find a valid decomposition for a given rank."""
        if time.time() > deadline - 20:
            return None

        for trial in range(max_trials):
            if time.time() > deadline - 20:
                break

            # Multiple initialization strategies
            strategy = trial % 3
            if strategy == 0:
                # Random initialization
                np.random.seed(rank * 1000 + trial)
                scale = rank ** (-1.0 / 3.0) * 2.0
                U = np.random.randn(n * m, rank) * scale
                V = np.random.randn(m * p, rank) * scale
                W = np.random.randn(p * n, rank) * scale
            elif strategy == 1:
                # Sparse random initialization
                np.random.seed(rank * 2000 + trial)
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                for c in range(rank):
                    u_row = np.random.randint(0, n * m)
                    v_row = np.random.randint(0, m * p)
                    w_row = np.random.randint(0, p * n)
                    U[u_row, c] = np.random.choice([-1, 1]) * 1.5
                    V[v_row, c] = np.random.choice([-1, 1]) * 1.5
                    W[w_row, c] = np.random.choice([-1, 1]) * 1.5
            else:
                # Structured initialization based on schoolbook pattern
                np.random.seed(rank * 3000 + trial)
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                # Distribute the 60 schoolbook terms across rank columns
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < rank:
                                U[i * m + j, col] = 1.0
                                V[j * p + k, col] = 1.0
                                W[k * n + i, col] = 1.0
                                col += 1
                # Add random perturbation to remaining columns
                for c in range(col, rank):
                    U[:, c] = np.random.randn(n * m) * 0.5
                    V[:, c] = np.random.randn(m * p) * 0.5
                    W[:, c] = np.random.randn(p * n) * 0.5

            # Optimization with adaptive learning rate
            lr = 0.05
            min_lr = 0.0005
            best_err = np.inf
            stalled = 0
            err = np.inf

            for iteration in range(800):
                if time.time() > deadline - 20:
                    break

                U, V, W, err = gradient_step(U, V, W, rank, lr, beta=50.0)

                if err < best_err:
                    best_err = err
                    stalled = 0
                    if lr > min_lr:
                        lr *= 0.998
                else:
                    stalled += 1
                    if stalled > 30:
                        lr = max(min_lr, lr * 0.85)
                        stalled = 0

                if err < 1e-8:
                    break

            # High-precision polish
            if err < 1e-4 and time.time() < deadline - 20:
                for polish_iter in range(200):
                    if time.time() > deadline - 20:
                        break
                    lr_p = 0.0005 * (0.97 ** polish_iter)
                    U, V, W, err = gradient_step(U, V, W, rank, lr_p, beta=500.0)
                    if err < 1e-10:
                        break

            # Check if converged
            if err < 1e-7:
                # Convert to fractions and round to nearest half-integer
                U_frac = [[float(U[i, c]) for c in range(rank)] for i in range(n * m)]
                V_frac = [[float(V[i, c]) for c in range(rank)] for i in range(m * p)]
                W_frac = [[float(W[i, c]) for c in range(rank)] for i in range(p * n)]

                U_frac = [[round(x * 2) / 2 for x in row] for row in U_frac]
                V_frac = [[round(x * 2) / 2 for x in row] for row in V_frac]
                W_frac = [[round(x * 2) / 2 for x in row] for row in W_frac]

                # Verify with rounded values
                U_r = np.array(U_frac)
                V_r = np.array(V_frac)
                W_r = np.array(W_frac)
                sums_r = compute_sum(U_r, V_r, W_r, rank)
                err_rounded = compute_error(sums_r)

                if err_rounded < 1e-6:
                    return {"U": U_frac, "V": V_frac, "W": W_frac}

        return None

    # Try ranks from 55 down
    for rank in range(55, 35, -1):
        if time.time() > deadline - 20:
            break

        result = try_rank(rank, max_trials=40)
        if result is not None:
            best_result = result
            best_rank = rank
            break

    return best_result
# EVOLVE-BLOCK-END
