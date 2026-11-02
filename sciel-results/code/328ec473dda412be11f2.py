"""Rewritten solver for matrix multiplication M_{3,3,7} rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s wall-clock

    n, m, p = DIMENSIONS

    # Trivial identity decomposition (rank = n*m*p = 63)
    rank_trivial = n * m * p
    U_t = np.zeros((n * m, rank_trivial), dtype=float)
    V_t = np.zeros((m * p, rank_trivial), dtype=float)
    W_t = np.zeros((p * n, rank_trivial), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_t[i * m + j, col] = 1.0
                V_t[j * p + k, col] = 1.0
                W_t[k * n + i, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-7):
        """Check tensor decomposition validity for M_{n,m,p}."""
        rank = U.shape[1]
        # Efficient verification using matrix operations
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each a, compute M_a[b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # M_a = V * diag(U[a]) * W^T
        for a in range(n * m):
            i = a // m
            j = a % m
            # Compute M_a = V @ diag(U[a]) @ W.T
            M_a = V @ np.diag(U[a]) @ W.T
            for b in range(m * p):
                j2 = b // p
                k = b % p
                for c in range(p * n):
                    k2 = c // n
                    i2 = c % n
                    target = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    if abs(M_a[b, c] - target) > tol:
                        return False
        return True

    def compute_loss_and_grads(U, V, W, target_indices, zero_subset, weight_zero=1.0):
        """Compute loss and gradients for the decomposition."""
        dim_u, R = U.shape
        dim_v, _ = V.shape
        dim_w, _ = W.shape
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Target entries: should equal 1
        for (a, b, c) in target_indices:
            s = np.dot(U[a], V[b] * W[c])
            err = s - 1.0
            loss += err * err
            for r in range(R):
                grad_U[a, r] += 2.0 * err * V[b, r] * W[c, r]
                grad_V[b, r] += 2.0 * err * U[a, r] * W[c, r]
                grad_W[c, r] += 2.0 * err * U[a, r] * V[b, r]
        
        # Zero entries: should equal 0
        for (a, b, c) in zero_subset:
            s = np.dot(U[a], V[b] * W[c])
            loss += weight_zero * s * s
            for r in range(R):
                grad_U[a, r] += 2.0 * weight_zero * s * V[b, r] * W[c, r]
                grad_V[b, r] += 2.0 * weight_zero * s * U[a, r] * W[c, r]
                grad_W[c, r] += 2.0 * weight_zero * s * U[a, r] * V[b, r]
        
        return loss, grad_U, grad_V, grad_W

    best_result = None
    best_rank = rank_trivial

    def try_rank(R, seed, max_iters, lr_init=0.05):
        """Attempt to find a rank-R decomposition via optimization."""
        nonlocal best_result, best_rank
        dim_u = n * m  # 9
        dim_v = m * p  # 21
        dim_w = p * n  # 21

        rng = np.random.RandomState(seed)
        U = rng.randn(dim_u, R) * 0.5
        V = rng.randn(dim_v, R) * 0.5
        W = rng.randn(dim_w, R) * 0.5

        # Precompute target tensor indices
        target_indices = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
        target_set = set(target_indices)

        # All (a,b,c) pairs that should be zero
        all_pairs = [(a, b, c) for a in range(dim_u) for b in range(dim_v) for c in range(dim_w)]
        zero_pairs = [(a, b, c) for a, b, c in all_pairs if (a, b, c) not in target_set]

        # Use a subset of zero pairs for speed
        zero_subset = zero_pairs[:500]  # larger sample for better convergence

        lr = lr_init
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        patience = 0
        max_patience = 30

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            loss, grad_U, grad_V, grad_W = compute_loss_and_grads(U, V, W, target_indices, zero_subset)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                patience = 0
            else:
                patience += 1
            
            if patience > max_patience:
                break

            # Adaptive learning rate based on loss magnitude
            if loss > 1.0:
                lr = min(0.1, lr * 1.01)
            elif loss < 0.01:
                lr = max(0.001, lr * 0.95)

            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

            # Periodic restart if stuck
            if it > 0 and it % 50 == 0 and loss > 0.1:
                U = best_U + rng.randn(dim_u, R) * 0.1
                V = best_V + rng.randn(dim_v, R) * 0.1
                W = best_W + rng.randn(dim_w, R) * 0.1

        # Use the best found solution
        U, V, W = best_U, best_V, best_W

        # Round to half-integers
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0

        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r, R
        return None

    # Try ranks below 63 with more attempts
    ranks_to_try = [60, 58, 55, 52, 50, 48, 45, 42, 40]
    seeds_to_try = [42, 123, 7, 99, 1234, 5678, 9012, 3456]

    for R in ranks_to_try:
        if time.time() > deadline - 30:
            break
        found = False
        for seed in seeds_to_try:
            if time.time() > deadline - 20:
                break
            result = try_rank(R, seed, max_iters=100)
            if result is not None:
                best_result = result
                best_rank = R
                found = True
                break
        if found:
            # Try to go lower
            continue

    if best_result is not None:
        U, V, W, _ = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to trivial decomposition
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
# EVOLVE-BLOCK-END
