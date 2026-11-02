"""Rewritten solver for matrix multiplication M_{3,3,7} rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 150s wall-clock

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
        # Precompute target indices: (i*m+j, j*p+k, k*n+i) -> 1
        # All other (a,b,c) -> 0
        # For efficiency, check all 9*21*21 = 3969 entries
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use batched computation
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    s = np.dot(U[a], V[b] * W[c])
                    # Determine target value
                    # a = i*m+j, b = j'*p+k, c = k'*n+i'
                    # Target = 1 iff i==i', j==j', k==k'
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    target = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    if abs(s - target) > tol:
                        return False
        return True

    # Try to find a lower-rank decomposition using gradient-based optimization
    # with rounding to half-integers at the end
    best_result = None

    def try_rank(R, seed, max_iters, lr_init=0.01):
        """Attempt to find a rank-R decomposition via optimization."""
        dim_u = n * m  # 9
        dim_v = m * p  # 21
        dim_w = p * n  # 21

        rng = np.random.RandomState(seed)
        U = rng.randn(dim_u, R) * 0.3
        V = rng.randn(dim_v, R) * 0.3
        W = rng.randn(dim_w, R) * 0.3

        # Precompute target tensor indices
        target_indices = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
        target_set = set(target_indices)

        # All (a,b,c) pairs that should be zero
        all_pairs = [(a, b, c) for a in range(dim_u) for b in range(dim_v) for c in range(dim_w)]
        zero_pairs = [(a, b, c) for a, b, c in all_pairs if (a, b, c) not in target_set]

        # We'll optimize only on the target entries for speed, then verify
        # Loss = sum_{target} (sum_r U[a,r]*V[b,r]*W[c,r] - 1)^2
        #       + lambda * sum_{zero} (sum_r U[a,r]*V[b,r]*W[c,r])^2
        # Use a subset of zero pairs for speed
        zero_subset = zero_pairs[:200]  # sample

        lr = lr_init
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)

            # Gradients from target entries
            for (a, b, c) in target_indices:
                s = np.dot(U[a], V[b] * W[c])
                err = s - 1.0
                for r in range(R):
                    grad_U[a, r] += 2.0 * err * V[b, r] * W[c, r]
                    grad_V[b, r] += 2.0 * err * U[a, r] * W[c, r]
                    grad_W[c, r] += 2.0 * err * U[a, r] * V[b, r]

            # Gradients from zero entries (with weight 0.5 to balance)
            weight_zero = 0.5
            for (a, b, c) in zero_subset:
                s = np.dot(U[a], V[b] * W[c])
                for r in range(R):
                    grad_U[a, r] += 2.0 * weight_zero * s * V[b, r] * W[c, r]
                    grad_V[b, r] += 2.0 * weight_zero * s * U[a, r] * W[c, r]
                    grad_W[c, r] += 2.0 * weight_zero * s * U[a, r] * V[b, r]

            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

            # Decay learning rate
            lr *= 0.995

        # Round to half-integers
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0

        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r, R
        return None

    # Try ranks below 63
    for R in [55, 50, 45, 40]:
        if time.time() > deadline - 20:
            break
        for seed in [42, 123, 7, 99]:
            if time.time() > deadline - 10:
                break
            result = try_rank(R, seed, max_iters=50)
            if result is not None:
                best_result = result
                break
        if best_result is not None:
            break

    if best_result is not None:
        U, V, W, _ = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to trivial decomposition
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
# EVOLVE-BLOCK-END
