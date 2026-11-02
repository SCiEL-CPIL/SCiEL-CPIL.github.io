"""Matrix multiplication 2,2,3: rank-10 exact tensor decomposition via randomized gradient descent."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=3

    # The matrix multiplication tensor T has entries:
    # T[i, j, k, l] = delta_{j,k} for i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r] = T[i,j,k,l]
    #
    # Shapes: U is (n*m, R) = (4, R), V is (m*p, R) = (6, R), W is (p*n, R) = (6, R)
    # The known rank is 10. We'll try to find an exact integer/half-integer solution.

    def make_target():
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i, j, k, l] = 1.0 if j == k else 0.0
        return T

    def compute_tensor(U, V, W):
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i, j, k, l] = np.dot(U[i * m + j], V[j * p + k] * W[k * n + l])
        return T

    def try_rank(R, max_seeds=50):
        T_target = make_target()
        best_err = float('inf')
        best_result = None

        for seed in range(max_seeds):
            np.random.seed(seed * 137 + 42)
            # Initialize with random values scaled appropriately
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1

            # Adaptive gradient descent
            lr = 0.05
            for iteration in range(20000):
                T_curr = compute_tensor(U, V, W)
                err = T_curr - T_target
                loss = np.sum(err ** 2)

                if loss < 1e-12:
                    break

                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)

                for i in range(n):
                    for j in range(m):
                        for k in range(m):
                            for l in range(p):
                                e = err[i, j, k, l]
                                u_row = U[i * m + j]
                                v_row = V[j * p + k]
                                w_row = W[k * n + l]
                                # dU
                                dU[i * m + j] += 2 * e * (v_row * w_row)
                                # dV
                                dV[j * p + k] += 2 * e * (u_row * w_row)
                                # dW
                                dW[k * n + l] += 2 * e * (u_row * v_row)

                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 1e-8:
                    eff_lr = lr * min(1.0, 1.0 / (1.0 + grad_norm))
                else:
                    eff_lr = lr

                U -= eff_lr * dU
                V -= eff_lr * dV
                W -= eff_lr * dW

                # Periodic restart if stuck
                if iteration % 5000 == 4999 and loss > 1e-6:
                    # Slight perturbation
                    U += np.random.randn(n * m, R) * 0.01
                    V += np.random.randn(m * p, R) * 0.01
                    W += np.random.randn(p * n, R) * 0.01

            T_final = compute_tensor(U, V, W)
            max_err = np.max(np.abs(T_final - T_target))
            if max_err < best_err:
                best_err = max_err
                best_result = (U.copy(), V.copy(), W.copy())

        if best_err < 1e-6 and best_result is not None:
            return best_result, best_err
        return None, best_err

    def find_exact_solution(U, V, W):
        """Try to convert numerical solution to exact integer or half-integer form."""
        all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        # Try to find a common scale that makes all values integers
        # Check if values are already close to integers
        for scale in [1, 2, 4, 8, 16, 32, 64]:
            U_s = U * scale
            V_s = V * scale
            W_s = W * scale

            U_int = np.round(U_s)
            V_int = np.round(V_s)
            W_int = np.round(W_s)

            U_err = np.max(np.abs(U_int - U_s))
            V_err = np.max(np.abs(V_int - V_s))
            W_err = np.max(np.abs(W_int - W_s))
            max_s_err = max(U_err, V_err, W_err)

            if max_s_err < 1e-6:
                # Verify
                T_check = compute_tensor(U_int.astype(float), V_int.astype(float), W_int.astype(float))
                T_target = make_target()
                if np.max(np.abs(T_check - T_target)) < 1e-8:
                    return (U_int.astype(int).tolist(), V_int.astype(int).tolist(), W_int.astype(int).tolist())

        # Try half-integers (multiply by 2, round, divide by 2)
        for scale in [2, 4, 8, 16]:
            U_s = U * scale
            V_s = V * scale
            W_s = W * scale

            U_half = np.round(U_s * 2) / 2
            V_half = np.round(V_s * 2) / 2
            W_half = np.round(W_s * 2) / 2

            U_err = np.max(np.abs(U_half - U_s))
            V_err = np.max(np.abs(V_half - V_s))
            W_err = np.max(np.abs(W_half - W_s))
            max_s_err = max(U_err, V_err, W_err)

            if max_s_err < 1e-6:
                T_check = compute_tensor(U_half, V_half, W_half)
                T_target = make_target()
                if np.max(np.abs(T_check - T_target)) < 1e-8:
                    return (U_half.tolist(), V_half.tolist(), W_half.tolist())

        return None

    # Try rank 10 (the known minimal rank for 2x2 by 2x3)
    for R in [10]:
        result, err = try_rank(R, max_seeds=30)
        if result is not None:
            U_num, V_num, W_num = result
            exact = find_exact_solution(U_num, V_num, W_num)
            if exact is not None:
                U_list, V_list, W_list = exact
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook decomposition with rank = n*m*p = 12
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                U[i * m + j][col] = 1
                V[j * p + l][col] = 1
                W[l * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
