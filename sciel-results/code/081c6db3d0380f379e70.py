"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 115  # 115s limit, leaving 5s margin

    # Trivial solution: rank = n*m*p = 90
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(nm)]
    V_trivial = [[0] * rank_trivial for _ in range(mp)]
    W_trivial = [[0] * rank_trivial for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1

    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = rank_trivial

    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # Strategy: try ranks from a reasonable lower bound up to 90
    # Lower bound: at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But realistically, for 3x5x6, we might find something around 60-80

    # Try a few candidate ranks
    candidate_ranks = [60, 65, 70, 75, 80, 85]

    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue

        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline - 5:
                break

            np.random.seed(42 + attempt * 1000 + target_rank)
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1

            # Gradient descent to minimize ||sum_r U_r o V_r o W_r - T||^2
            lr = 0.01
            converged = False

            for step in range(200):
                if time.time() > deadline - 5:
                    break

                # Compute current tensor value at all (i,j,k)
                # T_current[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_current = np.zeros((n, m, p))
                for r in range(target_rank):
                    u_r = U[:, r].reshape(n, m)
                    v_r = V[:, r].reshape(m, p)
                    w_r = W[:, r].reshape(p, n)
                    # T_current[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
                    # Vectorize: for each r, compute outer product structure
                    for i in range(n):
                        for j in range(m):
                            T_current[i, j, :] += u_r[i, j] * (v_r[j, :] * w_r[:, i].T)

                error = T_current - 1.0
                err_norm = np.linalg.norm(error)

                if err_norm < 1e-8:
                    converged = True
                    break

                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)

                for r in range(target_rank):
                    u_r = U[:, r].reshape(n, m)
                    v_r = V[:, r].reshape(m, p)
                    w_r = W[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                e = error[i, j, k]
                                dU[i * m + j, r] += e * v_r[j, k] * w_r[k, i]
                                dV[j * p + k, r] += e * u_r[i, j] * w_r[k, i]
                                dW[k * n + i, r] += e * u_r[i, j] * v_r[j, k]

                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

            if converged:
                # Verify exactness and round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2

                # Check if rounded values still give exact decomposition
                T_check = np.zeros((n, m, p))
                for r in range(target_rank):
                    u_r = U_rounded[:, r].reshape(n, m)
                    v_r = V_rounded[:, r].reshape(m, p)
                    w_r = W_rounded[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            T_check[i, j, :] += u_r[i, j] * (v_r[j, :] * w_r[:, i].T)

                if np.allclose(T_check, 1.0, atol=1e-9):
                    best_rank = target_rank
                    best_U = U_rounded.reshape(nm, target_rank).tolist()
                    best_V = V_rounded.reshape(mp, target_rank).tolist()
                    best_W = W_rounded.reshape(pn, target_rank).tolist()
                    break  # Found a valid decomposition for this rank, move to lower

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
