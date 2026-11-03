import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105

    # Build the multiplication tensor T[a,b,c]
    # a = i*m+k (A index), b = k*p+j (B index), c = j*n+i (C index per W convention)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1.0

    # Attempt rank-16 decomposition search
    rank = 16
    nU, nV, nW = n * m, m * p, p * n
    np.random.seed(42)
    best_error = float('inf')
    best_params = None

    for trial in range(40):
        if time.time() > deadline:
            break

        scale = 0.5
        U = np.random.randn(nU, rank) * scale
        V = np.random.randn(nV, rank) * scale
        W = np.random.randn(nW, rank) * scale

        def loss_and_grad(params):
            U_l = params[:nU * rank].reshape(nU, rank)
            V_l = params[nU * rank:nU * rank + nV * rank].reshape(nV, rank)
            W_l = params[nU * rank + nV * rank:].reshape(nW, rank)

            T_approx = np.einsum('at,bt,ct->abc', U_l, V_l, W_l)
            diff = T - T_approx
            error = np.sum(diff * diff)

            grad_U = -2.0 * np.einsum('abc,bt,ct->at', diff, V_l, W_l)
            grad_V = -2.0 * np.einsum('abc,at,ct->bt', diff, U_l, W_l)
            grad_W = -2.0 * np.einsum('abc,at,bt->ct', diff, U_l, V_l)

            grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
            return error, grad

        params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])

        try:
            result = minimize(loss_and_grad, params, jac=True, method='L-BFGS-B',
                              options={'maxiter': 300, 'gtol': 1e-12})
            if result.fun < best_error:
                best_error = result.fun
                best_params = result.x
        except Exception:
            pass

    # If a valid rank-16 solution is found, return it
    if best_params is not None and best_error < 1e-8:
        U = best_params[:nU * rank].reshape(nU, rank)
        V = best_params[nU * rank:nU * rank + nV * rank].reshape(nV, rank)
        W = best_params[nU * rank + nV * rank:].reshape(nW, rank)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: rank-17 decomposition
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
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

    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
