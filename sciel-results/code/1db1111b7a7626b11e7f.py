import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 24 # Schoolbook
    
    # Pre-compute the target tensor indices
    # T[I, J, K] = 1 if I=(i,j), J=(j,k), K=(k,i)
    # I = i*m + j
    # J = j*p + k
    # K = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Function to check if a decomposition is valid
    def check_decomposition(U, V, W, tol=1e-9):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Compute sum_r U[:, r] * V[:, r] * W[:, r]
        # This is a tensor product sum.
        # We can do this by looping over r or using einsum.
        # T_computed[I, J, K] = sum_r U[I, r] * V[J, r] * W[K, r]
        # Using einsum: 'ir,jr,kr->ijk'
        T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T_comp - T)
        return err < tol

    # Function to try to find a decomposition of rank R
    def find_decomposition(R, max_iter=500, lr=0.01):
        # Initialize
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            # Compute gradient
            # L = || sum U V W - T ||^2
            # dL/dU[I, r] = 2 * sum_{J,K} (T_comp[I,J,K] - T[I,J,K]) * V[J, r] * W[K, r]
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T_comp - T
            
            # Gradients
            # dU[I, r] = 2 * sum_{J,K} diff[I,J,K] * V[J, r] * W[K, r]
            # This is 2 * einsum('ijk,jr,kr->ir', diff, V, W)
            dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            err = np.linalg.norm(diff)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-9:
                break
                
        if best_err < 1e-7:
            # Try to round to nearest 0.5
            U_round = np.round(best_U * 2) / 2
            V_round = np.round(best_V * 2) / 2
            W_round = np.round(best_W * 2) / 2
            
            if check_decomposition(U_round, V_round, W_round):
                return U_round, V_round, W_round
            else:
                # Maybe it's exact without rounding?
                if check_decomposition(best_U, best_V, best_W):
                    return best_U, best_V, best_W
        return None

    # Try ranks from 23 down to 18
    for R in range(23, 17, -1):
        # Multiple restarts for better chance
        for restart in range(3):
            res = find_decomposition(R, max_iter=100, lr=0.05)
            if res is not None:
                U, V, W = res
                # Convert to list of lists for return
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
