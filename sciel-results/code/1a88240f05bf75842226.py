import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0

    def strassen_2x2x2():
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return np.array(U), np.array(V), np.array(W)

    def build_rank_56():
        U_cols = []
        V_cols = []
        W_cols = []
        
        U_s, V_s, W_s = strassen_2x2x2()
        
        # 4 Strassen blocks
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                for c in range(7):
                    U_col = np.zeros(n * m)
                    V_col = np.zeros(m * p)
                    W_col = np.zeros(p * n)
                    
                    for i in range(2):
                        for j in range(2):
                            U_col[i*m + (j0+j)] = U_s[i*2+j][c]
                    for j in range(2):
                        for k in range(2):
                            V_col[(j0+j)*p + (k0+k)] = V_s[j*2+k][c]
                    for k in range(2):
                        for i in range(2):
                            W_col[(k0+k)*n + i] = W_s[k*2+i][c]
                    
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
        
        # Schoolbook for remaining
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and k < 4:
                        continue
                    U_col = np.zeros(n * m)
                    V_col = np.zeros(m * p)
                    W_col = np.zeros(p * n)
                    U_col[i*m+j] = 1.0
                    V_col[j*p+k] = 1.0
                    W_col[k*n+i] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        return U, V, W

    def compute_error(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.sum((T - 1.0)**2)

    def gradient_descent(U, V, W, steps=200, lr=0.01):
        # Flatten parameters
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_ = x[:n*m*U.shape[1]].reshape(n*m, U.shape[1])
            V_ = x[n*m*U.shape[1]:n*m*U.shape[1]+m*p*U.shape[1]].reshape(m*p, U.shape[1])
            W_ = x[n*m*U.shape[1]+m*p*U.shape[1]:].reshape(p*n, U.shape[1])
            return compute_error(U_, V_, W_)
            
        # Simple gradient descent with numerical gradient
        best_x = x0
        best_err = objective(x0)
        for step in range(steps):
            if time.time() > deadline:
                break
            # Numerical gradient
            grad = np.zeros_like(x0)
            eps = 1e-5
            for i in range(len(x0)):
                x_plus = x0.copy()
                x_plus[i] += eps
                err_plus = objective(x_plus)
                grad[i] = (err_plus - best_err) / eps
                x_plus[i] = x0[i]
            
            x0 -= lr * grad
            err = objective(x0)
            if err < best_err:
                best_err = err
                best_x = x0.copy()
            else:
                # Simple restart or break if stuck
                if step > 50:
                    break
                    
        U_ = best_x[:n*m*U.shape[1]].reshape(n*m, U.shape[1])
        V_ = best_x[n*m*U.shape[1]:n*m*U.shape[1]+m*p*U.shape[1]].reshape(m*p, U.shape[1])
        W_ = best_x[n*m*U.shape[1]+m*p*U.shape[1]:].reshape(p*n, U.shape[1])
        return U_, V_, W_, best_err

    # Get baseline
    U, V, W = build_rank_56()
    baseline_err = compute_error(U, V, W)
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by 1
    # We try removing each column and optimizing
    # This is expensive, so we limit the number of attempts
    max_attempts = 10
    attempts = 0
    
    for c in range(U.shape[1]):
        if time.time() > deadline - 5:
            break
        if attempts >= max_attempts:
            break
            
        U_red = np.delete(U, c, axis=1)
        V_red = np.delete(V, c, axis=1)
        W_red = np.delete(W, c, axis=1)
        
        U_opt, V_opt, W_opt, err = gradient_descent(U_red, V_red, W_red, steps=50, lr=0.05)
        
        if err < 1e-4:
            # Success! Rank reduced
            best_rank = U_opt.shape[1]
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            break
        
        attempts += 1

    # Convert to lists of lists
    def to_list(mat):
        return mat.tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

