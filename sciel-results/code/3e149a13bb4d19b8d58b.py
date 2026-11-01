import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105  # 15s safety margin
    
    # Trivial solution
    R_trivial = n * m * p
    U_t = np.zeros((n*m, R_trivial))
    V_t = np.zeros((m*p, R_trivial))
    W_t = np.zeros((p*n, R_trivial))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_t[i*m+j, r] = 1
                V_t[j*p+k, r] = 1
                W_t[k*n+i, r] = 1
                r += 1
    
    np.random.seed(123)
    
    # Try to find better solutions
    for R in range(15, R_trivial):  # Start from 15 to save time
        if time.time() > deadline:
            break
        
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, R))
            
            # Quick local search
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Compute error
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(s - 1))
                
                if max_err < 1e-9:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Try a random flip
                mat = np.random.choice(3)
                if mat == 0:
                    row, col = np.random.randint(n*m, R)
                    old = U[row, col]
                    new = np.random.choice([-1, -0.5, 0, 0.5, 1])
                    if new != old:
                        U[row, col] = new
                        # Quick check
                        new_max_err = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if i*m+j == row:
                                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                        new_max_err = max(new_max_err, abs(s - 1))
                        if new_max_err > max_err:
                            U[row, col] = old
                elif mat == 1:
                    row, col = np.random.randint(m*p, R)
                    old = V[row, col]
                    new = np.random.choice([-1, -0.5, 0, 0.5, 1])
                    if new != old:
                        V[row, col] = new
                        new_max_err = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if j*p+k == row:
                                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                        new_max_err = max(new_max_err, abs(s - 1))
                        if new_max_err > max_err:
                            V[row, col] = old
                else:
                    row, col = np.random.randint(p*n, R)
                    old = W[row, col]
                    new = np.random.choice([-1, -0.5, 0, 0.5, 1])
                    if new != old:
                        W[row, col] = new
                        new_max_err = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if k*n+i == row:
                                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                        new_max_err = max(new_max_err, abs(s - 1))
                        if new_max_err > max_err:
                            W[row, col] = old
    
    # Return trivial solution
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
