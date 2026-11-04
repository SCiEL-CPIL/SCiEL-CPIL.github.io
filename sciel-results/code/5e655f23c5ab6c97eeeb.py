import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    def is_valid(U, V, W, R):
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0
                        for r in range(R):
                            s += U[i*m + j1, r] * V[j2*p + k, r] * W[k*n + i, r]
                        target = 1 if j1 == j2 else 0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a decomposition with R=40
    # Use random U, V and solve for W
    for R in [40, 38]:
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            
            # Construct the linear system A * W_flat = b
            # Equations: sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta(j1, j2)
            # Unknowns: W[k*n+i, r] for all k, i, r
            
            num_eq = n * p * m * m
            num_unk = p * n * R
            
            A = np.zeros((num_eq, num_unk))
            b = np.zeros(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            for r in range(R):
                                coeff = U[i*m + j1, r] * V[j2*p + k, r]
                                if coeff != 0:
                                    # W index: k*n + i, r
                                    w_idx = (k*n + i) * R + r
                                    A[eq_idx, w_idx] = coeff
                            b[eq_idx] = 1 if j1 == j2 else 0
                            eq_idx += 1
            
            # Solve A * W_flat = b
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if solution is close to integer/half-integer
            W_flat = np.round(W_flat * 2) / 2
            W = W_flat.reshape(p*n, R)
            
            if is_valid(U, V, W, R):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        if time.time() > deadline - 10:
            break

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
