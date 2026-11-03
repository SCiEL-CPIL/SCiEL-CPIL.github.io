import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = get_schoolbook()
    
    # Try a few ranks
    # We try 48 (half of 96) and 32 (lower bound)
    # If we find one, we return it.
    
    ranks_to_try = [48, 32]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Alternating Least Squares
            for _ in range(10):
                if time.time() > deadline - 2:
                    break
                    
                # Solve for U
                # A U = b
                num_eq = n * p * m * m
                num_unk = n * m * R
                A = np.zeros((num_eq, num_unk))
                b = np.zeros(num_eq)
                idx = 0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                b[idx] = 1.0 if j1 == j2 else 0.0
                                for r in range(R):
                                    A[idx, (i * m + j1) * R + r] = W[k * n + i, r] * V[j2 * p + k, r]
                                idx += 1
                U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U = U_flat.reshape(n * m, R)
                
                # Solve for V
                # sum_r W_ki,r U_ij1,r V_j2k,r = delta_j1j2
                # Linear in V.
                # A V = b
                A = np.zeros((num_eq, m * p * R))
                b = np.zeros(num_eq)
                idx = 0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                b[idx] = 1.0 if j1 == j2 else 0.0
                                for r in range(R):
                                    A[idx, (j2 * p + k) * R + r] = W[k * n + i, r] * U[i * m + j1, r]
                                idx += 1
                V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V = V_flat.reshape(m * p, R)
                
                # Solve for W
                A = np.zeros((num_eq, p * n * R))
                b = np.zeros(num_eq)
                idx = 0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                b[idx] = 1.0 if j1 == j2 else 0.0
                                for r in range(R):
                                    A[idx, (k * n + i) * R + r] = U[i * m + j1, r] * V[j2 * p + k, r]
                                idx += 1
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(p * n, R)
                
                # Check error
                error = 0.0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                val = 0.0
                                for r in range(R):
                                    val += W[k * n + i, r] * U[i * m + j1, r] * V[j2 * p + k, r]
                                target = 1.0 if j1 == j2 else 0.0
                                error += (val - target) ** 2
                                
                if error < 1e-6:
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    error_rounded = 0.0
                    for i in range(n):
                        for k in range(p):
                            for j1 in range(m):
                                for j2 in range(m):
                                    val = 0.0
                                    for r in range(R):
                                        val += W_rounded[k * n + i, r] * U_rounded[i * m + j1, r] * V_rounded[j2 * p + k, r]
                                    target = 1.0 if j1 == j2 else 0.0
                                    error_rounded += (val - target) ** 2
                    if error_rounded < 1e-9:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }

    return best_result
