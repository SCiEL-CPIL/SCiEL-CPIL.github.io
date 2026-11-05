import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=np.float64)
    V_school = np.zeros((m * p, rank_school), dtype=np.float64)
    W_school = np.zeros((p * n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school
    
    # Block decomposition: [4,5,5] = [4,4,4] + [4,4,1] + [4,1,5]
    n4 = 4
    
    # Schoolbook for [4,4,1]
    rank_441 = 16
    U_441 = np.zeros((n * m, rank_441))
    V_441 = np.zeros((m * p, rank_441))
    W_441 = np.zeros((p * n, rank_441))
    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i * m + j, col] = 1.0
            V_441[j * p, col] = 1.0
            W_441[i, col] = 1.0
            col += 1
    
    # Schoolbook for [4,1,5]
    rank_415 = 20
    U_415 = np.zeros((n * m, rank_415))
    V_415 = np.zeros((m * p, rank_415))
    W_415 = np.zeros((p * n, rank_415))
    col = 0
    for i in range(4):
        for k in range(5):
            U_415[i * m, col] = 1.0
            V_415[k, col] = 1.0
            W_415[k * n + i, col] = 1.0
            col += 1
    
    # Try to find low-rank decomposition for [4,4,4]
    for target_rank in [23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + trial)
            u = np.random.randn(target_rank, 16) * 0.1
            v = np.random.randn(target_rank, 16) * 0.1
            w = np.random.randn(target_rank, 16) * 0.1
            
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Solve for u
                A_u = np.zeros((64, target_rank * 16))
                b_u = np.ones(64)
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            eq = (i * 4 + j) * 4 + k
                            for r in range(target_rank):
                                A_u[eq, r * 16 + i * 4 + j] = v[r, j * 4 + k] * w[r, k * 4 + i]
                u = np.linalg.lstsq(A_u, b_u, rcond=None)[0].reshape(target_rank, 16)
                
                # Solve for v
                A_v = np.zeros((64, target_rank * 16))
                b_v = np.ones(64)
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            eq = (i * 4 + j) * 4 + k
                            for r in range(target_rank):
                                A_v[eq, r * 16 + j * 4 + k] = u[r, i * 4 + j] * w[r, k * 4 + i]
                v = np.linalg.lstsq(A_v, b_v, rcond=None)[0].reshape(target_rank, 16)
                
                # Solve for w
                A_w = np.zeros((64, target_rank * 16))
                b_w = np.ones(64)
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            eq = (i * 4 + j) * 4 + k
                            for r in range(target_rank):
                                A_w[eq, r * 16 + k * 4 + i] = u[r, i * 4 + j] * v[r, j * 4 + k]
                w = np.linalg.lstsq(A_w, b_w, rcond=None)[0].reshape(target_rank, 16)
            
            # Verify
            max_err = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        val = np.dot(u[:, i * 4 + j], v[:, j * 4 + k] * w[:, k * 4 + i])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                U_444 = np.zeros((n * m, target_rank))
                V_444 = np.zeros((m * p, target_rank))
                W_444 = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    for i in range(4):
                        for j in range(4):
                            U_444[i * m + j, r] = u[r, i * 4 + j]
                    for j in range(4):
                        for k in range(4):
                            V_444[j * p + k, r] = v[r, j * 4 + k]
                    for k in range(4):
                        for i in range(4):
                            W_444[k * n + i, r] = w[r, k * 4 + i]
                
                total_rank = target_rank + 16 + 20
                U_total = np.hstack([U_444, U_441, U_415])
                V_total = np.hstack([V_444, V_441, V_415])
                W_total = np.hstack([W_444, W_415, W_441])  # Wait, this is wrong
                W_total = np.hstack([W_444, W_441, W_415])
                
                # Verify
                max_err_full = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_total[i * m + j], V_total[j * p + k] * W_total[k * n + i])
                            max_err_full = max(max_err_full, abs(val - 1))
                
                if max_err_full < 1e-6:
                    best_U, best_V, best_W = U_total, V_total, W_total
                    best_rank = total_rank
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
