import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = 140.0
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def compute_error(U, V, W, rank):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
                    for i2 in range(n):
                        for k2 in range(p):
                            if i2 == i and k2 == k:
                                continue
                            s2 = 0.0
                            for r in range(rank):
                                s2 += U[i*m+j, r] * V[j*p+k, r] * W[k2*n+i2, r]
                            err += s2 ** 2
        return err
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
                    for i2 in range(n):
                        for k2 in range(p):
                            if i2 == i and k2 == k:
                                continue
                            s2 = 0.0
                            for r in range(rank):
                                s2 += U[i*m+j, r] * V[j*p+k, r] * W[k2*n+i2, r]
                            if abs(s2) > 1e-9:
                                return False
        return True
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    for target_rank in [32, 33, 34, 35]:
        if time.time() > deadline:
            break
        found = False
        for trial in range(8):
            if time.time() > deadline:
                break
            if found:
                break
            
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            for it in range(30):
                if time.time() > deadline:
                    break
                
                for i in range(n):
                    for j in range(m):
                        rows = []
                        bvals = []
                        for k in range(p):
                            row = np.array([V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank)])
                            rows.append(row)
                            bvals.append(1.0)
                            for i2 in range(n):
                                for k2 in range(p):
                                    if i2 == i and k2 == k:
                                        continue
                                    row2 = np.array([V[j*p+k, r] * W[k2*n+i2, r] for r in range(target_rank)])
                                    rows.append(row2)
                                    bvals.append(0.0)
                        A = np.array(rows)
                        b = np.array(bvals)
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                for j in range(m):
                    for k in range(p):
                        rows = []
                        bvals = []
                        for i in range(n):
                            row = np.array([U[i*m+j, r] * W[k*n+i, r] for r in range(target_rank)])
                            rows.append(row)
                            bvals.append(1.0)
                            for i2 in range(n):
                                for k2 in range(p):
                                    if i2 == i and k2 == k:
                                        continue
                                    row2 = np.array([U[i*m+j, r] * W[k2*n+i2, r] for r in range(target_rank)])
                                    rows.append(row2)
                                    bvals.append(0.0)
                        A = np.array(rows)
                        b = np.array(bvals)
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        rows = []
                        bvals = []
                        for j in range(m):
                            row = np.array([U[i*m+j, r] * V[j*p+k, r] for r in range(target_rank)])
                            rows.append(row)
                            bvals.append(1.0)
                            for i2 in range(n):
                                for k2 in range(p):
                                    if i2 == i and k2 == k:
                                        continue
                                    row2 = np.array([U[i2*m+j, r] * V[j*p+k2, r] for r in range(target_rank)])
                                    rows.append(row2)
                                    bvals.append(0.0)
                        A = np.array(rows)
                        b = np.array(bvals)
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
            
            err = compute_error(U, V, W, target_rank)
            if err < 1e-10:
                for scale in [100000, 10000, 1000]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    if verify(U_r, V_r, W_r, target_rank):
                        result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_result = result
                        best_rank = target_rank
                        found = True
                        break
    
    return best_result
