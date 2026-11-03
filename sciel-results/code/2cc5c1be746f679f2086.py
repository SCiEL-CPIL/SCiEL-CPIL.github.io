import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    best_result = None
    
    for rank in range(132, 145):
        if time.time() > deadline:
            break
        
        np.random.seed(42)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        found = False
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                    try:
                        U[idx_u, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    idx_v = j * p + k
                    A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                    try:
                        V[idx_v, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    idx_w = k * n + i
                    A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                    try:
                        W[idx_w, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            max_error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        max_error = max(max_error, abs(val - 1.0))
            
            if max_error < 1e-8:
                # Found a valid decomposition
                # Round to nearest 0.5
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.dot(U_int[idx_u, :], V_int[idx_v, :] * W_int[idx_w, :])
                            if abs(val - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_result = {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    found = True
                    break
                else:
                    # Try with higher precision
                    U_int = np.round(U * 1000000) / 1000000
                    V_int = np.round(V * 1000000) / 1000000
                    W_int = np.round(W * 1000000) / 1000000
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = np.dot(U_int[idx_u, :], V_int[idx_v, :] * W_int[idx_w, :])
                                if abs(val - 1.0) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_result = {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        found = True
                        break
        
        if found:
            break
    
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
