import numpy as np

def construct():
    n, m, p = 2, 4, 5
    
    for rank in range(10, 40):
        for trial in range(100):
            np.random.seed(trial)
            U = np.random.randint(-2, 3, size=(n*m, rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, rank)).astype(float)
            
            # Compute M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            M = np.zeros((n, p, rank))
            for i in range(n):
                for k in range(p):
                    for c in range(rank):
                        M[i,k,c] = np.dot(U[i*m:(i+1)*m, c], V[:, k])
                        # Wait, V is (m*p, rank), so V[j*p+k, c] for j in range(m)
                        # Let me fix this
            # Recompute M properly
            M = np.zeros((n, p, rank))
            for i in range(n):
                for k in range(p):
                    for c in range(rank):
                        s = 0
                        for j in range(m):
                            s += U[i*m + j, c] * V[j*p + k, c]
                        M[i,k,c] = s
            
            # For each (i,k), solve M[i,k,:] @ W_row = 1
            W = np.zeros((p*n, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Find a c where M[i,k,c] != 0
                    nonzero_cs = [c for c in range(rank) if abs(M[i,k,c]) > 1e-9]
                    if not nonzero_cs:
                        valid = False
                        break
                    # Set W[k*n+i, c] = 0 for all c except the last nonzero one
                    c_last = nonzero_cs[-1]
                    W[k*n+i, c_last] = 1.0 / M[i,k,c_last]
                    # Others remain 0
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                U_list = [[round(float(U[i,j]), 1) for j in range(rank)] for i in range(n*m)]
                V_list = [[round(float(V[i,j]), 1) for j in range(rank)] for i in range(m*p)]
                W_list = [[round(float(W[i,j]), 1) for j in range(rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
