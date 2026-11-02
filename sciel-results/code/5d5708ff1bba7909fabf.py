import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen 2x2x2 decomposition (7 terms)
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,1,0,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    total_rank = 2 * 3 * 2 * 7  # = 84
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                for (u_local, v_local, w_local) in strassen_terms:
                    for la in range(4):
                        di = la // 2
                        dj = la % 2
                        a_global = (2*i + di) * m + (2*k + dj)
                        U[a_global, col] += u_local[la]
                    
                    for lb in range(4):
                        dj = lb // 2
                        dk = lb % 2
                        b_global = (2*k + dj) * p + (2*j + dk)
                        V[b_global, col] += v_local[lb]
                    
                    for lc in range(4):
                        dk = lc // 2
                        di = lc % 2
                        c_global = (2*j + dk) * n + (2*i + di)
                        W[c_global, col] += w_local[lc]
                    
                    col += 1
    
    # Verify
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(total_rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(T_recon, T, atol=1e-6):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
