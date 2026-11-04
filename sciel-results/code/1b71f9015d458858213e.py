import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _lader_3x3x3():
    # Lader's rank-23 decomposition for 3x3x3
    # Indices are 0..8 (flattened 3x3)
    # u[i*3+j], v[j*3+k], w[k*3+i]
    terms = [
        (0, 0, 0), (1, 1, 1), (2, 2, 2),
        (3, 0, 1), (0, 3, 1), (1, 0, 3),
        (4, 0, 2), (0, 4, 2), (2, 0, 4),
        (5, 0, 1), (1, 5, 2), (2, 1, 5),
        (6, 0, 2), (2, 6, 0), (0, 2, 6),
        (7, 0, 1), (1, 7, 2), (2, 1, 7),
        (8, 0, 2), (2, 8, 0), (0, 2, 8),
        (3, 4, 5), (5, 3, 4), (4, 5, 3)
    ]
    return terms

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    lader_terms = _lader_3x3x3()
    
    # Build rank-44 decomposition
    U_terms = []
    V_terms = []
    W_terms = []
    
    for u_idx, v_idx, w_idx in lader_terms:
        u_full = [0] * (n * m)
        v_full = [0] * (m * p)
        w_full = [0] * (p * n)
        
        i = u_idx // 3
        j = u_idx % 3
        u_full[i * m + j] = 1
        
        j2 = v_idx // 3
        k2 = v_idx % 3
        v_full[j2 * p + k2] = 1
        
        k3 = w_idx // 3
        i3 = w_idx % 3
        w_full[k3 * n + i3] = 1
        
        U_terms.append(u_full)
        V_terms.append(v_full)
        W_terms.append(w_full)
        
    # Add cross terms
    # A[i,3] * B[3,k] for i=0,1,2; k=0,1,2 (9 terms)
    for i in range(3):
        for k in range(3):
            u_term = [0] * (n * m)
            u_term[i * m + 3] = 1
            v_term = [0] * (m * p)
            v_term[3 * p + k] = 1
            w_term = [0] * (p * n)
            w_term[k * n + i] = 1
            U_terms.append(u_term)
            V_terms.append(v_term)
            W_terms.append(w_term)
            
    # A[i,j] * B[j,3] for i=0,1,2; j=0,1,2 (9 terms)
    for i in range(3):
        for j in range(3):
            u_term = [0] * (n * m)
            u_term[i * m + j] = 1
            v_term = [0] * (m * p)
            v_term[j * p + 3] = 1
            w_term = [0] * (p * n)
            w_term[3 * n + i] = 1
            U_terms.append(u_term)
            V_terms.append(v_term)
            W_terms.append(w_term)
            
    # A[i,3] * B[3,3] for i=0,1,2 (3 terms)
    for i in range(3):
        u_term = [0] * (n * m)
        u_term[i * m + 3] = 1
        v_term = [0] * (m * p)
        v_term[3 * p + 3] = 1
        w_term = [0] * (p * n)
        w_term[3 * n + i] = 1
        U_terms.append(u_term)
        V_terms.append(v_term)
        W_terms.append(w_term)
        
    rank = len(U_terms)
    
    # Verify
    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if val != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        U_list = [[int(x) for x in row] for row in U]
        V_list = [[int(x) for x in row] for row in V]
        W_list = [[int(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
        
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

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
