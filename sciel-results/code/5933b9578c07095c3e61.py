import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 4, 4, 6
    
    # Strassen coefficients for 2x2x2 (Rank 7)
    # Terms are (U_dict, V_dict, W_dict)
    # Keys are (row, col) or (col, row) depending on tensor factor
    # U: (i, j) for A[i,j]
    # V: (j, k) for B[j,k]
    # W: (k, i) for C[k,i] (trace convention W[k*n+i])
    
    strassen_terms = [
        # 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # 3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # 4
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # 5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # 6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}),
        # 7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Hybrid decomposition for 2x2x3 (Rank 11)
    # Strassen for k=0,1 (7 terms)
    # Schoolbook for k=2 (4 terms)
    
    def get_223_hybrid_terms():
        terms = []
        # Add Strassen terms for k=0,1
        for t in strassen_terms:
            a, b, c = t
            # c keys are (k, i) with k in {0,1}
            # We keep them as is
            terms.append((a, b, c))
        
        # Add Schoolbook terms for k=2
        # A[i,j] * B[j,2] * C[2,i]
        for i in range(2):
            for j in range(2):
                a = {(i,j): 1}
                b = {(j,2): 1}
                c = {(2,i): 1}
                terms.append((a, b, c))
        
        return terms

    terms_223 = get_223_hybrid_terms()
    
    # Kronecker product of 2x2x2 (Rank 7) and 2x2x3 (Rank 11)
    # Total Rank = 7 * 11 = 77
    
    rank_total = 77
    U = np.zeros((16, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    col = 0
    for t1 in strassen_terms:
        a1, b1, c1 = t1
        for t2 in terms_223:
            a2, b2, c2 = t2
            
            # U: (i1*n2+i2, j1*m2+j2)
            for (i1, j1), c_a1 in a1.items():
                for (i2, j2), c_a2 in a2.items():
                    U[i1*2+i2, j1*2+j2, col] += c_a1 * c_a2
            
            # V: (j1*m2+j2, k1*p2+k2)
            for (j1, k1), c_b1 in b1.items():
                for (j2, k2), c_b2 in b2.items():
                    V[j1*2+j2, k1*3+k2, col] += c_b1 * c_b2
            
            # W: (k1*p2+k2, i1*n2+i2)
            for (k1, i1), c_c1 in c1.items():
                for (k2, i2), c_c2 in c2.items():
                    W[k1*3+k2, i1*2+i2, col] += c_c1 * c_c2
            
            col += 1
    
    # Verification
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j, :], V[j*p+k, :]) # This is wrong, need element-wise product then sum
                    # total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1) > 1e-9:
                        return False
        return True

    if verify(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1
    return {
        "U": U_s.tolist(),
        "V": V_s.tolist(),
        "W": W_s.tolist()
    }
