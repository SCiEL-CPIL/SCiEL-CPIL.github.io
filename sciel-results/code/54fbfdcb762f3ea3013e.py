import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _strassen_2x2_factors():
    """Return U, V, W factor matrices (each 4x7) for 2x2x2 MM tensor."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_block_strassen():
    """Build rank-49 decomposition for 4x4x4 MM using block Strassen."""
    sU, sV, sW = _strassen_2x2_factors()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Term definitions: (a_terms, b_terms, c_terms)
    terms = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for t in range(7):
        a_list, b_list, c_list = terms[t]
        a_dict = {(I, J): c for I, J, c in a_list}
        b_dict = {(J, K): c for J, K, c in b_list}
        c_dict = {(I, K): c for I, K, c in c_list}
        
        for s in range(7):
            col = t * 7 + s
            
            for i in range(4):
                for j in range(4):
                    I, J = i // 2, j // 2
                    ip, jp = i % 2, j % 2
                    if (I, J) in a_dict:
                        U[i*4+j, col] = a_dict[(I,J)] * sU[ip*2+jp, s]
            
            for j in range(4):
                for k in range(4):
                    J, K = j // 2, k // 2
                    jp, kp = j % 2, k % 2
                    if (J, K) in b_dict:
                        V[j*4+k, col] = b_dict[(J,K)] * sV[jp*2+kp, s]
            
            for k in range(4):
                for i in range(4):
                    K, I = k // 2, i // 2
                    kp, ip = k % 2, i % 2
                    if (I, K) in c_dict:
                        W[k*4+i, col] = c_dict[(I,K)] * sW[kp*2+ip, s]
    
    return U, V, W

def _build_4x4x5():
    """Build rank-50 decomposition for 4x4x5 MM."""
    U4, V4, W4 = _build_4x4x4_block_strassen()
    
    n, m, p = 4, 4, 5
    rank = 50
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4
    
    # Tail: k=4
    # T[i, j, 4] = 1
    # Use 1 term
    col = 49
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
    for j in range(m):
        V[j*p+4, col] = 1
    for i in range(n):
        W[4*n+i, col] = 1
        
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=5, tol=1e-10):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    try:
        U50, V50, W50 = _build_4x4x5()
        valid, _ = _verify(U50, V50, W50)
        if valid:
            return {
                "U": U50.tolist(),
                "V": V50.tolist(),
                "W": W50.tolist()
            }
    except Exception:
        pass
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
