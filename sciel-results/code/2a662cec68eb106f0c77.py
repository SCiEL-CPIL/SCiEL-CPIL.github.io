import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _verify_solution(U, V, W, n, m, p):
    """Verify that the decomposition is correct by checking against random matrices."""
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for c in range(len(U)):
        u = np.array(U[c]).reshape(n, m)
        v = np.array(V[c]).reshape(m, p)
        w = np.array(W[c]).reshape(p, n)
        # The contribution is: trace(u^T * A) * trace(v^T * B) * w
        # Actually, the convention is:
        # C[i,k] += sum_j A[i,j] * B[j,k]
        # Each rank-1 term: U[i,j,c] * V[j,k,c] * W[k,i,c] contributes to C[i,k]
        # So C[i,k] += sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] for each j
        # More precisely: C[i,k] += sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i,k] += U[c][i*m+j] * V[c][j*p+k] * W[c][k*n+i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin
    
    # Strassen 2x2x2 rank-1 terms
    strassen_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,-1]),
        ([0,0,0,1], [0,0,1,-1], [0,0,1,-1]),
        ([1,1,0,0], [0,0,0,1], [1,1,0,0]),
        ([0,1,-1,0], [1,1,0,0], [0,1,1,-1]),
        ([0,1,0,-1], [0,0,1,1], [0,1,-1,1]),
    ]
    
    U = []
    V = []
    W = []
    
    # Part 1: 4x4x4 block Strassen (49 terms)
    for s_u, s_v, s_w in strassen_2x2:
        for t_u, t_v, t_w in strassen_2x2:
            U_vec = [0] * (n*m)
            V_vec = [0] * (m*p)
            W_vec = [0] * (p*n)
            
            for i1 in range(2):
                for j1 in range(2):
                    su_val = s_u[i1*2+j1]
                    if su_val != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                tu_val = t_u[i2*2+j2]
                                if tu_val != 0:
                                    U_vec[2*i1+i2, 2*j1+j2] = su_val * tu_val
            
            for j1 in range(2):
                for k1 in range(2):
                    sv_val = s_v[j1*2+k1]
                    if sv_val != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                tv_val = t_v[j2*2+k2]
                                if tv_val != 0:
                                    V_vec[2*j1+j2, 2*k1+k2] = sv_val * tv_val
            
            for k1 in range(2):
                for i1 in range(2):
                    sw_val = s_w[k1*2+i1]
                    if sw_val != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                tw_val = t_w[k2*2+i2]
                                if tw_val != 0:
                                    W_vec[2*k1+k2, 2*i1+i2] = sw_val * tw_val
            
            U.append(U_vec)
            V.append(V_vec)
            W.append(W_vec)
    
    # Part 2: Schoolbook for 4x4x3 (48 terms)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U_vec = [0] * (n*m)
                V_vec = [0] * (m*p)
                W_vec = [0] * (p*n)
                U_vec[i*m+j] = 1
                V_vec[j*p+k] = 1
                W_vec[k*n+i] = 1
                U.append(U_vec)
                V.append(V_vec)
                W.append(W_vec)
    
    # Verify the solution
    if time.time() < deadline - 5:
        if not _verify_solution(U, V, W, n, m, p):
            # Fall back to schoolbook
            U = []
            V = []
            W = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_vec = [0] * (n*m)
                        V_vec = [0] * (m*p)
                        W_vec = [0] * (p*n)
                        U_vec[i*m+j] = 1
                        V_vec[j*p+k] = 1
                        W_vec[k*n+i] = 1
                        U.append(U_vec)
                        V.append(V_vec)
                        W.append(W_vec)
    
    return {"U": U, "V": V, "W": W}
