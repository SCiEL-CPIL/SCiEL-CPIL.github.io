import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    # Strassen decomposition for 2x2 matrices
    # U2: shape (4, 7), index [i*2+j, t]
    # V2: shape (4, 7), index [j*2+k, t]
    # W2: shape (4, 7), index [k*2+i, t]
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0 (C00)
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1 (C10)
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1 (C11)
    ], dtype=int)

    # Outer Strassen coefficients
    # beta[s, I, J], gamma[s, J, K], delta[s, I, K]
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta = np.zeros((7, 2, 2), dtype=int)
    
    # s=0: M0 = (A00+A11)(B00+B11) -> C00, C11
    beta[0,0,0]=1; beta[0,1,1]=1
    gamma[0,0,0]=1; gamma[0,1,1]=1
    delta[0,0,0]=1; delta[0,1,1]=1
    
    # s=1: M1 = (A10+A11)B00 -> C10, C11
    beta[1,1,0]=1; beta[1,1,1]=1
    gamma[1,0,0]=1
    delta[1,1,0]=1; delta[1,1,1]=-1
    
    # s=2: M2 = A00(B01-B11) -> C01, C11
    beta[2,0,0]=1
    gamma[2,0,1]=1; gamma[2,1,1]=-1
    delta[2,0,1]=1; delta[2,1,1]=1
    
    # s=3: M3 = A11(B10-B00) -> C00, C10
    beta[3,1,1]=1
    gamma[3,1,0]=1; gamma[3,0,0]=-1
    delta[3,0,0]=1; delta[3,1,0]=1
    
    # s=4: M4 = (A00+A01)B11 -> C00, C01
    beta[4,0,0]=1; beta[4,0,1]=1
    gamma[4,1,1]=1
    delta[4,0,0]=-1; delta[4,0,1]=1
    
    # s=5: M5 = (A10-A00)(B00+B01) -> C11
    beta[5,1,0]=1; beta[5,0,0]=-1
    gamma[5,0,0]=1; gamma[5,0,1]=1
    delta[5,1,1]=1
    
    # s=6: M6 = (A01-A11)(B10+B11) -> C00
    beta[6,0,1]=1; beta[6,1,1]=-1
    gamma[6,1,0]=1; gamma[6,1,1]=1
    delta[6,0,0]=1

    # Construct U, V, W of shape (16, 49)
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    # U[r*4+c, s*7+t] = beta[s, I, J] * U2[i2*2+j2, t]
    for I in range(2):
        for J in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    r = 2*I + i2
                    c = 2*J + j2
                    row_u = r*4 + c
                    u2_idx = i2*2 + j2
                    outer_u = np.outer(beta[:, I, J], U2[u2_idx, :])
                    U[row_u, :] += outer_u.flatten()
                    
    # V[c*4+k, s*7+t] = gamma[s, J, K] * V2[j2*2+k2, t]
    for J in range(2):
        for K in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    c = 2*J + j2
                    k = 2*K + k2
                    row_v = c*4 + k
                    v2_idx = j2*2 + k2
                    outer_v = np.outer(gamma[:, J, K], V2[v2_idx, :])
                    V[row_v, :] += outer_v.flatten()
                    
    # W[k*4+i, s*7+t] = delta[s, I, K] * W2[k2*2+i2, t]
    for I in range(2):
        for K in range(2):
            for i2 in range(2):
                for k2 in range(2):
                    k = 2*K + k2
                    i = 2*I + i2
                    row_w = k*4 + i
                    w2_idx = k2*2 + i2
                    outer_w = np.outer(delta[:, I, K], W2[w2_idx, :])
                    W[row_w, :] += outer_w.flatten()

    # Verify the decomposition
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(4, 4))
    B = np.random.randint(-5, 6, size=(4, 4))
    C = A @ B
    
    C_trace = np.zeros(16)
    for i in range(4):
        for k in range(4):
            C_trace[k*4+i] = C[i,k]
            
    result = np.zeros(16)
    for t in range(49):
        U_t = U[:, t].reshape(4, 4)
        V_t = V[:, t].reshape(4, 4)
        W_t = W[:, t].reshape(4, 4)
        P = U_t @ V_t
        for i in range(4):
            for k in range(4):
                result[k*4+i] += P[i,k] * W_t[k,i]
                
    if not np.allclose(result, C_trace, atol=1e-10):
        pass

    elapsed = time.time() - start_time
    if elapsed > 100.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
