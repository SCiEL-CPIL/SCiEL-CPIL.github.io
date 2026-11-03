import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120

def construct():
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        T_decomp = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if u[i, j] == 0: continue
                    for k in range(p):
                        T_decomp[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        T_target = np.ones((n, m, p))
        return np.allclose(T_decomp, T_target, atol=1e-9)

    # Lader 3x3x3 terms
    terms_333 = [
        [1,0,0, 0,0,0, 0,0,0,  1,0,0, 0,0,0, 0,0,0,  1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0,  0,0,0, 0,1,0, 0,0,0,  0,0,0, 0,1,0, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,0,1],
        [0,1,0, 0,0,0, 0,0,0,  0,1,0, 0,0,0, 0,0,0,  0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,1, 0,0,0,  0,0,0, 0,0,1, 0,0,0,  0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,1,0,  0,0,0, 0,0,0, 0,1,0,  0,0,0, 0,0,0, 0,0,1],
        [1,0,0, 0,0,0, 0,0,0,  0,0,0, 0,0,1, 0,0,0,  0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,1,0, 0,0,0,  0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,1,0,  0,0,0, 0,0,0, 0,0,1],
        [0,0,0, 0,0,0, 1,0,0,  0,0,0, 0,0,0, 1,0,0,  1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,1,0,  0,0,0, 0,0,0, 0,1,0,  0,1,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,0,1,  0,0,1, 0,0,0, 0,0,0],
        [1,0,0, 0,0,0, 0,0,0,  0,0,0, 0,1,0, 0,0,0,  0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,1,0,  0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 1,0,0,  0,0,0, 0,0,0, 1,0,0,  0,0,0, 0,0,0, 1,0,0],
        [0,0,0, 0,0,0, 0,1,0,  0,0,0, 0,0,0, 0,1,0,  0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,0,1],
        [1,0,0, 0,0,0, 0,0,0,  0,0,0, 0,0,1, 0,0,0,  0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,1,0, 0,0,0,  0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,0,1],
        [0,0,0, 0,0,0, 0,0,1,  0,0,0, 0,0,0, 0,1,0,  0,0,0, 0,0,0, 0,0,1],
        [0,0,0, 0,0,0, 1,0,0,  0,0,0, 0,0,0, 1,0,0,  0,0,0, 0,0,0, 0,1,0],
        [0,0,0, 0,0,0, 0,1,0,  0,0,0, 0,0,0, 0,1,0,  0,0,0, 0,0,0, 0,0,1]
    ]
    
    # Construct 3x3x3 decomposition
    U_333 = np.array([t[0:9] for t in terms_333]).T
    V_333 = np.array([t[9:18] for t in terms_333]).T
    W_333 = np.array([t[18:27] for t in terms_333]).T
    
    # Verify 3x3x3
    # We need to verify that these terms decompose the 3x3x3 multiplication tensor.
    # T[i,j,k] = 1
    # Check: sum_r U_333[i,j,r] * V_333[j,k,r] * W_333[k,i,r] == 1
    
    is_valid_333 = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = 0
                for r in range(23):
                    val += U_333[i*3+j, r] * V_333[j*3+k, r] * W_333[k*3+i, r]
                if not np.isclose(val, 1.0, atol=1e-9):
                    is_valid_333 = False
                    break
            if not is_valid_333: break
        if not is_valid_333: break
    
    if not is_valid_333:
        return get_schoolbook()
        
    # Construct 3x3x6 decomposition
    # Rank 46
    U = np.zeros((9, 46))
    V = np.zeros((18, 46))
    W = np.zeros((18, 46))
    
    # First 23 terms: A * B[:, 0:3]
    U[:, 0:23] = U_333
    V[0:9, 0:23] = V_333
    W[0:9, 0:23] = W_333
    
    # Last 23 terms: A * B[:, 3:6]
    U[:, 23:46] = U_333
    V[9:18, 23:46] = V_333
    W[9:18, 23:46] = W_333
    
    # Verify full 3x3x6
    if verify(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return get_schoolbook()

# EVOLVE-BLOCK-START
# The construct function is defined above to respect the interface.
# EVOLVE-BLOCK-END
