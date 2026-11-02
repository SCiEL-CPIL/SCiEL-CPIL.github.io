"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2_2x2_2x2_factors():
    """
    Returns U, V, W factors for 2x2x2 Strassen decomposition.
    Shapes: (4, 7), (4, 7), (4, 7)
    Indices: u = i*2+j, v = j*2+k, w = k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_2x2_4_factors():
    """
    Construct factors for 2x2x4 multiplication.
    Rank 14.
    This is two copies of 2x2x2 Strassen.
    """
    U_s, V_s, W_s = _strassen_2x2_2x2_2x2_factors()
    
    U_d = np.zeros((4, 14))
    V_d = np.zeros((8, 14))
    W_d = np.zeros((8, 14))
    
    # First copy (k=0,1)
    U_d[:, 0:7] = U_s
    # V_d rows: j*4+k. j in {0,1}, k in {0,1}.
    # Rows: 0, 1, 4, 5.
    V_d[0, 0:7] = V_s[0, :]
    V_d[1, 0:7] = V_s[1, :]
    V_d[4, 0:7] = V_s[2, :]
    V_d[5, 0:7] = V_s[3, :]
    # W_d rows: k*2+i. k in {0,1}, i in {0,1}.
    # Rows: 0, 1, 2, 3.
    W_d[0:4, 0:7] = W_s
    
    # Second copy (k=2,3)
    U_d[:, 7:14] = U_s
    # V_d rows: j*4+k. j in {0,1}, k in {2,3}.
    # Rows: 2, 3, 6, 7.
    V_d[2, 7:14] = V_s[0, :]
    V_d[3, 7:14] = V_s[1, :]
    V_d[6, 7:14] = V_s[2, :]
    V_d[7, 7:14] = V_s[3, :]
    # W_d rows: k*2+i. k in {2,3}, i in {0,1}.
    # Rows: 4, 5, 6, 7.
    W_d[4:8, 7:14] = W_s
    
    return U_d, V_d, W_d

def _construct_4x4_8_factors():
    """
    Construct factors for 4x4x8 multiplication using Kronecker product of 2x2x2 and 2x2x4.
    Rank 98.
    """
    U_s, V_s, W_s = _strassen_2x2_2x2_2x2_factors()
    U_d, V_d, W_d = _construct_2x2_4_factors()
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # n=4, m=4, p=8
    # i = i_s * 2 + i_d
    # j = j_s * 2 + j_d
    # k = k_s * 4 + k_d
    
    for r_s in range(7):
        for r_d in range(14):
            r = r_s * 14 + r_d
            
            # U
            for i_s in range(2):
                for j_s in range(2):
                    u_s_val = U_s[i_s * 2 + j_s, r_s]
                    if u_s_val == 0:
                        continue
                    for i_d in range(2):
                        for j_d in range(2):
                            u_d_val = U_d[i_d * 2 + j_d, r_d]
                            if u_d_val == 0:
                                continue
                            i = i_s * 2 + i_d
                            j = j_s * 2 + j_d
                            U[i * 4 + j, r] = u_s_val * u_d_val
            
            # V
            for j_s in range(2):
                for k_s in range(2):
                    v_s_val = V_s[j_s * 2 + k_s, r_s]
                    if v_s_val == 0:
                        continue
                    for j_d in range(2):
                        for k_d in range(4):
                            v_d_val = V_d[j_d * 4 + k_d, r_d]
                            if v_d_val == 0:
                                continue
                            j = j_s * 2 + j_d
                            k = k_s * 4 + k_d
                            V[j * 8 + k, r] = v_s_val * v_d_val
            
            # W
            for k_s in range(2):
                for i_s in range(2):
                    w_s_val = W_s[k_s * 2 + i_s, r_s]
                    if w_s_val == 0:
                        continue
                    for k_d in range(4):
                        for i_d in range(2):
                            w_d_val = W_d[k_d * 2 + i_d, r_d]
                            if w_d_val == 0:
                                continue
                            k = k_s * 4 + k_d
                            i = i_s * 2 + i_d
                            W[k * 4 + i, r] = w_s_val * w_d_val
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook rank 128
    best = _schoolbook(n, m, p)
    best_rank = 128
    
    try:
        U, V, W = _construct_4x4_8_factors()
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best
    
    return best
# EVOLVE-BLOCK-END
