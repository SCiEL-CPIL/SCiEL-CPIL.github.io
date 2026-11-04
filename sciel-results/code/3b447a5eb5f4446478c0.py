import time
import numpy as np

START = time.time()
DEADLINE = 140.0

def _check_time():
    return time.time() >= DEADLINE

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    r = 7
    U = np.zeros((4, r), dtype=int)
    V = np.zeros((4, r), dtype=int)
    W = np.zeros((4, r), dtype=int)
    
    # p1 = a * e -> C00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # p2 = c * f -> C10 (k=0, i=1 -> idx 1)
    U[2, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
    # p3 = (a + b) * (f - h) -> C01 (k=1, i=0 -> idx 1)
    U[0, 2] = 1; U[1, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    # p4 = (d - c) * (e + f) -> C11 (k=1, i=1 -> idx 3)
    U[3, 3] = 1; U[2, 3] = -1
    V[0, 3] = 1; V[1, 3] = 1
    W[3, 3] = 1
    # p5 = d * h -> C11
    U[3, 4] = 1; V[3, 4] = 1; W[3, 4] = 1
    # p6 = (a - b) * (e + h) -> C00 (idx 0), C10 (idx 1, neg)
    U[0, 5] = 1; U[1, 5] = -1
    V[0, 5] = 1; V[3, 5] = 1
    W[0, 5] = 1; W[1, 5] = -1
    # p7 = (c + b) * (g + e) -> C10 (idx 1), C11 (idx 3)
    U[2, 6] = 1; U[1, 6] = 1
    V[2, 6] = 1; V[0, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n*m, r), dtype=int)
    V = np.zeros((m*p, r), dtype=int)
    W = np.zeros((p*n, r), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W

def _mat2x4x2():
    """
    2x4 x 4x2 -> 2x2.
    Rank 14.
    U shape (8, 14), V shape (8, 14), W shape (4, 14).
    U[i*4+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    U0, V0, W0 = _strassen_2x2()
    r = 14
    U = np.zeros((8, r), dtype=int)
    V = np.zeros((8, r), dtype=int)
    W = np.zeros((4, r), dtype=int)
    
    # Block 0: A cols 0,1; B rows 0,1
    U[0:4, 0:7] = U0
    V[0:4, 0:7] = V0
    W[:, 0:7] = W0
    
    # Block 1: A cols 2,3; B rows 2,3
    # U rows: i in 0,1. j in 2,3.
    # U[i*4 + (j+2)]
    U[2:4, 7:14] = 0 # clear
    U[0:4, 7:14] = 0 # clear
    # Actually, U0 is 4x7.
    # A[0,2] -> U[2, 7..13] = U0[0, 0..6]
    # A[0,3] -> U[3, 7..13] = U0[1, 0..6]
    # A[1,2] -> U[6, 7..13] = U0[2, 0..6]
    # A[1,3] -> U[7, 7..13] = U0[3, 0..6]
    U[2, 7:14] = U0[0]
    U[3, 7:14] = U0[1]
    U[6, 7:14] = U0[2]
    U[7, 7:14] = U0[3]
    
    # V rows: j in 2,3. k in 0,1.
    # V[(j+2)*2 + k]
    # V[4:8, 7:14] = V0
    V[4, 7:14] = V0[0] # j=2, k=0
    V[5, 7:14] = V0[1] # j=2, k=1
    V[6, 7:14] = V0[2] # j=3, k=0
    V[7, 7:14] = V0[3] # j=3, k=1
    
    # W is same for both blocks?
    # W[k*2+i]
    # Yes, output is 2x2.
    W[:, 7:14] = W0
    
    return U, V, W

def _mat4x4x2():
    """
    4x4 x 4x2 -> 4x2.
    Rank 28.
    U shape (16, 28), V shape (8, 28), W shape (8, 28).
    U[i*4+j, c]
    V[j*2+k, c]
    W[k*4+i, c]  <-- Note: W index is k*n+i with n=4.
    """
    U0, V0, W0 = _mat2x4x2()
    r = 28
    U = np.zeros((16, r), dtype=int)
    V = np.zeros((8, r), dtype=int)
    W = np.zeros((8, r), dtype=int)
    
    # Block 0: A rows 0,1.
    # U[0:8, 0:14] = U0
    U[0:8, 0:14] = U0
    V[0:8, 0:14] = V0
    # W for block 0: i in 0,1.
    # W[k*4 + i]
    # W0 is 4x14, indexed by k*2+i_local.
    # i_local 0 -> i_global 0. i_local 1 -> i_global 1.
    # k 0 -> W[0:4, 0:14] = W0[0:2, :] (k=0, i=0,1)
    # k 1 -> W[4:8, 0:14] = W0[2:4, :] (k=1, i=0,1)
    W[0, 0:14] = W0[0]
    W[1, 0:14] = W0[1]
    W[4, 0:14] = W0[2]
    W[5, 0:14] = W0[3]
    
    # Block 1: A rows 2,3.
    # U[8:16, 14:28] = U0
    U[8:16, 14:28] = U0
    V[0:8, 14:28] = V0
    # W for block 1: i in 2,3.
    # W[k*4 + i]
    # i 2 -> W[2, ...], i 3 -> W[3, ...]
    # k 0 -> W[2, 14:28] = W0[0], W[3, 14:28] = W0[1]
    # k 1 -> W[6, 14:28] = W0[2], W[7, 14:28] = W0[3]
    W[2, 14:28] = W0[0]
    W[3, 14:28] = W0[1]
    W[6, 14:28] = W0[2]
    W[7, 14:28] = W0[3]
    
    return U, V, W

def _mat4x4x4():
    """
    4x4 x 4x4 -> 4x4.
    Rank 28.
    U shape (16, 28), V shape (16, 28), W shape (16, 28).
    U[i*4+j, c]
    V[j*4+k, c]
    W[k*4+i, c]
    """
    U0, V0, W0 = _strassen_2x2()
    r = 28
    U = np.zeros((16, r), dtype=int)
    V = np.zeros((16, r), dtype=int)
    W = np.zeros((16, r), dtype=int)
    
    for bi in range(2):
        for bj in range(2):
            base = (bi*2 + bj) * 7
            
            # U
            # A rows bi*2, bi*2+1. Cols bj*2, bj*2+1.
            # U[i*4+j]
            # i=0 (local) -> i_global=bi*2. j=0 (local) -> j_global=bj*2.
            U[bi*2*4 + bj*2, base:base+7] = U0[0]
            U[bi*2*4 + bj*2 + 1, base:base+7] = U0[1]
            U[(bi*2+1)*4 + bj*2, base:base+7] = U0[2]
            U[(bi*2+1)*4 + bj*2 + 1, base:base+7] = U0[3]
            
            # V
            # B rows bj*2, bj*2+1. Cols bi*2, bi*2+1.
            # V[j*4+k]
            V[bj*2*4 + bi*2, base:base+7] = V0[0]
            V[bj*2*4 + bi*2 + 1, base:base+7] = V0[1]
            V[(bj*2+1)*4 + bi*2, base:base+7] = V0[2]
            V[(bj*2+1)*4 + bi*2 + 1, base:base+7] = V0[3]
            
            # W
            # C rows bi*2, bi*2+1. Cols bj*2, bj*2+1.
            # W[k*4+i]
            # k=0 (local) -> k_global=bj*2. i=0 (local) -> i_global=bi*2.
            W[bj*2*4 + bi*2, base:base+7] = W0[0]
            W[bj*2*4 + bi*2 + 1, base:base+7] = W0[1]
            W[(bj*2+1)*4 + bi*2, base:base+7] = W0[2]
            W[(bj*2+1)*4 + bi*2 + 1, base:base+7] = W0[3]
            
    return U, V, W

def construct():
    if _check_time():
        # Fallback schoolbook
        n, m, p = 4, 5, 6
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # _mat4x4x4 returns (rank, U, V, W) but we defined it to return (U, V, W)
    # Let me check the return statements...
    
    # Actually, looking at the code, _mat4x4x4 returns U, V, W (3 values)
    # But we're unpacking as r1, U1, V1, W1 (4 values) - this is a bug!
    
    U1, V1, W1 = _mat4x4x4()
    r1 = U1.shape[1]
    
    U2, V2, W2 = _schoolbook(4, 1, 4)
    r2 = U2.shape[1]
    
    U3, V3, W3 = _mat4x4x2()
    r3 = U3.shape[1]
    
    U4, V4, W4 = _schoolbook(4, 1, 2)
    r4 = U4.shape[1]
    
    total_rank = r1 + r2 + r3 + r4
    
    U_g = np.zeros((20, total_rank), dtype=int)
    V_g = np.zeros((30, total_rank), dtype=int)
    W_g = np.zeros((24, total_rank), dtype=int)
    
    base = 0
    # 1: 4x4x4
    U_g[0:16, base:base+r1] = U1
    V_g[0:16, base:base+r1] = V1
    W_g[0:16, base:base+r1] = W1
    base += r1
    
    # 2: 4x1x4
    # A col 4. B row 4.
    # U[i*5+4]
    # V[4*6+k]
    # W[k*4+i]
    U_g[4::5, base:base+r2] = U2 # U2 is 4x16. Rows i=0..3.
    # U2[i, c] -> U_g[i*5+4, base+c]
    # U2 shape (4, 16).
    # U_g[4, base:base+16] = U2[0]
    # U_g[9, base:base+16] = U2[1]
    # U_g[14, base:base+16] = U2[2]
    # U_g[19, base:base+16] = U2[3]
    U_g[4, base:base+r2] = U2[0]
    U_g[9, base:base+r2] = U2[1]
    U_g[14, base:base+r2] = U2[2]
    U_g[19, base:base+r2] = U2[3]
    
    # V[4*6+k] -> V[24+k]
    # V2 shape (4, 16). Rows k=0..3.
    V_g[24, base:base+r2] = V2[0]
    V_g[25, base:base+r2] = V2[1]
    V_g[26, base:base+r2] = V2[2]
    V_g[27, base:base+r2] = V2[3]
    
    # W[k*4+i]
    # W2 shape (16, 16). Rows k*4+i.
    W_g[0:16, base:base+r2] = W2
    base += r2
    
    # 3: 4x4x2
    # A cols 0..3. B cols 4,5.
    # U[0:16, base:base+r3] = U3
    U_g[0:16, base:base+r3] = U3
    # V[j*6+k_g] where k_g in 4,5.
    # V3 shape (8, 28). Rows j*2+k_local (k_local 0,1).
    # k_g = k_local + 4.
    # V3[0] (j=0,k=0) -> V_g[0*6+4] = V_g[4]
    # V3[1] (j=0,k=1) -> V_g[0*6+5] = V_g[5]
    # V3[2] (j=1,k=0) -> V_g[1*6+4] = V_g[10]
    # V3[3] (j=1,k=1) -> V_g[1*6+5] = V_g[11]
    # ...
    for j in range(4):
        V_g[j*6+4, base:base+r3] = V3[j*2]
        V_g[j*6+5, base:base+r3] = V3[j*2+1]
        
    # W[k_g*4+i] where k_g in 4,5.
    # W3 shape (8, 28). Rows k_local*4+i.
    # k_g = k_local + 4.
    # W3[0] (k=0,i=0) -> W_g[4*4+0] = W_g[16]
    # W3[1] (k=0,i=1) -> W_g[4*4+1] = W_g[17]
    # ...
    for k_local in range(2):
        for i in range(4):
            W_g[(k_local+4)*4+i, base:base+r3] = W3[k_local*4+i]
    base += r3
    
    # 4: 4x1x2
    # A col 4. B row 4, cols 4,5.
    # U[4::5]
    U_g[4, base:base+r4] = U4[0]
    U_g[9, base:base+r4] = U4[1]
    U_g[14, base:base+r4] = U4[2]
    U_g[19, base:base+r4] = U4[3]
    
    # V[24+k_g]
    V_g[24+4, base:base+r4] = V4[0]
    V_g[24+5, base:base+r4] = V4[1]
    
    # W[(k_g)*4+i]
    W_g[16:24, base:base+r4] = W4
    base += r4
    
    assert base == total_rank
    
    # Verify
    n, m, p = 4, 5, 6
    
    # Check all non-zero entries of the expected tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                w_idx = k*n + i
                
                val = np.dot(U_g[a_idx], V_g[b_idx] * W_g[w_idx])
                if val != 1:
                    raise AssertionError(f"Verification failed at ({i},{j},{k}): got {val}")
    
    # Check some zero entries to ensure no spurious contributions
    import random
    random.seed(42)
    for _ in range(1000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        w = random.randint(0, p*n-1)
        
        # Skip if this is a non-zero entry
        # a = i*m+j, b = j*p+k, w = k*n+i
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_w = w // n
        i_w = w % n
        
        if i_a == i_w and j_a == j_b and k_b == k_w:
            continue
        
        val = np.dot(U_g[a], V_g[b] * W_g[w])
        if val != 0:
            raise AssertionError(f"Non-zero at ({a},{b},{w}): got {val}")
    
    return {"U": U_g.tolist(), "V": V_g.tolist(), "W": W_g.tolist()}
