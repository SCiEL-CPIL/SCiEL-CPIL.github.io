import time
import numpy as np

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Rank 98 decomposition using Kronecker product of Strassen (2x2x2) and Rank-14 (2x2x4)
    rank = 98
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen vectors for 2x2x2
    # u: A (4), v: B (4), w: C (4) with W index k*2+i
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1: C00(0), C11(3)
        [0, 1, 0, -1],# M2: C10(1), C11(3)
        [0, 0, 1, 1], # M3: C01(2), C11(3)
        [1, 1, 0, 0], # M4: C00(0), C10(1)
        [-1, 0, 1, 0],# M5: C00(0), C01(2)
        [0, 0, 0, 1], # M6: C11(3)
        [-1, 0, 0, 0] # M7: C00(0)
    ]
    
    # Rank-14 vectors for 2x2x4
    # This is two copies of Strassen, one for B cols 0-1, one for B cols 2-3
    # U_A, V_A, W_A dimensions: U(4), V(8), W(8)
    U_A = []
    V_A = []
    W_A = []
    
    for c in range(7):
        # Left block (cols 0-1 of B, cols 0-1 of C)
        U_A.append(strassen_u[c])
        V_A.append(strassen_v[c] + [0, 0, 0, 0]) # B indices 0-3
        W_A.append(strassen_w[c] + [0, 0, 0, 0]) # W indices 0-3 (k=0,1)
        
    for c in range(7):
        # Right block (cols 2-3 of B, cols 2-3 of C)
        U_A.append(strassen_u[c])
        V_A.append([0, 0, 0, 0] + strassen_v[c]) # B indices 4-7
        W_A.append([0, 0, 0, 0] + strassen_w[c]) # W indices 4-7 (k=2,3)
        
    # Inner 2x2x2 vectors (B part of Kronecker)
    # U_B, V_B, W_B dimensions: U(4), V(4), W(4)
    U_B = strassen_u
    V_B = strassen_v
    W_B = strassen_w
    
    # Construct full U, V, W
    col = 0
    for c_A in range(14):
        for c_B in range(7):
            # U: 16 entries
            # i = i_A*2 + i_B, j = j_A*2 + j_B
            # idx = i*4 + j = (2*i_A + i_B)*4 + (2*j_A + j_B) = 8*i_A + 4*i_B + 2*j_A + j_B
            for i_A in range(2):
                for i_B in range(2):
                    for j_A in range(2):
                        for j_B in range(2):
                            idx_U = 8*i_A + 4*i_B + 2*j_A + j_B
                            idx_U_A = 2*i_A + j_A
                            idx_U_B = 2*i_B + j_B
                            U[idx_U, col] = U_A[c_A][idx_U_A] * U_B[c_B][idx_U_B]
            
            # V: 32 entries
            # j = j_A*2 + j_B, k = k_A*2 + k_B
            # idx = j*8 + k = (2*j_A + j_B)*8 + (2*k_A + k_B) = 16*j_A + 8*j_B + 2*k_A + k_B
            for j_A in range(2):
                for j_B in range(2):
                    for k_A in range(4):
                        for k_B in range(2):
                            idx_V = 16*j_A + 8*j_B + 2*k_A + k_B
                            idx_V_A = 4*j_A + k_A
                            idx_V_B = 2*j_B + k_B
                            V[idx_V, col] = V_A[c_A][idx_V_A] * V_B[c_B][idx_V_B]
                            
            # W: 32 entries
            # k = k_A*2 + k_B, i = i_A*2 + i_B
            # idx = k*4 + i = (2*k_A + k_B)*4 + (2*i_A + i_B) = 8*k_A + 4*k_B + 2*i_A + i_B
            for k_A in range(4):
                for k_B in range(2):
                    for i_A in range(2):
                        for i_B in range(2):
                            idx_W = 8*k_A + 4*k_B + 2*i_A + i_B
                            idx_W_A = 2*k_A + i_A
                            idx_W_B = 2*k_B + i_B
                            W[idx_W, col] = W_A[c_A][idx_W_A] * W_B[c_B][idx_W_B]
                            
            col += 1
            
    if time.time() > deadline:
        # Fallback to schoolbook if timeout (should not happen)
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
                    if time.time() > deadline:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
