import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    rank = 72
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U entries (i, j) for i in 0,1 and j in 0,1
    # V entries (j, k) for j in 0,1 and k in 0,1
    # W entries (k, i) for k in 0,1 and i in 0,1
    
    # We define the 7 ranks for Strassen
    # Each rank is a triple (u_vec, v_vec, w_vec)
    # u_vec is length 4 (2x2), v_vec is length 4 (2x2), w_vec is length 4 (2x2)
    
    strassen_u = [
        [1, 0, 1, 0], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [1, 0, -1, 0],# M6: A10, -A00 (wait, A10-A00 -> indices 1,0 and 0,0)
        [0, 1, 0, -1] # M7: A01, -A11 (wait, A01-A11 -> indices 0,1 and 1,1)
    ]
    # Correction on indices for u_vec (i*m+j local):
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    
    strassen_u = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [1, 0, -1, 0],# M6: A00, A10 (A10-A00) -> indices 2, 0. So [ -1, 0, 1, 0]
        [0, 1, 0, -1] # M7: A01, A11 (A01-A11) -> indices 1, 3. So [ 0, 1, 0, -1]
    ]
    # Let's re-verify M6: (A10 - A00)(B00 + B01)
    # A part: A10 (idx 2) - A00 (idx 0). So u = [-1, 0, 1, 0]
    # B part: B00 (idx 0) + B01 (idx 1). So v = [1, 1, 0, 0]
    # C part: C10 (idx 2) = M6. So w = [0, 0, 1, 0]
    
    # M7: (A01 - A11)(B10 + B11)
    # A part: A01 (idx 1) - A11 (idx 3). So u = [0, 1, 0, -1]
    # B part: B10 (idx 2) + B11 (idx 3). So v = [0, 0, 1, 1]
    # C part: C11 (idx 3) = M7. So w = [0, 0, 0, 1]
    
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
        [1, 1, 0, 0], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01, -B11
        [0, 0, 1, -1],# M4: B10, -B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B01
        [0, 0, 1, 1]  # M7: B10, B11
    ]
    
    strassen_w = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, 0], # M2: C10
        [0, 1, 0, 0], # M3: C01
        [1, 0, 1, 0], # M4: C00, C10
        [0, 1, 0, -1],# M5: C01, -C11
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 1]  # M7: C00, C11
    ]
    # Check W indices:
    # C00 (idx 0), C01 (idx 1), C10 (idx 2), C11 (idx 3)
    # M1: C00 + C11 -> [1, 0, 0, 1]
    # M2: C10 -> [0, 0, 1, 0]
    # M3: C01 -> [0, 1, 0, 0]
    # M4: C00 + C10 -> [1, 0, 1, 0]
    # M5: C01 - C11 -> [0, 1, 0, -1]
    # M6: C11 -> [0, 0, 0, 1]
    # M7: C00 + C11 -> [1, 0, 0, 1]
    
    # Schoolbook for 1x2x2 (i=2)
    # Rank 2
    # r0: A20 B0k -> U[2,0]=1, V[0,k]=1, W[k,2]=1
    # r1: A21 B1k -> U[2,1]=1, V[1,k]=1, W[k,2]=1
    
    # Local indices for 1x2x2:
    # i=2 (global), j in 0,1, k in 0,1
    # U local: [0, 0, 0, 0] for r0? No, U is 1x2.
    # Let's just define the global updates directly.
    
    for j_block in range(2):
        for k_block in range(4):
            # Strassen part (7 ranks)
            for s in range(7):
                # U updates
                for i_local in range(2):
                    for j_local in range(2):
                        val = strassen_u[s][i_local * 2 + j_local]
                        if val != 0:
                            i_global = i_local
                            j_global = j_block * 2 + j_local
                            u_idx = i_global * m + j_global
                            U[u_idx, col] = val
                            
                # V updates
                for j_local in range(2):
                    for k_local in range(2):
                        val = strassen_v[s][j_local * 2 + k_local]
                        if val != 0:
                            j_global = j_block * 2 + j_local
                            k_global = k_block * 2 + k_local
                            v_idx = j_global * p + k_global
                            V[v_idx, col] = val
                            
                # W updates
                for k_local in range(2):
                    for i_local in range(2):
                        val = strassen_w[s][k_local * 2 + i_local]
                        if val != 0:
                            k_global = k_block * 2 + k_local
                            i_global = i_local
                            w_idx = k_global * n + i_global
                            W[w_idx, col] = val
                            
                col += 1
                
            # Schoolbook part (2 ranks)
            # r0: j_local=0
            for j_local in range(2):
                # U
                i_global = 2
                j_global = j_block * 2 + j_local
                u_idx = i_global * m + j_global
                U[u_idx, col] = 1
                
                # V
                for k_local in range(2):
                    k_global = k_block * 2 + k_local
                    v_idx = j_global * p + k_global
                    V[v_idx, col] = 1
                    
                # W
                for k_local in range(2):
                    k_global = k_block * 2 + k_local
                    w_idx = k_global * n + i_global
                    W[w_idx, col] = 1
                    
                col += 1
                
    # Convert to list of lists
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

if __name__ == "__main__":
    res = construct()
    # Quick check
    n, m, p = 3, 4, 8
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    # Verify
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Check T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m+j]
                v_val = V[j*p+k]
                w_val = W[k*n+i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > max_err:
                    max_err = abs(dot - 1.0)
    print(f"Max Error: {max_err}")
