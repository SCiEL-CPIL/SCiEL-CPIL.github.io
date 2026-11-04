import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen-based decomposition: 
    # Split m=4 into two blocks of size 2.
    # Split p=8 into four blocks of size 2.
    # This gives 2 * 4 = 8 blocks of size 2x2x2.
    # Each 2x2x2 block uses 7 terms (Strassen).
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i*m + j][col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j*p + k][col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k*n + i][col_idx] += val

    col = 0
    
    # Loop over j-blocks (2 blocks of size 2)
    for j_block in [0, 2]:
        # Loop over k-blocks (4 blocks of size 2)
        for k_block in [0, 2, 4, 6]:
            # Strassen 7 terms for 2x2 * 2x2
            # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
            # Global indices: I = i, J = j_block + j, K = k_block + k
            
            # 1. (A00+A11)(B00+B11) -> C00, C11
            add_term({(0, j_block): 1, (1, j_block+1): 1}, 
                     {(j_block, k_block): 1, (j_block+1, k_block+1): 1}, 
                     {(k_block, 0): 1, (k_block+1, 1): 1}, col); col += 1
            
            # 2. (A10+A11)(B00) -> C10, C11
            add_term({(1, j_block): 1, (1, j_block+1): 1}, 
                     {(j_block, k_block): 1}, 
                     {(k_block, 1): 1, (k_block+1, 1): -1}, col); col += 1
            
            # 3. (A00)(B01-B11) -> C01, C11
            add_term({(0, j_block): 1}, 
                     {(j_block, k_block+1): 1, (j_block+1, k_block+1): -1}, 
                     {(k_block+1, 0): 1, (k_block+1, 1): 1}, col); col += 1
            
            # 4. (A11)(B10-B00) -> C00, C10
            add_term({(1, j_block+1): 1}, 
                     {(j_block+1, k_block): 1, (j_block, k_block): -1}, 
                     {(k_block, 0): 1, (k_block, 1): 1}, col); col += 1
            
            # 5. (A00+A01)(B11) -> C00, C01
            add_term({(0, j_block): 1, (0, j_block+1): 1}, 
                     {(j_block+1, k_block+1): 1}, 
                     {(k_block, 0): -1, (k_block+1, 0): 1}, col); col += 1
            
            # 6. (A10-A00)(B00+B01) -> C01
            add_term({(1, j_block): 1, (0, j_block): -1}, 
                     {(j_block, k_block): 1, (j_block, k_block+1): 1}, 
                     {(k_block+1, 1): 1}, col); col += 1
            
            # 7. (A01-A11)(B10+B11) -> C00
            add_term({(0, j_block+1): 1, (1, j_block+1): -1}, 
                     {(j_block+1, k_block): 1, (j_block+1, k_block+1): 1}, 
                     {(k_block, 0): 1}, col); col += 1

    # Validate the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook if validation fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U_np.tolist(),
        "V": V_np.tolist(),
        "W": W_np.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
