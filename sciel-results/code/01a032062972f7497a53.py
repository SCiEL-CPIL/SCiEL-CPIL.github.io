import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    # Schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try to find a better rank
    # We will try rank 50
    R = 50
    for trial in range(10):
        if time.time() > deadline:
            break
            
        U_rand = np.random.randn(n * m, R)
        V_rand = np.random.randn(m * p, R)
        W_rand = np.random.randn(p * n, R)
        
        # Simple check with random A, B
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        U_r = U_rand.reshape(n, m, R)
        V_r = V_rand.reshape(m, p, R)
        W_r = W_rand.reshape(p, n, R)
        
        u_vals = np.einsum('ij,ijr->r', A, U_r)
        v_vals = np.einsum('jk,jkr->r', B, V_r)
        C_decomp = np.einsum('r,r,ki->ki', u_vals, v_vals, W_r)
        
        # If error is very small, round and check
        if np.linalg.norm(C_true - C_decomp) < 1e-6:
             # Round to half-integers
             U_int = np.round(U_rand * 2) / 2
             V_int = np.round(V_rand * 2) / 2
             W_int = np.round(W_rand * 2) / 2
             
             # Verify
             A2 = np.random.randn(n, m)
             B2 = np.random.randn(m, p)
             C2_true = A2 @ B2
             
             U_r2 = U_int.reshape(n, m, R)
             V_r2 = V_int.reshape(m, p, R)
             W_r2 = W_int.reshape(p, n, R)
             
             u2 = np.einsum('ij,ijr->r', A2, U_r2)
             v2 = np.einsum('jk,jkr->r', B2, V_r2)
             C2_decomp = np.einsum('r,r,ki->ki', u2, v2, W_r2)
             
             if np.linalg.norm(C2_true - C2_decomp) < 1e-6:
                 best_rank = R
                 best_U = U_int.tolist()
                 best_V = V_int.tolist()
                 best_W = W_int.tolist()
                 break
                 
    return {"U": best_U, "V": best_V, "W": best_W}
