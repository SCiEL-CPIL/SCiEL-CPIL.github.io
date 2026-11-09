import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 17 Fallback
    rank_17 = 17
    U17 = np.zeros((n * m, rank_17))
    V17 = np.zeros((m * p, rank_17))
    W17 = np.zeros((p * n, rank_17))
    
    # ... (Keep the existing hardcoded U17, V17, W17) ...
    # I will copy the existing one to ensure correctness.
    
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    U17[0, 7] = 1
    V17[2, 7] = 1
    W17[4, 7] = 1
    
    U17[1, 8] = 1
    V17[5, 8] = 1
    W17[4, 8] = 1
    
    U17[3, 9] = 1
    V17[2, 9] = 1
    W17[5, 9] = 1
    
    U17[4, 10] = 1
    V17[5, 10] = 1
    W17[5, 10] = 1
    
    U17[2, 11] = 1
    V17[6, 11] = 1
    W17[0, 11] = 1
    
    U17[2, 12] = 1
    V17[7, 12] = 1
    W17[2, 12] = 1
    
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    U17[5, 14] = 1
    V17[6, 14] = 1
    W17[1, 14] = 1
    
    U17[5, 15] = 1
    V17[7, 15] = 1
    W17[3, 15] = 1
    
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1

    def to_list(mat):
        return mat.tolist()
    
    def verify_solution(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w_target = k * n + i
                    prod = U[u, :] * V[v, :]
                    contrib = W.T.dot(prod)
                    for ww in range(p * n):
                        expected = 1.0 if ww == w_target else 0.0
                        if abs(contrib[ww] - expected) > 1e-9:
                            return False
        return True

    # Search for Rank 16
    deadline = start_time + 110.0
    R = 16
    np.random.seed(42)
    
    # Precompute tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_entries.append((u, v, w))
                
    num_attempts = 0
    while time.time() < deadline and num_attempts < 50000:
        num_attempts += 1
        
        U = np.zeros((n * m, R), dtype=np.int32)
        V = np.zeros((m * p, R), dtype=np.int32)
        
        # Structured Initialization: Strassen for 2x2x2
        # Indices for 2x2 block: u in {0,1,3,4}, v in {0,1,3,4}, w in {0,1,2,3}
        # Term 0: M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1
        # C00 += 1, C11 += 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 1: M2 = (A10+A11)B00
        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        # C10 += 1, C11 -= 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # Term 2: M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[4, 2] = -1
        # C01 += 1, C11 += 1
        W[2, 2] = 1; W[3, 2] = 1
        
        # Term 3: M4 = A11(B10-B00)
        U[4, 3] = 1
        V[3, 3] = 1; V[0, 3] = -1
        # C10 += 1, C00 -= 1
        W[1, 3] = 1; W[0, 3] = -1
        
        # Term 4: M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[4, 4] = 1
        # C00 -= 1, C01 += 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # Term 5: M6 = (A10-A00)(B00+B01)
        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C11 += 1
        W[3, 5] = 1
        
        # Term 6: M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[4, 6] = -1
        V[3, 6] = 1; V[4, 6] = 1
        # C00 += 1
        W[0, 6] = 1
        
        # Check if Strassen part is correct for the 2x2 subblock
        # It should be.
        
        # Randomize the remaining 9 terms
        for r in range(7, R):
            num_nonzero_u = np.random.randint(1, 4)
            indices = np.random.choice(n * m, size=num_nonzero_u, replace=False)
            for idx in indices:
                U[idx, r] = np.random.choice([-1, 1])
                
            num_nonzero_v = np.random.randint(1, 4)
            indices = np.random.choice(m * p, size=num_nonzero_v, replace=False)
            for idx in indices:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Build A and b for the full system
        A_mat = np.zeros((108, 96), dtype=np.float64)
        b_vec = np.zeros(108, dtype=np.float64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w_target = k * n + i
                    uv_prod = U[u, :] * V[v, :]
                    for w in range(p * n):
                        A_mat[eq_idx, w*R : w*R+R] = uv_prod
                        if w == w_target:
                            b_vec[eq_idx] = 1.0
                        else:
                            b_vec[eq_idx] = 0.0
                        eq_idx += 1
        
        try:
            W_sol, residuals, rank_sol, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        W_sol = W_sol.reshape((p * n, R))
        W_int = np.round(W_sol)
        
        if not np.allclose(W_sol, W_int, atol=1e-8):
            continue
            
        if verify_solution(U, V, W_int):
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W_int)}
            
    return {"U": to_list(U17), "V": to_list(V17), "W": to_list(W17)}
