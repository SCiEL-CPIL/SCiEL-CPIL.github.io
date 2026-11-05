import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Helper to verify decomposition
    def verify(U, V, W, T):
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(computed, T, atol=1e-6)

    # Helper to round to half-integers
    def round_half(U, V, W):
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        return U_r, V_r, W_r

    # Function to search for low-rank decomposition of n x m x p
    def find_decomposition(n, m, p, target_rank, max_time=5.0):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        best_U, best_V, best_W = None, None, None
        best_rank = n*m*p
        
        # Try target rank and slightly higher
        for rank in [target_rank, target_rank + 1, target_rank + 2]:
            if time.time() > start_time + max_time:
                break
            
            # Random initialization
            for attempt in range(50):
                if time.time() > start_time + max_time:
                    break
                U = np.random.randn(n*m, rank)
                V = np.random.randn(m*p, rank)
                W = np.random.randn(p*n, rank)
                
                # ALS
                for it in range(100):
                    # Update U
                    for i in range(n*m):
                        A = np.column_stack([np.outer(V[:,r], W[:,r]) for r in range(rank)])
                        b = T[i,:,:].flatten()
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i,:] = sol
                    # Update V
                    for j in range(m*p):
                        A = np.column_stack([np.outer(U[:,r], W[:,r]) for r in range(rank)])
                        b = T[:,j,:].flatten()
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j,:] = sol
                    # Update W
                    for k in range(p*n):
                        A = np.column_stack([np.outer(U[:,r], V[:,r]) for r in range(rank)])
                        b = T[:,:,k].flatten()
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k,:] = sol
                        
                    # Check convergence
                    if it > 10:
                        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
                        if np.allclose(computed, T, atol=1e-5):
                            # Try rounding
                            U_r, V_r, W_r = round_half(U, V, W)
                            if verify(U_r, V_r, W_r, T):
                                # Check if rank is actually rank (remove zero cols)
                                # ... simplify ...
                                best_U, best_V, best_W = U_r, V_r, W_r
                                best_rank = rank
                                return best_U, best_V, best_W, best_rank
                            break
        
        # Fallback to schoolbook
        return schoolbook(n, m, p)

    # 1. Find 3x3x3 decomposition (Target Rank 23)
    U_333, V_333, W_333, rank_333 = find_decomposition(3, 3, 3, 23, max_time=10.0)
    
    # 2. Find 3x3x2 decomposition (Target Rank 14)
    U_332, V_332, W_332, rank_332 = find_decomposition(3, 3, 2, 14, max_time=10.0)
    
    # 3. Assemble 3x3x8
    # U_total: (9, R1+R1+R2)
    # V_total: (24, R1+R1+R2)
    # W_total: (24, R1+R1+R2)
    
    R1 = rank_333
    R2 = rank_332
    TotalRank = R1 + R1 + R2
    
    U_total = np.zeros((9, TotalRank))
    V_total = np.zeros((24, TotalRank))
    W_total = np.zeros((24, TotalRank))
    
    # Block 1: k=0,1,2
    U_total[:, 0:R1] = U_333
    # V rows: j*8 + k. k=0,1,2.
    # V_333 has rows j*3 + k_local.
    # We need to map k_local 0,1,2 to global k 0,1,2.
    for j in range(3):
        for k_local in range(3):
            global_k = k_local
            V_total[j*8 + global_k, 0:R1] = V_333[j*3 + k_local, 0:R1]
    # W rows: k*3 + i. k=0,1,2.
    # W_333 has rows k_local*3 + i.
    # Map k_local 0,1,2 to global k 0,1,2.
    for k_local in range(3):
        global_k = k_local
        for i in range(3):
            W_total[global_k*3 + i, 0:R1] = W_333[k_local*3 + i, 0:R1]
            
    # Block 2: k=3,4,5
    U_total[:, R1:2*R1] = U_333
    for j in range(3):
        for k_local in range(3):
            global_k = 3 + k_local
            V_total[j*8 + global_k, R1:2*R1] = V_333[j*3 + k_local, 0:R1]
    for k_local in range(3):
        global_k = 3 + k_local
        for i in range(3):
            W_total[global_k*3 + i, R1:2*R1] = W_333[k_local*3 + i, 0:R1]
            
    # Block 3: k=6,7
    U_total[:, 2*R1:2*R1+R2] = U_332
    for j in range(3):
        for k_local in range(2):
            global_k = 6 + k_local
            V_total[j*8 + global_k, 2*R1:2*R1+R2] = V_332[j*3 + k_local, 0:R2]
    for k_local in range(2):
        global_k = 6 + k_local
        for i in range(3):
            W_total[global_k*3 + i, 2*R1:2*R1+R2] = W_332[k_local*3 + i, 0:R2]
            
    # Verify final result
    T_total = np.zeros((9, 24, 24))
    for i in range(3):
        for j in range(3):
            for k in range(8):
                T_total[i*3+j, j*8+k, k*3+i] = 1.0
                
    if verify(U_total, V_total, W_total, T_total):
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U_total]
        V_list = [[float(x) for x in row] for row in V_total]
        W_list = [[float(x) for x in row] for row in W_total]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to Schoolbook 3x3x8
        U, V, W = schoolbook(3, 3, 8)
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
