import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 4, 5
    U_dim = n * m  # 12
    V_dim = m * p  # 20
    W_dim = p * n  # 15
    
    def verify(U, V, W, rank):
        """Verify decomposition using random test matrices."""
        np.random.seed(123)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_actual = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    fa = np.dot(U[:, c], A.reshape(-1))
                    fb = np.dot(V[:, c], B.reshape(-1))
                    val += W[k * n + i, c] * fa * fb
                C_actual[i, k] = val
        return np.allclose(C_actual, C_expected, atol=1e-9)
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def strassen_block_decomposition():
        """Block decomposition using Strassen for 2x2x2 blocks."""
        columns = []
        
        def add_term(U_dict, V_dict, W_dict):
            U_vec = np.zeros(U_dim)
            V_vec = np.zeros(V_dim)
            W_vec = np.zeros(W_dim)
            for (i, j) in U_dict:
                U_vec[i * m + j] = U_dict[(i, j)]
            for (j, k) in V_dict:
                V_vec[j * p + k] = V_dict[(j, k)]
            for (i, k) in W_dict:
                W_vec[k * n + i] = W_dict[(i, k)]
            return U_vec, V_vec, W_vec
        
        def strassen_2x2x2(i_off, j_off, k_off):
            cols = []
            U1 = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
            V1 = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
            W1 = {(i_off, k_off): 1, (i_off+1, k_off+1): 1}
            cols.append(add_term(U1, V1, W1))
            U2 = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
            V2 = {(j_off, k_off): 1}
            W2 = {(i_off+1, k_off): 1}
            cols.append(add_term(U2, V2, W2))
            U3 = {(i_off, j_off): 1}
            V3 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): -1}
            W3 = {(i_off, k_off): 1, (i_off, k_off+1): -1}
            cols.append(add_term(U3, V3, W3))
            U4 = {(i_off+1, j_off+1): 1}
            V4 = {(j_off, k_off): 1, (j_off, k_off+1): -1}
            W4 = {(i_off+1, k_off): 1, (i_off+1, k_off+1): -1}
            cols.append(add_term(U4, V4, W4))
            U5 = {(i_off, j_off): 1, (i_off, j_off+1): 1}
            V5 = {(j_off+1, k_off+1): 1}
            W5 = {(i_off, k_off+1): 1}
            cols.append(add_term(U5, V5, W5))
            U6 = {(i_off+1, j_off): 1, (i_off, j_off): -1}
            V6 = {(j_off, k_off): 1, (j_off, k_off+1): 1}
            W6 = {(i_off+1, k_off): 1, (i_off+1, k_off+1): 1, (i_off, k_off): -1, (i_off, k_off+1): -1}
            cols.append(add_term(U6, V6, W6))
            U7 = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
            V7 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
            W7 = {(i_off, k_off): 1, (i_off, k_off+1): 1, (i_off+1, k_off): -1, (i_off+1, k_off+1): -1}
            cols.append(add_term(U7, V7, W7))
            return cols
        
        def schoolbook_2x2x1(i_off, j_off, k_idx):
            cols = []
            for i_local in range(2):
                for j_local in range(2):
                    U_dict = {(i_off + i_local, j_off + j_local): 1}
                    V_dict = {(j_off + j_local, k_idx): 1}
                    W_dict = {(i_off + i_local, k_idx): 1}
                    cols.append(add_term(U_dict, V_dict, W_dict))
            return cols
        
        def schoolbook_1x2xp(i_idx, j_off, k_range):
            cols = []
            for j_local in range(2):
                for k in k_range:
                    U_dict = {(i_idx, j_off + j_local): 1}
                    V_dict = {(j_off + j_local, k): 1}
                    W_dict = {(i_idx, k): 1}
                    cols.append(add_term(U_dict, V_dict, W_dict))
            return cols
        
        columns.extend(strassen_2x2x2(0, 0, 0))
        columns.extend(strassen_2x2x2(0, 0, 2))
        columns.extend(strassen_2x2x2(0, 2, 0))
        columns.extend(strassen_2x2x2(0, 2, 2))
        columns.extend(schoolbook_2x2x1(0, 0, 4))
        columns.extend(schoolbook_2x2x1(0, 2, 4))
        columns.extend(schoolbook_1x2xp(2, 0, range(5)))
        columns.extend(schoolbook_1x2xp(2, 2, range(5)))
        
        rank = len(columns)
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        for c, (U_c, V_c, W_c) in enumerate(columns):
            U[:, c] = U_c
            V[:, c] = V_c
            W[:, c] = W_c
        return U, V, W, rank
    
    def try_random_search(target_rank, max_iters=200):
        """Try to find a rank-target_rank decomposition via random init + alternating minimization."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(42 + target_rank)
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(3):
            U = np.random.randn(U_dim, target_rank) * 0.5
            V = np.random.randn(V_dim, target_rank) * 0.5
            W = np.random.randn(W_dim, target_rank) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute current error
                np.random.seed(99)
                A = np.random.randn(n, m)
                B = np.random.randn(m, p)
                C_expected = A @ B
                C_actual = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            fa = np.dot(U[:, c], A.reshape(-1))
                            fb = np.dot(V[:, c], B.reshape(-1))
                            val += W[k * n + i, c] * fa * fb
                        C_actual[i, k] = val
                error = np.linalg.norm(C_actual - C_expected)
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-8:
                    break
                
                # Alternating minimization: update W
                for c in range(target_rank):
                    # For each (i,k), the contribution is W[k*n+i, c] * (U[:,c]·A_flat) * (V[:,c]·B_flat)
                    # We want sum_c W[k*n+i,c] * U_c(A) * V_c(B) = C[i,k]
                    # This is linear in W for fixed U,V
                    # Build linear system for W
                    pass
                
                # Simpler: use gradient-free random perturbation
                if np.random.rand() < 0.3:
                    idx = np.random.randint(target_rank)
                    W[np.random.randint(W_dim), idx] += np.random.randn() * 0.1
                elif np.random.rand() < 0.6:
                    idx = np.random.randint(target_rank)
                    U[np.random.randint(U_dim), idx] += np.random.randn() * 0.1
                else:
                    idx = np.random.randint(target_rank)
                    V[np.random.randint(V_dim), idx] += np.random.randn() * 0.1
            
            if best_error < 1e-8:
                break
        
        if best_error < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Start with Strassen block decomposition (rank 56)
    U, V, W, rank = strassen_block_decomposition()
    
    if not verify(U, V, W, rank):
        U, V, W, rank = schoolbook_decomposition()
    
    best_rank = rank
    
    # Try to reduce rank using random search
    for target_rank in range(rank - 1, max(rank - 15, 40), -1):
        if time.time() > deadline - 15:
            break
        result = try_random_search(target_rank, max_iters=100)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, target_rank):
                best_rank = target_rank
                break
    
    if time.time() > deadline - 5:
        pass
    
    # Round to nearest half-integer for cleaner output
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Final verification
    if not verify(U, V, W, best_rank):
        U, V, W, best_rank = schoolbook_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
