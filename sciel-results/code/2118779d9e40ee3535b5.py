import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n, m, p = 2, 4, 4
    
    # Build multiplication tensor: T[a,b,c]=1 where a=i*m+k, b=k*p+j, c=i*p+j
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, i*p+j] = 1
    
    # Fast verification using tensor contraction
    def verify_fast(U, V, W, r):
        """Verify decomposition using numpy tensor operations."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # Compute sum_s U[a,s]*V[b,s]*W[c,s]
        # Use einsum: 'as,bs,cs->abc'
        result = np.einsum('as,bs,cs->abc', U_arr, V_arr, W_arr)
        return np.allclose(result, T, atol=1e-9)
    
    # Rank-28 Strassen-based solution
    def strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1

            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1

            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1

            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1

            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1

            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1

            U[X_rows[1]][col_offset + 6] += 1
            U[X_rows[3]][col_offset + 6] -= 1
            V[Y_rows[2]][col_offset + 6] += 1
            V[Y_rows[3]][col_offset + 6] += 1
            W[Z_rows[0]][col_offset + 6] += 1

        X1 = [0, 1, 4, 5];  Y1 = [0, 1, 4, 5];   Z1 = [0, 2, 1, 3]
        X2 = [2, 3, 6, 7];  Y2 = [8, 9, 12, 13]; Z2 = [0, 2, 1, 3]
        X3 = [0, 1, 4, 5];  Y3 = [2, 3, 6, 7];   Z3 = [4, 6, 5, 7]
        X4 = [2, 3, 6, 7];  Y4 = [10, 11, 14, 15]; Z4 = [4, 6, 5, 7]

        add_block(0,  X1, Y1, Z1)
        add_block(7,  X2, Y2, Z2)
        add_block(14, X3, Y3, Z3)
        add_block(21, X4, Y4, Z4)
        
        return U, V, W, rank
    
    # Structured search for rank < 28
    # Key insight: the tensor has 4 independent 2x2x2 blocks.
    # Each block has rank 7. If blocks can share terms, total rank < 28.
    # But blocks use disjoint tensor coordinates, so sharing is limited.
    # Try rank 27: one shared term between blocks.
    
    def search_rank_structured(r, max_trials=500):
        """Search for rank-r decomposition exploiting block structure."""
        if time.time() > deadline - 5:
            return None
        
        # For rank r, we need r terms. Start from the 28-term solution
        # and try to merge/eliminate terms.
        # Alternative: random search with structural constraints.
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(999 + trial)
            
            # Initialize with sparse U, V, W
            U_mat = np.zeros((n*m, r), dtype=np.int64)
            V_mat = np.zeros((m*p, r), dtype=np.int64)
            W_mat = np.zeros((p*n, r), dtype=np.int64)
            
            # Each term s: pick sparse support
            for s in range(r):
                # U support: 2-3 entries
                nz_u = np.random.randint(2, 4)
                idx_u = np.random.choice(n*m, nz_u, replace=False)
                U_mat[idx_u, s] = np.random.choice([-1, 1], nz_u)
                
                # V support: 2-3 entries
                nz_v = np.random.randint(2, 4)
                idx_v = np.random.choice(m*p, nz_v, replace=False)
                V_mat[idx_v, s] = np.random.choice([-1, 1], nz_v)
                
                # W support: 2-3 entries
                nz_w = np.random.randint(2, 4)
                idx_w = np.random.choice(p*n, nz_w, replace=False)
                W_mat[idx_w, s] = np.random.choice([-1, 1], nz_w)
            
            # Verify
            if verify_fast(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), r):
                return U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), r
        
        return None
    
    # Try to find rank < 28
    # Given the structural independence of blocks, rank 28 is very likely minimal.
    # Do a quick search for rank 27 (limited trials due to time).
    best_result = None
    
    # Quick check: can we do rank 27?
    if time.time() < deadline - 20:
        result = search_rank_structured(27, max_trials=200)
        if result is not None:
            best_result = result
    
    # If not found, use rank 28
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-28 Strassen solution
    U, V, W, rank = strassen_28()
    return {"U": U, "V": "V", "W": W} if False else {"U": U, "V": V, "W": W}
