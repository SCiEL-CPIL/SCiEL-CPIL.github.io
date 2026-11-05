import time
import numpy as np

def construct():
    """
    Returns factor matrices U, V, W that decompose the tensor for (3, 4, 4) matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 sub-blocks and schoolbook for 1x4x4 sub-blocks.
    Base rank: 44. Attempts rank reduction via null-space analysis.
    
    Returns:
    Dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    def build_base_rank44():
        """Build the rank-44 decomposition using 4 Strassen blocks + 16 schoolbook terms."""
        rank = 44
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        
        col = 0
        
        def add_strassen(X_u, Y_v, Z_w, col):
            """Add 7 Strassen terms for X(2x2) * Y(2x2) = Z(2x2)."""
            # M1 = (X00+X11)(Y00+Y11)
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            # M2 = (X10+X11)Y00
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            # M3 = X00(Y01-Y11)
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            # M4 = X11(Y10-Y00)
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            # M5 = (X00+X01)Y11
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            # M6 = (X10-X00)(Y00+Y01)
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            # M7 = (X01-X11)(Y10+Y11)
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col
        
        # Strassen1: A_top_left(2x2) * B_left_top(2x2) -> C_top_left(2x2)
        X_u = [0, 1, 4, 5]
        Y_v = [0, 1, 4, 5]
        Z_w = [0, 3, 1, 4]
        col = add_strassen(X_u, Y_v, Z_w, col)
        
        # Strassen2: A_top_right(2x2) * B_left_bot(2x2) -> C_top_left(2x2)
        X_u = [2, 3, 6, 7]
        Y_v = [8, 9, 12, 13]
        Z_w = [0, 3, 1, 4]
        col = add_strassen(X_u, Y_v, Z_w, col)
        
        # Strassen3: A_top_left(2x2) * B_right_top(2x2) -> C_top_right(2x2)
        X_u = [0, 1, 4, 5]
        Y_v = [2, 3, 6, 7]
        Z_w = [6, 9, 7, 10]
        col = add_strassen(X_u, Y_v, Z_w, col)
        
        # Strassen4: A_top_right(2x2) * B_right_bot(2x2) -> C_top_right(2x2)
        X_u = [2, 3, 6, 7]
        Y_v = [10, 11, 14, 15]
        Z_w = [6, 9, 7, 10]
        col = add_strassen(X_u, Y_v, Z_w, col)
        
        # Bottom-left: A_bot(1x4) * B_left(4x2) -> C[2,0], C[2,1]
        for j in range(4):
            u_idx = 8 + j
            v_idx_0 = j * 4 + 0
            v_idx_1 = j * 4 + 1
            w_idx_0 = 2
            w_idx_1 = 5
            U[u_idx][col] += 1; V[v_idx_0][col] += 1; W[w_idx_0][col] += 1
            col += 1
            U[u_idx][col] += 1; V[v_idx_1][col] += 1; W[w_idx_1][col] += 1
            col += 1
        
        # Bottom-right: A_bot(1x4) * B_right(4x2) -> C[2,2], C[2,3]
        for j in range(4):
            u_idx = 8 + j
            v_idx_0 = j * 4 + 2
            v_idx_1 = j * 4 + 3
            w_idx_0 = 8
            w_idx_1 = 11
            U[u_idx][col] += 1; V[v_idx_0][col] += 1; W[w_idx_0][col] += 1
            col += 1
            U[u_idx][col] += 1; V[v_idx_1][col] += 1; W[w_idx_1][col] += 1
            col += 1
        
        return U, V, W
    
    def verify(U_np, V_np, W_np):
        """Verify the decomposition is correct."""
        T_computed = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        T_true = np.zeros((dim_u, dim_v, dim_w))
        for row_C in range(n):
            for col_C in range(p):
                k = col_C * n + row_C
                for j in range(m):
                    i = row_C * m + j
                    jj = j * p + col_C
                    T_true[i, jj, k] = 1
        return np.max(np.abs(T_computed - T_true)) < 1e-10
    
    def try_reduce_rank(U, V, W, max_iterations=20):
        """
        Attempt to reduce rank by finding and removing linearly dependent terms.
        Uses the Gram matrix approach: G[r,s] = sum_i U[i,r]*U[i,s] * sum_j V[j,r]*V[j,s] * sum_k W[k,r]*W[k,s]
        If rank(G) < current_rank, there's a null direction we can exploit.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        current_rank = U_np.shape[1]
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Compute the Gram matrix
            UU = U_np.T @ U_np
            VV = V_np.T @ V_np
            WW = W_np.T @ W_np
            G = UU * VV * WW
            
            # Check rank of G
            try:
                rank_G = np.linalg.matrix_rank(G, tol=1e-10)
            except:
                break
            
            if rank_G >= current_rank:
                break
            
            # Find null space vectors
            try:
                _, s, vh = np.linalg.svd(G)
            except:
                break
            
            # Find indices where singular values are near zero
            null_indices = np.where(s < 1e-8)[0]
            
            if len(null_indices) == 0:
                break
            
            # For each null vector, try removing the term with largest coefficient
            removed_any = False
            for ni in null_indices:
                if time.time() > deadline - 5:
                    break
                
                null_vec = vh[ni]
                # Find the term to remove (largest absolute coefficient)
                remove_idx = np.argmax(np.abs(null_vec))
                
                keep = [i for i in range(current_rank) if i != remove_idx]
                U_red = [[U[i][j] for j in keep] for i in range(dim_u)]
                V_red = [[V[i][j] for j in keep] for i in range(dim_v)]
                W_red = [[W[i][j] for j in keep] for i in range(dim_w)]
                
                U_red_np = np.array(U_red, dtype=float)
                V_red_np = np.array(V_red, dtype=float)
                W_red_np = np.array(W_red, dtype=float)
                
                if verify(U_red_np, V_red_np, W_red_np):
                    U, V, W = U_red, V_red, W_red
                    U_np, V_np, W_np = U_red_np, V_red_np, W_red_np
                    current_rank -= 1
                    removed_any = True
                    break
            
            if not removed_any:
                break
        
        return U, V, W, current_rank
    
    # Build base decomposition
    U, V, W = build_base_rank44()
    
    # Verify base decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if not verify(U_np, V_np, W_np):
        # Fallback: schoolbook rank 48
        rank = 48
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Attempt rank reduction
    U, V, W, final_rank = try_reduce_rank(U, V, W)
    
    return {"U": U, "V": V, "W": W}
