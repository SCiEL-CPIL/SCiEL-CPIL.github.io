import time
import numpy as np

def construct():
    """
    Returns factor matrices U, V, W that decompose the tensor for (3, 4, 4) matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 sub-blocks and schoolbook for 1x4x4 sub-blocks.
    Total rank: 44 (4*7 + 2*8).
    
    Returns:
    Dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    start_time = time.time()
    deadline = 110  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
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
    
    # W index convention: W[k*n+i, r] where k=col_C (0..3), i=row_C (0..2)
    # So W index = col_C * 3 + row_C
    
    # Strassen1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    X_u = [0, 1, 4, 5]
    Y_v = [0, 1, 4, 5]
    Z_w = [0, 3, 1, 4]
    col = add_strassen(X_u, Y_v, Z_w, col)
    
    # Strassen2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    X_u = [2, 3, 6, 7]
    Y_v = [8, 9, 12, 13]
    Z_w = [0, 3, 1, 4]
    col = add_strassen(X_u, Y_v, Z_w, col)
    
    # Strassen3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    X_u = [0, 1, 4, 5]
    Y_v = [2, 3, 6, 7]
    Z_w = [6, 9, 7, 10]
    col = add_strassen(X_u, Y_v, Z_w, col)
    
    # Strassen4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    X_u = [2, 3, 6, 7]
    Y_v = [10, 11, 14, 15]
    Z_w = [6, 9, 7, 10]
    col = add_strassen(X_u, Y_v, Z_w, col)
    
    # Bottom row: A[2,0:4] * B[0:4,0:2] -> C[2,0:2]
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
    
    # Bottom row: A[2,0:4] * B[0:4,2:4] -> C[2,2:4]
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
    
    # Verify correctness with proper W indexing
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_computed = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    # True tensor with correct W indexing: k = col_C * 3 + row_C
    T_true = np.zeros((12, 16, 12))
    for row_C in range(3):
        for col_C in range(4):
            k = col_C * 3 + row_C
            for j in range(4):
                i = row_C * 4 + j
                jj = j * 4 + col_C
                T_true[i, jj, k] = 1
    
    max_err = np.max(np.abs(T_computed - T_true))
    
    if max_err < 1e-10:
        # Try to reduce rank by checking for linear dependencies
        term_vectors = []
        for r in range(rank):
            tv = np.kron(np.kron(U_np[:, r], V_np[:, r]), W_np[:, r])
            term_vectors.append(tv)
        
        term_matrix = np.array(term_vectors).T
        
        try:
            s = np.linalg.svd(term_matrix, compute_uv=False)
            tol = 1e-10
            numerical_rank = np.sum(s > tol)
        except:
            numerical_rank = rank
        
        if numerical_rank < rank:
            current_terms = list(range(rank))
            removed = []
            
            for r in range(rank):
                if time.time() - start_time > deadline - 5:
                    break
                if r not in current_terms:
                    continue
                test_terms = [t for t in current_terms if t != r]
                if len(test_terms) == 0:
                    continue
                
                T_test = np.zeros((12, 16, 12))
                for t in test_terms:
                    T_test += np.einsum('i,j,k->ijk', U_np[:, t], V_np[:, t], W_np[:, t])
                
                if np.max(np.abs(T_test - T_true)) < 1e-10:
                    current_terms = test_terms
                    removed.append(r)
            
            if removed:
                new_rank = len(current_terms)
                U_new = np.zeros((dim_u, new_rank))
                V_new = np.zeros((dim_v, new_rank))
                W_new = np.zeros((dim_w, new_rank))
                for new_idx, old_idx in enumerate(current_terms):
                    U_new[:, new_idx] = U_np[:, old_idx]
                    V_new[:, new_idx] = V_np[:, old_idx]
                    W_new[:, new_idx] = W_np[:, old_idx]
                
                T_final = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                if np.max(np.abs(T_final - T_true)) < 1e-10:
                    return {
                        "U": U_new.tolist(),
                        "V": V_new.tolist(),
                        "W": W_new.tolist()
                    }
        
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U, "V": V, "W": W}
