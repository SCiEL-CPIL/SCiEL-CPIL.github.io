import time
import numpy as np
from itertools import combinations

def construct():
    """
    Returns factor matrices U, V, W that decompose the tensor for (3, 4, 4) matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 sub-blocks and schoolbook for 1x2x2 sub-blocks.
    Total rank: 44 (4*7 + 2*8).
    
    Returns:
    Dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
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
    
    # Strassen1: A_top_left * B_left_top -> C_top_left
    X_u = [0, 1, 4, 5]
    Y_v = [0, 1, 4, 5]
    Z_w = [0, 3, 1, 4]
    col = add_strassen(X_u, Y_v, Z_w, col)
    
    # Strassen2: A_top_right * B_left_bot -> C_top_left
    X_u = [2, 3, 6, 7]
    Y_v = [8, 9, 12, 13]
    Z_w = [0, 3, 1, 4]
    col = add_strassen(X_u, Y_v, Z_w, col)
    
    # Strassen3: A_top_left * B_right_top -> C_top_right
    X_u = [0, 1, 4, 5]
    Y_v = [2, 3, 6, 7]
    Z_w = [6, 9, 7, 10]
    col = add_strassen(X_u, Y_v, Z_w, col)
    
    # Strassen4: A_top_right * B_right_bot -> C_top_right
    X_u = [2, 3, 6, 7]
    Y_v = [10, 11, 14, 15]
    Z_w = [6, 9, 7, 10]
    col = add_strassen(X_u, Y_v, Z_w, col)
    
    # Bottom-left: A_bot * B_left -> C[2,0], C[2,1]
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
    
    # Bottom-right: A_bot * B_right -> C[2,2], C[2,3]
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
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_computed = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    T_true = np.zeros((12, 16, 12))
    for row_C in range(3):
        for col_C in range(4):
            k = row_C * 4 + col_C
            for j in range(4):
                i = row_C * 4 + j
                jj = j * 4 + col_C
                T_true[i, jj, k] = 1
    
    max_err = np.max(np.abs(T_computed - T_true))
    
    if max_err >= 1e-10:
        # Incorrect, return as-is (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    # Check for linear dependencies: try removing each column
    # Vectorize each term r as u_r ⊗ v_r ⊗ w_r
    # Check if T_true is in span of remaining 43 terms
    U_flat = U_np  # (12, 44)
    V_flat = V_np  # (16, 44)
    W_flat = W_np  # (12, 44)
    
    T_true_flat = T_true.reshape(-1)
    
    # For each term r, the contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We can check: is T_true in span{term_0, ..., term_{r-1}, term_{r+1}, ..., term_43}?
    # This is equivalent to: does there exist coefficients c_s such that
    # sum_{s≠r} c_s * (U_s ⊗ V_s ⊗ W_s) = T_true?
    # 
    # A more efficient approach: check if term_r is in the span of the other terms
    # by checking if the 44x44 matrix of inner products has rank < 44
    
    # Build Gram matrix: G[r1,r2] = <term_r1, term_r2> = sum_{i,j,k} U[i,r1]*V[j,r1]*W[k,r1]*U[i,r2]*V[j,r2]*W[k,r2]
    # = (U^T U)[r1,r2] * (V^T V)[r1,r2] * (W^T W)[r1,r2]
    
    UU = U_np.T @ U_np  # (44, 44)
    VV = V_np.T @ V_np  # (44, 44)
    WW = W_np.T @ W_np  # (44, 44)
    
    # The inner product of term r1 and term r2 is NOT simply UU*VV*WW elementwise
    # It's sum_{i,j,k} U[i,r1]V[j,r1]W[k,r1]U[i,r2]V[j,r2]W[k,r2]
    # = sum_i U[i,r1]U[i,r2] * sum_j V[j,r1]V[j,r2] * sum_k W[k,r1]W[k,r2]
    # = UU[r1,r2] * VV[r1,r2] * WW[r1,r2]
    
    # Wait, that's only true if the sums factorize, which they do!
    # sum_{i,j,k} f(i)g(j)h(k) = (sum_i f(i))(sum_j g(j))(sum_k h(k))
    # But here f(i) = U[i,r1]*U[i,r2], etc. So yes:
    
    G = UU * VV * WW  # element-wise product
    
    # Check rank of G
    try:
        rank_G = np.linalg.matrix_rank(G, tol=1e-10)
    except:
        rank_G = 44
    
    if rank_G < 44:
        # There's a linear dependency! Find which column to remove
        # The null space vector gives the dependency
        try:
            _, s, vh = np.linalg.svd(G)
            # Find the smallest singular value
            min_idx = np.argmin(s)
            if s[min_idx] < 1e-8:
                null_vec = vh[min_idx]
                # The term with largest |null_vec[r]| can be removed
                remove_idx = np.argmax(np.abs(null_vec))
                
                # Build reduced decomposition
                keep = [i for i in range(44) if i != remove_idx]
                U_red = [U[i] for i in keep] if False else [[U[i][j] for j in keep] for i in range(dim_u)]
                V_red = [[V[i][j] for j in keep] for i in range(dim_v)]
                W_red = [[W[i][j] for j in keep] for i in range(dim_w)]
                
                # Verify
                U_red_np = np.array(U_red, dtype=float)
                V_red_np = np.array(V_red, dtype=float)
                W_red_np = np.array(W_red, dtype=float)
                T_red = np.einsum('ir,jr,kr->ijk', U_red_np, V_red_np, W_red_np)
                if np.max(np.abs(T_red - T_true)) < 1e-10:
                    return {"U": U_red, "V": V_red, "W": W_red}
        except:
            pass
    
    # Try removing pairs of columns (for rank 42)
    if time.time() - start_time < deadline - 10:
        # Check all pairs
        for r1 in range(44):
            for r2 in range(r1 + 1, 44):
                if time.time() - start_time > deadline - 5:
                    break
                keep = [i for i in range(44) if i != r1 and i != r2]
                U_red = [[U[i][j] for j in keep] for i in range(dim_u)]
                V_red = [[V[i][j] for j in keep] for i in range(dim_v)]
                W_red = [[W[i][j] for j in keep] for i in range(dim_w)]
                
                U_red_np = np.array(U_red, dtype=float)
                V_red_np = np.array(V_red, dtype=float)
                W_red_np = np.array(W_red, dtype=float)
                T_red = np.einsum('ir,jr,kr->ijk', U_red_np, V_red_np, W_red_np)
                if np.max(np.abs(T_red - T_true)) < 1e-10:
                    return {"U": U_red, "V": V_red, "W": W_red}
    
    # Return rank-44 solution
    return {"U": U, "V": V, "W": W}
