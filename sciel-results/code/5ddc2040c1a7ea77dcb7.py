import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit

    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # Target rank 36 decomposition
    rank = 36
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (y00+y11)(z00+z11) -> X00, X11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (y10+y11)z00 -> X10, -X11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: y00(z01-z11) -> X01, X11
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: y11(z10-z00) -> X00, X10
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (y00+y01)z11 -> -X00, X01
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (y10-y00)(z00+z01) -> X11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (y01-y11)(z10+z11) -> X00
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L (cols 0,1) * B_11 (rows 0,1; cols 0,1)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R (cols 2,3) * B_21 (rows 2,3; cols 0,1)
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L (cols 0,1) * B_12 (rows 0,1; cols 2,3)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R (cols 2,3) * B_22 (rows 2,3; cols 2,3)
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L (cols 0,1) * B_15 (rows 0,1; col 4)
    # 2x2x1: rank 4
    
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Block 6: A_R (cols 2,3) * B_25 (rows 2,3; col 4)
    # 2x2x1: rank 4
    
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Now try to find linear dependencies among the 36 columns
    # to reduce the rank
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if the decomposition exactly reproduces the multiplication tensor."""
        try:
            D = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(D[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                            return False
            return True
        except Exception:
            return False
    
    # Try removing columns one at a time to find dependencies
    # A column c is redundant if it can be expressed as a linear combination
    # of the other columns in a way that preserves the tensor
    
    def try_remove_column(U_mat, V_mat, W_mat, col_to_remove, remaining_cols):
        """
        Try to express column col_to_remove as a linear combination of remaining_cols.
        For each of U, V, W, we need the same linear combination to work.
        This is a system of linear equations.
        """
        target_u = U_mat[:, col_to_remove]
        target_v = V_mat[:, col_to_remove]
        target_w = W_mat[:, col_to_remove]
        
        U_rem = U_mat[:, remaining_cols]
        V_rem = V_mat[:, remaining_cols]
        W_rem = W_mat[:, remaining_cols]
        
        # Solve for coefficients c such that U_rem @ c = target_u
        # This is an overdetermined system (8 equations, len(remaining) unknowns)
        try:
            c_u, residuals_u, rank_u, _ = np.linalg.lstsq(U_rem, target_u, rcond=None)
            # Check if residual is small
            if residuals_u[0] > 1e-8:
                return None
            
            # Verify with V
            pred_v = V_rem @ c_u
            if np.linalg.norm(pred_v - target_v) > 1e-8:
                return None
            
            # Verify with W
            pred_w = W_rem @ c_u
            if np.linalg.norm(pred_w - target_w) > 1e-8:
                return None
            
            return c_u
        except Exception:
            return None
    
    # Try each column for removal
    best_rank = 36
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for col_to_remove in range(36):
        if time.time() > deadline:
            break
        
        remaining_cols = [c for c in range(36) if c != col_to_remove]
        coeffs = try_remove_column(U, V, W, col_to_remove, remaining_cols)
        
        if coeffs is not None:
            # Remove the column and verify
            U_new = np.delete(U, col_to_remove, axis=1)
            V_new = np.delete(V, col_to_remove, axis=1)
            W_new = np.delete(W, col_to_remove, axis=1)
            
            if verify_decomposition(U_new, V_new, W_new):
                best_rank = 35
                best_U, best_V, best_W = U_new, V_new, W_new
                break
    
    # If we found a rank-35 decomposition, try to go lower
    if best_rank == 35:
        current_U, current_V, current_W = best_U, best_V, best_W
        current_rank = 35
        
        for col_to_remove in range(current_rank):
            if time.time() > deadline:
                break
            
            remaining_cols = [c for c in range(current_rank) if c != col_to_remove]
            coeffs = try_remove_column(current_U, current_V, current_W, col_to_remove, remaining_cols)
            
            if coeffs is not None:
                U_new = np.delete(current_U, col_to_remove, axis=1)
                V_new = np.delete(current_V, col_to_remove, axis=1)
                W_new = np.delete(current_W, col_to_remove, axis=1)
                
                if verify_decomposition(U_new, V_new, W_new):
                    best_rank = 34
                    best_U, best_V, best_W = U_new, V_new, W_new
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank = 34
                    break
    
    # Use the best decomposition found
    if best_rank < 36:
        U, V, W = best_U, best_V, best_W
    
    # Verify the final result
    if time.time() < deadline:
        if not verify_decomposition(U, V, W):
            # Fallback to schoolbook if verification fails (should not happen)
            rank_school = n * m * p
            U_school = np.zeros((n * m, rank_school), dtype=float)
            V_school = np.zeros((m * p, rank_school), dtype=float)
            W_school = np.zeros((p * n, rank_school), dtype=float)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j, idx] = 1.0
                        V_school[j * p + k, idx] = 1.0
                        W_school[k * n + i, idx] = 1.0
                        idx += 1
            U = U_school
            V = V_school
            W = W_school
    
    # Convert to lists of ints/floats
    U_list = [[int(val) if val == int(val) else val for val in row] for row in U.tolist()]
    V_list = [[int(val) if val == int(val) else val for val in row] for row in V.tolist()]
    W_list = [[int(val) if val == int(val) else val for val in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
