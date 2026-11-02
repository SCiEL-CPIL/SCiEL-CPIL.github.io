"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    deadline = time.time() + 135  # safety margin before 150s kill

    # Build the block-Strassen rank-36 decomposition
    rank_init = 36
    
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
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

    def verify_decomposition(U_arr, V_arr, W_arr):
        """Check if the decomposition is valid by testing on random inputs."""
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(W_arr.shape[1]):
                C_computed += U_arr @ V_arr[:, r] * W_arr[:, r].reshape(p, n).T
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    # First, verify the initial decomposition
    if not verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank 40
        U_sb = np.zeros((n * m, n * m * p))
        V_sb = np.zeros((m * p, n * m * p))
        W_sb = np.zeros((p * n, n * m * p))
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[u_idx(i, j)][col_sb] = 1
                    V_sb[v_idx(j, k)][col_sb] = 1
                    W_sb[w_idx(k, i)][col_sb] = 1
                    col_sb += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Try to remove redundant columns using tensor-based linear algebra
    def _check_column_redundant(U_arr, V_arr, W_arr, col_idx, other_cols):
        """Check if column col_idx is in the span of other columns."""
        # Build the tensor for column col_idx
        # T_c[i,j,k] = U[i,c] * V[j,c] * W[k,c]
        target = np.zeros(n * m * m * p * p * n)
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    target[(i * (m*p) + j) * (p*n) + k] = U_arr[i, col_idx] * V_arr[j, col_idx] * W_arr[k, col_idx]
        
        if len(other_cols) == 0:
            return False, None
        
        # Build basis matrix from other columns
        basis = np.zeros((n * m * m * p * p * n, len(other_cols)))
        for idx, oc in enumerate(other_cols):
            for i in range(n * m):
                for j in range(m * p):
                    for k in range(p * n):
                        basis[(i * (m*p) + j) * (p*n) + k, idx] = U_arr[i, oc] * V_arr[j, oc] * W_arr[k, oc]
        
        # Check if target is in span of basis using least squares
        try:
            coeffs, residuals, rank_basis, sv = np.linalg.lstsq(basis, target, rcond=None)
            reconstruction = basis @ coeffs
            if np.allclose(reconstruction, target, atol=1e-10):
                return True, coeffs
        except:
            pass
        return False, None

    def try_remove_columns(U_arr, V_arr, W_arr, max_removals=5):
        """Try to remove linearly dependent columns."""
        current_U = U_arr.copy()
        current_V = V_arr.copy()
        current_W = W_arr.copy()
        current_rank = current_U.shape[1]
        
        removed = 0
        for attempt in range(max_removals):
            if time.time() > deadline:
                break
            if current_rank <= 1:
                break
            
            found_removal = False
            for col_idx in range(current_rank):
                if time.time() > deadline:
                    break
                
                other_cols = [c for c in range(current_rank) if c != col_idx]
                redundant, _ = _check_column_redundant(current_U, current_V, current_W, col_idx, other_cols)
                
                if redundant:
                    new_U = np.delete(current_U, col_idx, axis=1)
                    new_V = np.delete(current_V, col_idx, axis=1)
                    new_W = np.delete(current_W, col_idx, axis=1)
                    
                    if verify_decomposition(new_U, new_V, new_W):
                        current_U = new_U
                        current_V = new_V
                        current_W = new_W
                        current_rank -= 1
                        removed += 1
                        found_removal = True
                        break
            
            if not found_removal:
                break
        
        return current_U, current_V, current_W, current_rank

    # Try to remove columns
    best_U, best_V, best_W, best_rank = try_remove_columns(U, V, W, max_removals=5)

    # Try pair removals: check if two columns sum to a third or similar combinations
    def try_pair_operations(U_arr, V_arr, W_arr, max_ops=10):
        """Try to find pair operations that reduce rank."""
        current_U = U_arr.copy()
        current_V = V_arr.copy()
        current_W = W_arr.copy()
        current_rank = current_U.shape[1]
        
        ops_performed = 0
        for attempt in range(max_ops):
            if time.time() > deadline:
                break
            if current_rank <= 2:
                break
            
            found_op = False
            for i in range(current_rank):
                if time.time() > deadline:
                    break
                for j in range(i + 1, current_rank):
                    if time.time() > deadline:
                        break
                    
                    # Check if column i + column j equals column k for some k
                    # Or column i - column j, etc.
                    for op in [1, -1]:
                        sum_tensor = None
                        for k in range(current_rank):
                            if k == i or k == j:
                                continue
                            
                            # Check if U[:,i] + op*U[:,j] is proportional to U[:,k]
                            # and V[:,i] + op*V[:,j] is proportional to V[:,k]
                            # and W[:,i] + op*W[:,j] is proportional to W[:,k]
                            
                            # This is complex, so use a simpler check:
                            # Check if the tensor of col i + op * tensor of col j 
                            # equals the tensor of col k
                            
                            # Build target tensor
                            target = np.zeros(n * m * m * p * p * n)
                            for idx in range(n * m):
                                for jdx in range(m * p):
                                    for kdx in range(p * n):
                                        target[(idx * (m*p) + jdx) * (p*n) + kdx] = \
                                            (current_U[idx, i] + op * current_U[idx, j]) * \
                                            (current_V[jdx, i] + op * current_V[jdx, j]) * \
                                            (current_W[kdx, i] + op * current_W[kdx, j])
                            
                            # Build tensor for column k
                            target_k = np.zeros(n * m * m * p * p * n)
                            for idx in range(n * m):
                                for jdx in range(m * p):
                                    for kdx in range(p * n):
                                        target_k[(idx * (m*p) + jdx) * (p*n) + kdx] = \
                                            current_U[idx, k] * current_V[jdx, k] * current_W[kdx, k]
                            
                            if np.allclose(target, target_k, atol=1e-10):
                                # Column k is redundant: it equals col i + op * col j
                                # Remove column k
                                other_cols = [c for c in range(current_rank) if c != k]
                                new_U = np.delete(current_U, k, axis=1)
                                new_V = np.delete(current_V, k, axis=1)
                                new_W = np.delete(current_W, k, axis=1)
                                
                                if verify_decomposition(new_U, new_V, new_W):
                                    current_U = new_U
                                    current_V = new_V
                                    current_W = new_W
                                    current_rank -= 1
                                    ops_performed += 1
                                    found_op = True
                                    break
                    
                    if found_op:
                        break
                
                if found_op:
                    break
            
            if not found_op:
                break
        
        return current_U, current_V, current_W, current_rank

    # Try pair operations
    best_U, best_V, best_W, best_rank = try_pair_operations(best_U, best_V, best_W, max_ops=5)

    # Convert to lists with clean values
    def clean_val(v):
        if abs(v - round(v)) < 1e-10:
            return int(round(v))
        elif abs(v*2 - round(v*2)) < 1e-10:
            return round(v*2) / 2.0
        else:
            return v

    best_U_list = [[clean_val(best_U[i, j]) for j in range(best_U.shape[1])] for i in range(best_U.shape[0])]
    best_V_list = [[clean_val(best_V[i, j]) for j in range(best_V.shape[1])] for i in range(best_V.shape[0])]
    best_W_list = [[clean_val(best_W[i, j]) for j in range(best_W.shape[1])] for i in range(best_W.shape[0])]
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
