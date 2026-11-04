import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T with indices (x, y, z) where x=i*m+k, y=k*p+j, z=j*n+i
    T = np.zeros((8, 16, 8), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1

    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    # Known rank-28 solution
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
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

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    best_U = [row[:] for row in U_known]
    best_V = [row[:] for row in V_known]
    best_W = [row[:] for row in W_known]
    best_rank = 28

    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T_est += np.einsum('x,y,z->xyz', u, v, w)
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    # Verify the known solution
    if not check_solution(best_U, best_V, best_W, best_rank):
        # Fallback: construct trivial rank-32 solution
        best_rank = 32
        best_U = [[0] * 32 for _ in range(8)]
        best_V = [[0] * 32 for _ in range(16)]
        best_W = [[0] * 32 for _ in range(8)]
        idx = 0
        for i in range(2):
            for k in range(4):
                for j in range(4):
                    x = i * 4 + k
                    y = k * 4 + j
                    z = j * 2 + i
                    best_U[x][idx] = 1
                    best_V[y][idx] = 1
                    best_W[z][idx] = 1
                    idx += 1
        if not check_solution(best_U, best_V, best_W, best_rank):
            # This should never happen, but return something valid
            return {"U": best_U, "V": best_V, "W": best_W}

    def rationalize_vector(vec, max_denom=16):
        """Rationalize a vector to have small integer/half-integer entries."""
        rat = []
        for val in vec:
            best_frac = Fraction(0)
            best_err = abs(val)
            for d in range(1, max_denom + 1):
                num = round(val * d)
                frac = Fraction(num, d)
                err = abs(val - float(frac))
                if err < best_err:
                    best_err = err
                    best_frac = frac
            rat.append(best_frac)
        return rat

    def rationalize_decomposition(U_cols, V_cols, W_cols):
        """Rationalize a list of columns."""
        U_rat = [rationalize_vector(u) for u in U_cols]
        V_rat = [rationalize_vector(v) for v in V_cols]
        W_rat = [rationalize_vector(w) for w in W_cols]
        return U_rat, V_rat, W_rat

    def build_matrices(U_rat, V_rat, W_rat):
        """Build matrix lists from rationalized vectors."""
        rank = len(U_rat)
        U_list = [[0] * rank for _ in range(8)]
        V_list = [[0] * rank for _ in range(16)]
        W_list = [[0] * rank for _ in range(8)]
        
        for r in range(rank):
            for i in range(8):
                U_list[i][r] = float(U_rat[r][i])
            for i in range(16):
                V_list[i][r] = float(V_rat[r][i])
            for i in range(8):
                W_list[i][r] = float(W_rat[r][i])
        return U_list, V_list, W_list

    # Strategy: Try to find rank < 28 using structured search
    # The tensor has symmetries: permuting rows/cols of the 4x4 blocks
    # We exploit this by trying structured perturbations of the known solution
    
    # Approach 1: Try to merge pairs of rank-1 terms
    # If two terms have overlapping support, we might combine them
    
    # Approach 2: Use the fact that the tensor is block-structured
    # The 2x4x4 multiplication can be seen as 4 independent 2x1x4 multiplications
    # each with rank 2, giving total rank 8? No, that's not right.
    
    # Let's try a more systematic search for rank 24-27
    
    found_better = False
    
    for target_rank in [27, 26, 25, 24]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            random.seed(target_rank * 1000 + trial)
            np.random.seed(target_rank * 1000 + trial)
            
            # Start from the known solution and try to reduce rank
            # by finding linear dependencies
            
            # Build the known solution as a 3D tensor
            U_np = np.array(best_U, dtype=float)
            V_np = np.array(best_V, dtype=float)
            W_np = np.array(best_W, dtype=float)
            
            # Try to find a rank-reduced decomposition
            # Use SVD-like approach on the tensor
            
            residual = T.astype(float).copy()
            U_cols = []
            V_cols = []
            W_cols = []
            
            for step in range(target_rank):
                if time.time() > deadline - 5:
                    break
                if np.linalg.norm(residual) < 1e-12:
                    break
                
                # Random initialization
                u = np.random.randn(8)
                v = np.random.randn(16)
                w = np.random.randn(8)
                u /= np.linalg.norm(u)
                v /= np.linalg.norm(v)
                w /= np.linalg.norm(w)
                
                # Alternating least squares
                for it in range(30):
                    u_new = np.einsum('xyz,y,z->x', residual, v, w)
                    v_new = np.einsum('xyz,x,z->y', residual, u, w)
                    w_new = np.einsum('xyz,x,y->z', residual, u, v)
                    
                    nu = np.linalg.norm(u_new)
                    nv = np.linalg.norm(v_new)
                    nw = np.linalg.norm(w_new)
                    
                    if nu < 1e-15 or nv < 1e-15 or nw < 1e-15:
                        break
                    
                    u_new /= nu
                    v_new /= nv
                    w_new /= nw
                    
                    if (np.linalg.norm(u_new - u) < 1e-10 and
                        np.linalg.norm(v_new - v) < 1e-10 and
                        np.linalg.norm(w_new - w) < 1e-10):
                        u, v, w = u_new, v_new, w_new
                        break
                    u, v, w = u_new, v_new, w_new
                
                c = np.einsum('xyz,x,y,z->', residual, u, v, w)
                
                if abs(c) < 1e-12:
                    break
                
                residual -= c * np.einsum('x,y,z->xyz', u, v, w)
                
                U_cols.append(c * u)
                V_cols.append(v)
                W_cols.append(w)
                
                if np.linalg.norm(residual) < 1e-10:
                    break
            
            rank = len(U_cols)
            if rank >= best_rank or rank > target_rank:
                continue
            
            # Rationalize
            U_rat, V_rat, W_rat = rationalize_decomposition(U_cols, V_cols, W_cols)
            U_list, V_list, W_list = build_matrices(U_rat, V_rat, W_rat)
            
            if check_solution(U_list, V_list, W_list, rank):
                best_rank = rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                found_better = True
                break
        
        if found_better:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
