import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill

    def build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        
        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            # 7 Strassen terms for 2x2 blocks
            # X_u = [x0, x1, x2, x3] indices in U (row indices for A block)
            # Y_v = [y0, y1, y2, y3] indices in V (col indices for B block)
            # Z_w = [z0, z1, z2, z3] indices in W (output block positions)
            
            # Term 1: (x0+x3)(y0+y3) -> z0, z3
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            
            # Term 2: (x2+x3)y0 -> z2, z3
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            
            # Term 3: x0(y1-y3) -> z1, z3
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            
            # Term 4: x3(y2-y0) -> z0, z2
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            
            # Term 5: (x0+x1)y3 -> z0, z1
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            
            # Term 6: (x2-x0)(y0+y1) -> z3
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            
            # Term 7: (x1-x3)(y2+y3) -> z0
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col
        
        # Block structure for 3x4 * 4x4
        # A is 3x4, B is 4x4, C is 3x4
        # We split A into 2x2 blocks (rows 0-1, 2) and B into 2x2 blocks (cols 0-1, 2-3)
        # Actually, let's be more careful with the indexing.
        # U index: i*m + j where i in [0,3), j in [0,4)
        # V index: j*p + k where j in [0,4), k in [0,4)
        # W index: k*n + i where k in [0,4), i in [0,3)
        
        # Let's use the same block structure as the original program which worked.
        # Original used:
        # add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        # This corresponds to:
        # U rows: 0,1 (i=0,j=0,1) and 4,5 (i=1,j=0,1) -> 2x2 block of A
        # V cols: 0,1 (j=0,k=0,1) and 4,5 (j=1,k=0,1) -> 2x2 block of B
        # W: 0,3 (k=0,i=0,1) and 1,4 (k=1,i=0,1) -> 2x2 block of C
        
        # Let's just use the exact same construction as it was verified to work.
        col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)
        col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
        col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)
        
        # Schoolbook for the remaining 1x4 block of A (row i=2)
        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1
        
        assert col == 44, f"Expected rank 44, got {col}"
        return U, V, W
    
    U, V, W = build_rank44()
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    coeff = 0
                    for r in range(U.shape[1]):
                        coeff += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(coeff - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U_np, V_np, W_np, n, m, p):
        # Fallback to schoolbook
        rank = 48
        U = [[0]*rank for _ in range(dim_u)]
        V = [[0]*rank for _ in range(dim_v)]
        W = [[0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by checking linear dependence
    # Form the matrix M where column r is [U[:,r], V[:,r], W[:,r]]
    M = np.zeros((dim_u + dim_v + dim_w, 44))
    for r in range(44):
        M[:dim_u, r] = U_np[:, r]
        M[dim_u:dim_u+dim_v, r] = V_np[:, r]
        M[dim_u+dim_v:, r] = W_np[:, r]
    
    # Iteratively remove dependent columns with verification
    current_indices = list(range(44))
    removed = []
    
    # Use SVD to identify potential dependencies
    try:
        _, s_vals, _ = np.linalg.svd(M, full_matrices=False)
        tol = max(M.shape) * np.finfo(float).eps * (s_vals[0] if len(s_vals) > 0 else 1)
        num_rank = int(np.sum(s_vals > tol))
    except:
        num_rank = 44
    
    if num_rank < 44:
        # Try to remove (44 - num_rank) columns
        for _ in range(44 - num_rank):
            if time.time() > deadline:
                break
            removed_any = False
            for r_idx in range(len(current_indices)):
                if time.time() > deadline:
                    break
                r = current_indices[r_idx]
                others = [i for i in current_indices if i != r]
                if len(others) == 0:
                    break
                M_reduced = M[:, others]
                try:
                    x, residuals, rank_r, sv = np.linalg.lstsq(M_reduced, M[:, r], rcond=None)
                    if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-8:
                        # Column r is in the span of others, try removing it
                        test_indices = [i for i in current_indices if i != r]
                        U_test = U_np[:, test_indices]
                        V_test = V_np[:, test_indices]
                        W_test = W_np[:, test_indices]
                        
                        if verify(U_test, V_test, W_test, n, m, p):
                            current_indices = test_indices
                            removed.append(r)
                            removed_any = True
                            break
                except:
                    pass
            if not removed_any:
                break
    
    if removed:
        U_new = U_np[:, current_indices]
        V_new = V_np[:, current_indices]
        W_new = W_np[:, current_indices]
        
        if verify(U_new, V_new, W_new, n, m, p):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    return {"U": U, "V": V, "W": W}
