"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Build the known rank-55 Strassen-based decomposition
    def build_rank55():
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B)
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
        # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11][col] += 1
            U[u_22][col] += 1
            V[v_11][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21][col] += 1
            U[u_22][col] += 1
            V[v_11][col] += 1
            W[w_10][col] += 1
            W[w_11][col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11][col] += 1
            V[v_12][col] += 1
            V[v_22][col] += -1
            W[w_01][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22][col] += 1
            V[v_21][col] += 1
            V[v_11][col] += -1
            W[w_00][col] += 1
            W[w_10][col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11][col] += 1
            U[u_12][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += -1
            W[w_01][col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21][col] += 1
            U[u_11][col] += -1
            V[v_11][col] += 1
            V[v_12][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12][col] += 1
            U[u_22][col] += -1
            V[v_21][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += 1
            col += 1
            
        return U, V, W
    
    U, V, W = build_rank55()
    rank = 55
    
    # Build the true tensor for verification
    dim_u = n * m  # 6
    dim_v = m * p  # 30
    dim_w = p * n  # 20
    
    T_true = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_true[alpha, beta, gamma] = 1
    
    # Verify the rank-55 decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_reconstructed = np.zeros((dim_u, dim_v, dim_w))
    for c in range(rank):
        T_reconstructed += np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c])
    
    if not np.allclose(T_reconstructed, T_true, atol=1e-10):
        # Fallback to schoolbook if something went wrong
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col_s] = 1
                    V_school[j * p + k][col_s] = 1
                    W_school[k * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a lower rank decomposition using ALS
    # Start with rank 54 and try to optimize
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try ranks from 54 down to 50 (limited by time budget)
    for target_rank in range(54, 49, -1):
        if time.time() > deadline:
            break
        
        # Initialize with the rank-55 decomposition, truncated
        U_init = U_np[:, :target_rank].copy()
        V_init = V_np[:, :target_rank].copy()
        W_init = W_np[:, :target_rank].copy()
        
        # Add small random perturbation to escape local minima
        np.random.seed(42)
        U_init += 0.01 * np.random.randn(dim_u, target_rank)
        V_init += 0.01 * np.random.randn(dim_v, target_rank)
        W_init += 0.01 * np.random.randn(dim_w, target_rank)
        
        # ALS optimization
        success = False
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Update U: fix V and W, solve for U
            # T[i, :, :] = sum_c U[i,c] * V[:,c] * W[:,c]
            # For each i: T[i, :, :] = U[i, :] @ (V * W.T)
            # Reshape: T[i] is (dim_v, dim_w), V*W.T is (dim_v, dim_w, target_rank)
            
            # Compute the "effective" matrix for U update
            # T[i, v, w] = sum_c U[i,c] * V[v,c] * W[w,c]
            # = sum_c U[i,c] * (V[v,c] * W[w,c])
            # Let M[v, w, c] = V[v,c] * W[w,c]
            # Then T[i, v, w] = sum_c U[i,c] * M[v, w, c]
            # Reshape: T[i] (dim_v*dim_w) = U[i] (target_rank) @ M_flat (target_rank, dim_v*dim_w)
            
            M_vw = V_init[:, :, None] * W_init[None, :, :]  # (dim_v, dim_w, target_rank)
            M_flat = M_vw.transpose(2, 0, 1).reshape(target_rank, dim_v * dim_w)
            
            T_flat = T_true.reshape(dim_u, dim_v * dim_w)
            
            # Solve U_flat @ M_flat.T = T_flat
            # U_flat = T_flat @ M_flat.T @ pinv(M_flat @ M_flat.T)
            try:
                M_flat_T = M_flat.T  # (dim_v*dim_w, target_rank)
                M_flat_MT = M_flat @ M_flat_T  # (target_rank, target_rank)
                U_new = np.linalg.solve(M_flat_MT, M_flat_T.T @ T_flat.T).T
                U_init = U_new
            except np.linalg.LinAlgError:
                pass
            
            # Update V: fix U and W
            M_uw = U_init[:, :, None] * W_init[None, :, :]  # (dim_u, dim_w, target_rank)
            M_flat_v = M_uw.transpose(2, 0, 1).reshape(target_rank, dim_u * dim_w)
            T_flat_v = T_true.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
            
            try:
                M_flat_v_T = M_flat_v.T
                M_flat_v_MT = M_flat_v @ M_flat_v_T
                V_new = np.linalg.solve(M_flat_v_MT, M_flat_v_T.T @ T_flat_v.T).T
                V_init = V_new
            except np.linalg.LinAlgError:
                pass
            
            # Update W: fix U and V
            M_uv = U_init[:, :, None] * V_init[None, :, :]  # (dim_u, dim_v, target_rank)
            M_flat_w = M_uv.transpose(2, 0, 1).reshape(target_rank, dim_u * dim_v)
            T_flat_w = T_true.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
            
            try:
                M_flat_w_T = M_flat_w.T
                M_flat_w_MT = M_flat_w @ M_flat_w_T
                W_new = np.linalg.solve(M_flat_w_MT, M_flat_w_T.T @ T_flat_w.T).T
                W_init = W_new
            except np.linalg.LinAlgError:
                pass
        
        # Check if the optimized decomposition is valid
        T_check = np.zeros((dim_u, dim_v, dim_w))
        for c in range(target_rank):
            T_check += np.einsum('i,j,k->ijk', U_init[:, c], V_init[:, c], W_init[:, c])
        
        if np.allclose(T_check, T_true, atol=1e-8):
            # Round to nearest integer or half-integer if close
            U_rounded = np.round(U_init * 2) / 2
            V_rounded = np.round(V_init * 2) / 2
            W_rounded = np.round(W_init * 2) / 2
            
            # Verify rounded version
            T_rounded = np.zeros((dim_u, dim_v, dim_w))
            for c in range(target_rank):
                T_rounded += np.einsum('i,j,k->ijk', U_rounded[:, c], V_rounded[:, c], W_rounded[:, c])
            
            if np.allclose(T_rounded, T_true, atol=1e-10):
                # Convert to lists
                U_list = [[float(U_rounded[i, c]) for c in range(target_rank)] for i in range(dim_u)]
                V_list = [[float(V_rounded[i, c]) for c in range(target_rank)] for i in range(dim_v)]
                W_list = [[float(W_rounded[i, c]) for c in range(target_rank)] for i in range(dim_w)]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break  # Found a valid lower rank, stop searching
        
        # Also try without rounding (in case the solution has non-integer values)
        if np.allclose(T_check, T_true, atol=1e-8):
            U_list = [[float(U_init[i, c]) for c in range(target_rank)] for i in range(dim_u)]
            V_list = [[float(V_init[i, c]) for c in range(target_rank)] for i in range(dim_v)]
            W_list = [[float(W_init[i, c]) for c in range(target_rank)] for i in range(dim_w)]
            
            if best_rank > target_rank:
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
    
    return best_result
# EVOLVE-BLOCK-END
