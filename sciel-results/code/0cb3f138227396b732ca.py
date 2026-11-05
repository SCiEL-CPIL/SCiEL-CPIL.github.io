import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # The target tensor T is sparse.
    # T[ (i*m+j), (j*p+k), (k*n+i) ] = 1
    # We can represent T as a flattened vector or a 3D array.
    # For ALS, it's easier to work with the linear system.
    # Sum_c U[:,c] V[:,c]^T W[:,c]^T ... no.
    # The equation is:
    # T = sum_c u_c (v_c^T \otimes w_c^T) ? No.
    # T is a 3-tensor.
    # Let's flatten T into a vector of length dim_u * dim_v * dim_w.
    # This is huge: 6 * 12 * 8 = 576.
    # This is small. We can store the full tensor.
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Try ranks from low to high
    # Known lower bound is around 19. Schoolbook is 24.
    for rank in range(19, 25):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random restarts for this rank
        num_restarts = 50 if rank < 22 else 20
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # Normalize to avoid scaling issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # ALS iterations
            max_iter = 100
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                # T = sum_c u_c v_c^T w_c^T (flattened)
                # Fix V, W.
                # T is linear in U.
                # Reshape T to (dim_u, dim_v * dim_w)
                T_reshaped = T.reshape(dim_u, dim_v * dim_w)
                
                # Construct the RHS for U
                # For each c, the contribution to T is u_c \otimes (v_c \otimes w_c)
                # So T = U * (V \otimes W)^T ?
                # Let M_vw be a matrix of size (dim_v * dim_w, rank)
                # M_vw[:, c] = flatten(v_c \otimes w_c)
                # Then T_flat = U @ M_vw.T
                # U = T_flat @ pinv(M_vw.T)
                
                M_vw = np.zeros((dim_v * dim_w, rank))
                for c in range(rank):
                    M_vw[:, c] = np.kron(V[:, c], W[:, c])
                
                U_new = T_reshaped @ np.linalg.pinv(M_vw.T)
                
                # Solve for V
                T_reshaped_v = T.reshape(dim_v, dim_u * dim_w)
                M_uw = np.zeros((dim_u * dim_w, rank))
                for c in range(rank):
                    M_uw[:, c] = np.kron(U_new[:, c], W[:, c])
                V_new = T_reshaped_v @ np.linalg.pinv(M_uw.T)
                
                # Solve for W
                T_reshaped_w = T.reshape(dim_w, dim_u * dim_v)
                M_uv = np.zeros((dim_u * dim_v, rank))
                for c in range(rank):
                    M_uv[:, c] = np.kron(U_new[:, c], V_new[:, c])
                W_new = T_reshaped_w @ np.linalg.pinv(M_uv.T)
                
                # Check convergence
                diff = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                if diff < 1e-8:
                    break
            
            # Check if the decomposition is exact
            # Reconstruct T
            T_recon = np.zeros((dim_u, dim_v, dim_w))
            for c in range(rank):
                T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            
            residual = np.linalg.norm(T_recon - T)
            
            if residual < 1e-6:
                # Try to snap to half-integers
                # Scale by 2 to make integers
                U_int = np.round(2 * U) / 2
                V_int = np.round(2 * V) / 2
                W_int = np.round(2 * W) / 2
                
                # Check snapped version
                T_recon_snapped = np.zeros((dim_u, dim_v, dim_w))
                for c in range(rank):
                    T_recon_snapped += np.einsum('i,j,k->ijk', U_int[:, c], V_int[:, c], W_int[:, c])
                
                residual_snapped = np.linalg.norm(T_recon_snapped - T)
                
                if residual_snapped < 1e-6:
                    # Found a valid half-integer decomposition
                    best_rank = rank
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    found = True
                    break
        
        if found:
            break

    if best_U is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    # Convert to lists of lists as required
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
