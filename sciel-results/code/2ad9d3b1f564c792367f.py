import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Build the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if A[a] * B[b] contributes to C[c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # Condition: j from a == j from b, k from b == k from c, i from a == i from c
    
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    
    T = np.zeros((dim_a, dim_b, dim_c), dtype=float)
    
    for i in range(n):
        for j in range(m):
            a_idx = i * m + j
            for k in range(p):
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # We want to find rank r such that T = sum_{l=1}^r u_l v_l w_l^T (outer products)
    # This is equivalent to flattening T into a matrix and finding low rank approximations?
    # No, it's a tensor rank decomposition.
    # We can use ALS.
    
    def als_decompose(T, rank, max_iters=50, tol=1e-8):
        dim_a, dim_b, dim_c = T.shape
        np.random.seed(42) # Reproducibility
        
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
        
        for it in range(max_iters):
            # Solve for W
            # T[a, b, c] = sum_l U[a, l] V[b, l] W[c, l]
            # For fixed U, V:
            # T[a, b, :] = sum_l (U[a, l] V[b, l]) W[:, l]
            # Let M[a, b, l] = U[a, l] V[b, l]
            # We want to find W such that sum_l M[a, b, l] W[:, l] approx T[a, b, :]
            # This is a linear system:
            # Reshape T to (dim_a * dim_b, dim_c)
            # Reshape M to (dim_a * dim_b, rank)
            # Then T_mat = M_mat @ W.T  =>  T_mat.T = W @ M_mat.T
            # W = T_mat.T @ M_mat.T @ (M_mat @ M_mat.T)^-1
            # Using lstsq: W.T = lstsq(M_mat, T_mat)
            
            M_mat = (U[:, :, None] * V[None, :, :]).reshape(-1, rank)
            T_mat = T.reshape(-1, dim_c)
            
            W_new, _, _, _ = np.linalg.lstsq(M_mat, T_mat, rcond=None)
            W = W_new.T
            
            # Solve for V
            # T[a, b, c] = sum_l U[a, l] V[b, l] W[c, l]
            # T[:, b, c] = sum_l (U[a, l] W[c, l]) V[b, l]
            # M_mat2 = (W[None, :, :] * U[:, None, :]).reshape(dim_c, dim_a * rank)? No.
            # T[a, b, c] = V[b, l] * (U[a, l] W[c, l])
            # For fixed U, W:
            # Let N[a, c, l] = U[a, l] W[c, l]
            # T[a, b, c] = sum_l N[a, c, l] V[b, l]
            # Reshape T to (dim_b, dim_a * dim_c)
            # Reshape N to (dim_a * dim_c, rank)
            # T_mat2 = T.transpose(1, 0, 2).reshape(dim_b, -1)
            # N_mat2 = N.reshape(-1, rank)
            # T_mat2 = N_mat2.T @ V.T  =>  T_mat2.T = V @ N_mat2
            # V = lstsq(N_mat2.T, T_mat2.T).T ?
            # Actually: T_mat2 = V.T @ N_mat2
            # V.T = lstsq(N_mat2, T_mat2)
            
            N = (W[None, :, :] * U[:, None, :]) # shape (dim_a, dim_c, rank)
            N_mat2 = N.reshape(dim_a * dim_c, rank)
            T_mat2 = T.transpose(1, 0, 2).reshape(dim_b, dim_a * dim_c)
            
            V_new, _, _, _ = np.linalg.lstsq(N_mat2, T_mat2, rcond=None)
            V = V_new.T
            
            # Solve for U
            # T[a, b, c] = sum_l U[a, l] V[b, l] W[c, l]
            # For fixed V, W:
            # Let P[b, c, l] = V[b, l] W[c, l]
            # T[a, b, c] = sum_l P[b, c, l] U[a, l]
            # Reshape T to (dim_a, dim_b * dim_c)
            # Reshape P to (dim_b * dim_c, rank)
            # T_mat3 = T.transpose(0, 1, 2).reshape(dim_a, -1)
            # P_mat3 = P.reshape(-1, rank)
            # T_mat3 = P_mat3.T @ U.T
            # U.T = lstsq(P_mat3, T_mat3)
            
            P = (W[None, :, :] * V[:, None, :]) # shape (dim_b, dim_c, rank)
            P_mat3 = P.reshape(dim_b * dim_c, rank)
            T_mat3 = T.reshape(dim_a, dim_b * dim_c)
            
            U_new, _, _, _ = np.linalg.lstsq(P_mat3, T_mat3, rcond=None)
            U = U_new.T
            
            # Check convergence
            # Reconstruct T
            T_rec = np.einsum('al,bl,cl->abc', U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < tol:
                break
                
        return U, V, W, err

    def try_snap_and_validate(U, V, W, T):
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check if snapped version is valid
        T_rec = np.einsum('al,bl,cl->abc', U_snap, V_snap, W_snap)
        err = np.linalg.norm(T_rec - T)
        
        if err < 1e-5:
            # Convert to lists of floats (or ints if they are integers)
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            # Clean up floats that are effectively integers
            def clean_list(lst):
                out = []
                for row in lst:
                    new_row = []
                    for val in row:
                        if abs(val - round(val)) < 1e-9:
                            new_row.append(int(round(val)))
                        else:
                            new_row.append(val)
                    out.append(new_row)
                return out
            
            return clean_list(U_list), clean_list(V_list), clean_list(W_list)
        
        return None

    # Try ranks from 96 down to 80
    # We know 112 works. Strassen suggests 97 is possible.
    # Let's try to beat 97.
    
    start_time = time.time()
    max_time = 100 # Leave 20s buffer
    
    best_solution = None
    
    # Try a few ranks
    for r in range(96, 79, -1):
        if time.time() - start_time > max_time:
            break
            
        try:
            U, V, W, err = als_decompose(T, r, max_iters=20, tol=1e-6)
            if err < 1e-4:
                res = try_snap_and_validate(U, V, W, T)
                if res:
                    U_list, V_list, W_list = res
                    best_solution = (r, U_list, V_list, W_list)
                    break # Found a valid lower rank
        except Exception as e:
            pass
            
    if best_solution:
        r, U, V, W = best_solution
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook rank 112
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
