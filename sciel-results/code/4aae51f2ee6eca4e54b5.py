import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Dimensions of the tensor modes
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the matrix multiplication tensor T
    # T is a 3-tensor of shape (dim_u, dim_v, dim_w)
    # Non-zero entries are at (i*m+j, j*p+k, k*n+i) = 1
    # We will work with the unfolded matrix or use a sparse representation if possible, 
    # but for GTD, we need to contract with vectors.
    
    # For GTD, we need to compute the tensor contraction with two vectors.
    # Let's store the indices of the non-zero entries.
    # Since n, m, p are small, the total number of non-zeros is 3*4*6 = 72.
    # The total size of the tensor is (12) * (24) * (18) = 5184. This is small enough to store densely.
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Greedy Tensor Decomposition
    # We will iteratively find the best rank-1 approximation to the residual T.
    # The best rank-1 approximation of a tensor T is given by the top singular vectors of its unfoldings.
    # Specifically, we can use the following algorithm:
    # 1. Start with residual R = T
    # 2. Find the best rank-1 approximation (u, v, w, lambda) to R.
    #    This can be done by maximizing || R * u * v * w ||_F.
    #    A common heuristic is to use the top singular vectors of the mode-n unfoldings.
    #    Or, more simply, since the tensor is sparse and small, we can use a local search or SVD-based approach.
    #    
    # A robust way for small tensors:
    # The best rank-1 component is proportional to the top singular vectors of the unfoldings.
    # However, for exact decomposition, we might need to be careful with rounding.
    
    # Let's use a simpler approach: 
    # Since the tensor is a matrix multiplication tensor, it has a known structure.
    # But let's try GTD first.
    
    R = T.copy()
    components = []
    
    max_rank = 100  # Cap to avoid infinite loops
    min_lambda = 1e-9
    
    for rank in range(max_rank):
        if time.time() > deadline:
            break
            
        # Find best rank-1 approximation to R
        # We can use the top singular vectors of the 3 unfoldings.
        # Unfold 1: shape (dim_u, dim_v * dim_w)
        # Unfold 2: shape (dim_v, dim_u * dim_w)
        # Unfold 3: shape (dim_w, dim_u * dim_v)
        
        # A simpler heuristic for the best rank-1 component:
        # Maximize the norm of the contraction.
        # We can use the top left singular vectors of the unfoldings.
        
        U_mat = R.reshape(dim_u, dim_v * dim_w)
        V_mat = R.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
        W_mat = R.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
        
        # Get top singular vectors
        try:
            _, _, Vt_u = np.linalg.svd(U_mat, full_matrices=False)
            u_vec = Vt_u[0]
            
            _, _, Vt_v = np.linalg.svd(V_mat, full_matrices=False)
            v_vec = Vt_v[0]
            
            _, _, Vt_w = np.linalg.svd(W_mat, full_matrices=False)
            w_vec = Vt_w[0]
        except:
            break
        
        # Normalize vectors
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = v_vec / np.linalg.norm(v_vec)
        w_vec = w_vec / np.linalg.norm(w_vec)
        
        # Compute the scalar lambda
        # lambda = <R, u \otimes v \otimes w>
        # lambda = sum(R[i,j,k] * u[i] * v[j] * w[k])
        # We can compute this by contracting R with u and v to get a vector, then dot with w.
        temp = np.tensordot(R, u_vec, axes=([0], [0]))  # shape (dim_v, dim_w)
        temp = np.tensordot(temp, v_vec, axes=([0], [0]))  # shape (dim_w)
        lambda_val = np.dot(temp, w_vec)
        
        if abs(lambda_val) < min_lambda:
            break
            
        # Add component
        components.append((lambda_val, u_vec, v_vec, w_vec))
        
        # Update residual
        # R = R - lambda * u \otimes v \otimes w
        # We can do this by broadcasting
        update = lambda_val * np.outer(np.outer(u_vec, v_vec), w_vec)
        R = R - update
        
        # Check if residual is close to zero
        if np.linalg.norm(R) < 1e-9:
            break
            
    if not components:
        # Fallback to schoolbook if GTD fails or finds nothing
        return _schoolbook()
        
    # Construct U, V, W matrices
    R_final = len(components)
    U = np.zeros((dim_u, R_final))
    V = np.zeros((dim_v, R_final))
    W = np.zeros((dim_w, R_final))
    
    for r, (lam, u, v, w) in enumerate(components):
        U[:, r] = lam * u  # Or just u, and scale in V/W? 
        # The decomposition is T = sum r (u_r \otimes v_r \otimes w_r)
        # Usually we can absorb lambda into one of the vectors.
        # Let's put lambda in U.
        V[:, r] = v
        W[:, r] = w
        
    # Verify exactness
    # Reconstruct T_check = sum u_r \otimes v_r \otimes w_r
    T_check = np.zeros((dim_u, dim_v, dim_w))
    for r in range(R_final):
        T_check += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        
    if np.allclose(T_check, T, atol=1e-9):
        # Round to integers or half-integers if close
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        T_check_rounded = np.zeros((dim_u, dim_v, dim_w))
        for r in range(R_final):
            T_check_rounded += np.outer(np.outer(U_rounded[:, r], V_rounded[:, r]), W_rounded[:, r])
            
        if np.allclose(T_check_rounded, T, atol=1e-9):
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
            
    # If GTD result is not exact or rounding failed, fallback to schoolbook
    return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
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
