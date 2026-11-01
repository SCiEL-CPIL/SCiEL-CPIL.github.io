import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n, m, p = 2, 2, 15
    
    # Dimensions for the tensor T
    # T is a 3-way tensor of size (n*m) x (m*p) x (p*n)
    # Indices:
    # U index: i*m + j  (i in 0..n-1, j in 0..m-1) -> size 4
    # V index: j*p + k  (j in 0..m-1, k in 0..p-1) -> size 30
    # W index: k*n + i  (k in 0..p-1, i in 0..n-1) -> size 30
    # Note: The problem statement says W[k*n+i, r].
    # Standard matrix mult tensor T_{i,j,k} = 1 if i,j,k are valid indices for C[i,k] = sum_j A[i,j]B[j,k].
    # In flattened form:
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Initialize the tensor T as zeros
    T = np.zeros((dim_u, dim_v, dim_w))
    
    # Populate T with 1s where the multiplication is defined
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So for each valid (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Greedy Rank-1 Extraction
    U_list = []
    V_list = []
    W_list = []
    
    T_curr = T.copy()
    max_rank = dim_u * dim_v * dim_w # Upper bound (schoolbook is n*m*p = 60, but let's be safe)
    
    # We will attempt to extract rank-1 components until the tensor is close to zero
    # or we hit a time/rank limit.
    
    for rank_iter in range(max_rank):
        if time.time() > deadline:
            break
            
        # Flatten T_curr to find the maximum absolute value
        # This is a simple heuristic for finding the dominant rank-1 component
        # A better approach would be to use SVD on matricizations, but 
        # for this specific sparse-ish structure, checking the max entry
        # and then optimizing the other two vectors might be faster and robust.
        
        # Let's try a more robust approach: 
        # 1. Find the index of the max absolute value in T_curr.
        # 2. Set that value to 1 (or its sign) in the rank-1 component.
        # 3. Solve for the other two vectors by minimizing the residual
        #    || T_curr - u \otimes v \otimes w ||_F
        #    This is non-convex. 
        #    
        # Alternative: Use the "Power Method" on the tensor? 
        # Or, since we know the structure is a sum of Kronecker products,
        # we can try to match the known Strassen structure if we detect it,
        # but the directive says "different strategy".
        #
        # Let's stick to a simpler iterative method:
        # 1. Find max entry (u_idx, v_idx, w_idx).
        # 2. Initialize u, v, w as unit vectors at these indices.
        # 3. Iterate:
        #    a. Fix v, w. Update u to best match T_curr[:, v_idx, w_idx] (scaled).
        #    b. Fix u, w. Update v to best match T_curr[u_idx, :, w_idx].
        #    c. Fix u, v. Update w to best match T_curr[u_idx, v_idx, :].
        # 4. Subtract u \otimes v \otimes w from T_curr.
        
        max_flat_idx = np.argmax(np.abs(T_curr))
        u_idx, v_idx, w_idx = np.unravel_index(max_flat_idx, T_curr.shape)
        
        if abs(T_curr[u_idx, v_idx, w_idx]) < 1e-9:
            break
            
        # Initialize u, v, w
        u = np.zeros(dim_u)
        v = np.zeros(dim_v)
        w = np.zeros(dim_w)
        
        # Set initial values to 1 at the max index
        u[u_idx] = 1.0
        v[v_idx] = 1.0
        w[w_idx] = 1.0
        
        # Refine using alternating least squares (ALS) for rank-1
        # We want to minimize || T_curr - u \otimes v \otimes w ||_F
        # This is equivalent to maximizing the alignment.
        
        for _ in range(10): # Few iterations to converge
            if time.time() > deadline:
                break
                
            # Update u: 
            # We want u to be proportional to the "slice" of T_curr along u-axis 
            # weighted by v and w? No, that's for SVD.
            # For rank-1 update in ALS:
            # u_new = (T_curr \cdot v \cdot w) / (||v||^2 ||w||^2) ? No.
            # The gradient of || T - u \otimes v \otimes w ||^2 w.r.t u is:
            # -2 * (T \cdot v \cdot w) + 2 * (u \cdot v \cdot w) \cdot (v \cdot w)
            # Setting to 0: u \propto (T \cdot v \cdot w)
            
            # Compute T_contracted_u = sum_{v,w} T_curr[:, v, w] * v[v] * w[w]
            # This is the matrix product of T_curr (reshaped) with v and w.
            
            # Reshape T_curr to (dim_u, dim_v * dim_w)
            T_mat_u = T_curr.reshape(dim_u, -1)
            vw_vec = np.outer(v, w).flatten()
            u_new = T_mat_u @ vw_vec
            
            # Normalize u_new to prevent scaling issues, we will rescale later
            norm_u = np.linalg.norm(u_new)
            if norm_u > 1e-12:
                u_new /= norm_u
            else:
                break
            
            # Similarly for v
            T_mat_v = T_curr.reshape(dim_v, dim_u * dim_w)
            uw_vec = np.outer(u_new, w).flatten()
            v_new = T_mat_v @ uw_vec
            norm_v = np.linalg.norm(v_new)
            if norm_v > 1e-12:
                v_new /= norm_v
            else:
                break
            
            # Similarly for w
            T_mat_w = T_curr.reshape(dim_w, dim_u * dim_v)
            uv_vec = np.outer(u_new, v_new).flatten()
            w_new = T_mat_w @ uv_vec
            norm_w = np.linalg.norm(w_new)
            if norm_w > 1e-12:
                w_new /= norm_w
            else:
                break
            
            u, v, w = u_new, v_new, w_new
            
        # After convergence, determine the optimal scaling factor alpha
        # We want to minimize || T_curr - alpha * u \otimes v \otimes w ||
        # Optimal alpha = <T_curr, u \otimes v \otimes w> / || u \otimes v \otimes w ||^2
        # Since u, v, w are normalized, || u \otimes v \otimes w ||^2 = 1.
        # So alpha = <T_curr, u \otimes v \otimes w>
        
        # Compute inner product
        # <T, u \otimes v \otimes w> = sum_{i,j,k} T[i,j,k] u[i] v[j] w[k]
        # = (T_mat_u @ vw_vec) @ u  ... wait, u is already updated.
        # Let's just compute it directly using the current u, v, w
        alpha = np.einsum('ijk,i,j,k->', T_curr, u, v, w)
        
        if abs(alpha) < 1e-9:
            break
            
        # Subtract the rank-1 component
        T_curr -= alpha * np.einsum('i,j,k->ijk', u, v, w)
        
        # Store the components
        # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r
        # Here, the r-th term is alpha * u \otimes v \otimes w.
        # We can absorb alpha into one of the vectors, say U.
        U_list.append(alpha * u)
        V_list.append(v)
        W_list.append(w)
        
        # Check if T_curr is close to zero
        if np.linalg.norm(T_curr) < 1e-9:
            break
            
    if not U_list:
        # Fallback to schoolbook if something went wrong
        U_list = []
        V_list = []
        W_list = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(dim_u); u[i*m+j] = 1.0
                    v = np.zeros(dim_v); v[j*p+k] = 1.0
                    w = np.zeros(dim_w); w[k*n+i] = 1.0
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)

    rank = len(U_list)
    
    # Assemble matrices
    U = np.column_stack(U_list) if U_list else np.zeros((dim_u, 0))
    V = np.column_stack(V_list) if V_list else np.zeros((dim_v, 0))
    W = np.column_stack(W_list) if W_list else np.zeros((dim_w, 0))
    
    # Verify with a random test
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    C_computed_flat = np.zeros(dim_w)
    for r in range(rank):
        u_dot = np.dot(U[:, r], A_flat)
        v_dot = np.dot(V[:, r], B_flat)
        C_computed_flat += W[:, r] * u_dot * v_dot
        
    C_computed = C_computed_flat.reshape(p, n).T # Wait, W index is k*n+i. 
    # C[i,k] is stored at index k*n+i.
    # So C_computed_flat is indexed by k*n+i.
    # We need to reshape it to (p, n) such that C_flat[k*n+i] corresponds to C[i,k].
    # If we reshape (p, n) in row-major, index k*n+i is row k, col i.
    # So C_mat[k, i] = C[i, k].
    # Thus C_computed = C_mat.T
    
    C_computed = C_computed_flat.reshape(p, n).T
    
    if not np.allclose(C_computed, C_expected, atol=1e-5):
        # Fallback to schoolbook
        U_list = []
        V_list = []
        W_list = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(dim_u); u[i*m+j] = 1.0
                    v = np.zeros(dim_v); v[j*p+k] = 1.0
                    w = np.zeros(dim_w); w[k*n+i] = 1.0
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
        rank = len(U_list)
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
