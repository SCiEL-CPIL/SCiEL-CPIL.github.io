import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 21
    dim_w = p * n  # 14
    
    # Construct the dense multiplication tensor T
    # T[a, b, c] = 1 if a=im+j, b=mp+k, c=pn+i for some i,j,k
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try to find a lower rank decomposition using greedy peeling
    # We try ranks from a lower bound up to schoolbook rank - 1
    schoolbook_rank = n * m * p
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    lower_bound = max(dim_u, dim_v, dim_w)
    
    # We will try a few ranks below schoolbook
    # To save time, we might not try all, but the tensor is small.
    # Let's try ranks from 35 down to lower_bound? Or up?
    # Greedy peeling accumulates error. Lower rank is harder.
    # Let's try a few specific ranks or a range.
    
    # Given the 120s limit, and small tensor, we can try many.
    # But we must ensure we return *something* valid.
    
    found = False
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank

    # Try ranks from 41 down to 30? 
    # Actually, if we find one at 41, it's not an improvement over 42.
    # We need rank < 42.
    
    # Let's try a limited number of ranks to avoid timeout if it's hard.
    # But SVD is fast.
    
    for r in range(41, 29, -1):
        # Initialize residual tensor
        T_res = T.copy()
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        for _ in range(r):
            # Matricize T_res
            # Mode 1: (dim_u, dim_v * dim_w)
            M1 = T_res.reshape(dim_u, -1)
            # Mode 2: (dim_v, dim_u * dim_w)
            M2 = T_res.transpose(1, 0, 2).reshape(dim_v, -1)
            # Mode 3: (dim_w, dim_u * dim_v)
            M3 = T_res.transpose(2, 0, 1).reshape(dim_w, -1)
            
            # SVD
            _, _, Vt1 = np.linalg.svd(M1, full_matrices=False)
            u = Vt1[0]
            
            _, _, Vt2 = np.linalg.svd(M2, full_matrices=False)
            v = Vt2[0]
            
            _, _, Vt3 = np.linalg.svd(M3, full_matrices=False)
            w = Vt3[0]
            
            # Normalize vectors
            u = u / np.linalg.norm(u)
            v = v / np.linalg.norm(v)
            w = w / np.linalg.norm(w)
            
            # Calculate scalar lambda
            # lambda = <T_res, u v w> / (||u||^2 ||v||^2 ||w||^2) = <T_res, u v w> since norms are 1
            # <T_res, u v w> = sum(T_res * u[:, None, None] * v[None, :, None] * w[None, None, :])
            # Efficiently:
            temp = np.tensordot(T_res, u, axes=([0], [0]))
            temp = np.tensordot(temp, v, axes=([0], [0]))
            temp = np.tensordot(temp, w, axes=([0], [0]))
            lam = temp
            
            # Subtract
            T_res = T_res - lam * np.einsum('i,j,k->ijk', u, v, w)
            
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        # Project to half-integers
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        # Verify exactness
        # Reconstruct tensor
        T_recon = np.einsum('ia,ja,ka->ijk', U_half, V_half, W_half)
        
        # Check if T_recon equals T
        # T has 1s at specific indices, 0 elsewhere.
        # T_recon should be exactly 1 or 0.
        # Due to floating point, check closeness to integers?
        # The task requires integer/half-integer entries.
        # If the entries are half-integers, the sum will be integer or half-integer?
        # 0.5 * 0.5 * 0.5 = 0.125.
        # Wait. The entries of U, V, W are half-integers.
        # The product $U_{a,\alpha} V_{b,\alpha} W_{c,\alpha}$ can be 1/8.
        # The sum over $\alpha$ must be 1 or 0.
        # So we check if $T_{recon}$ is close to $T$.
        
        if np.allclose(T_recon, T, atol=1e-9):
            # It's a valid decomposition!
            # Convert to list of lists for return
            found = True
            best_U = U_half
            best_V = V_half
            best_W = W_half
            best_rank = r
            break
            
    if not found:
        # Fall back to schoolbook
        rank = schoolbook_rank
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        best_U = U
        best_V = V
        best_W = W
        best_rank = rank

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
