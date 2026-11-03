def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target flattened matrix M
    M = [[Fraction(0) for _ in range(dim_w)] for _ in range(dim_u * dim_v)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                M[u_idx * dim_v + v_idx][w_idx] = Fraction(1)
                
    # Try rank 17 decomposition
    try:
        U, V, W = _build_optimal_decomp()
        # Convert to numpy for fast verification
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # A[row, r] = U[u, r] * V[v, r] where row = u * dim_v + v
        # We can reshape U to (dim_u, rank) and V to (dim_v, rank)
        # Then A is the outer product of U and V per column?
        # A is (dim_u * dim_v) x rank
        # A[row, r] = U[u, r] * V[v, r]
        # This is not a simple matmul.
        # But we can check M == A @ W.T
        # Let's construct A
        # A = (U[:, :, None] * V[None, :, :]).reshape(dim_u * dim_v, -1) ?
        # U is (dim_u, rank). V is (dim_v, rank).
        # We want A such that A[u*dim_v+v, r] = U[u, r] * V[v, r].
        
        # Efficient way:
        # For each r, column r of A is U[:, r] * V[:, r] (element-wise product, then flatten? No, U[:,r] is vector dim_u, V[:,r] is vector dim_v. Their outer product is dim_u x dim_v. Flatten it).
        # So A[:, r] = (U[:, r] * V[:, r, None]).flatten()? No.
        # A[:, r] corresponds to the tensor slice T[:, :, r] flattened.
        # T[:, :, r] = U[:, r] * V[:, r, None] (outer product).
        # So A[:, r] = (U[:, r] * V[:, r, None]).flatten().
        
        A_np = np.zeros((dim_u * dim_v, U_np.shape[1]), dtype=float)
        for r in range(U_np.shape[1]):
            # Outer product of U[:, r] and V[:, r]
            slice_r = np.outer(U_np[:, r], V_np[:, r])
            A_np[:, r] = slice_r.flatten()
            
        M_recon = A_np @ W_np.T
        M_target = np.array(M, dtype=float)
        
        if np.allclose(M_recon, M_target, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook rank 18
            U_sch = np.zeros((dim_u, dim_u * dim_v // dim_u), dtype=int) # This logic is wrong, just build it properly
            # Schoolbook has rank n*m*p = 18
            rank_sch = n * m * p
            U_sch = np.zeros((dim_u, rank_sch), dtype=int)
            V_sch = np.zeros((dim_v, rank_sch), dtype=int)
            W_sch = np.zeros((dim_w, rank_sch), dtype=int)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U_sch[u_idx, idx] = 1
                        V_sch[v_idx, idx] = 1
                        W_sch[w_idx, idx] = 1
                        idx += 1
                        
            return {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
            
    except Exception:
        # Fallback to schoolbook
        rank_sch = n * m * p
        U_sch = np.zeros((dim_u, rank_sch), dtype=int)
        V_sch = np.zeros((dim_v, rank_sch), dtype=int)
        W_sch = np.zeros((dim_w, rank_sch), dtype=int)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_sch[u_idx, idx] = 1
                    V_sch[v_idx, idx] = 1
                    W_sch[w_idx, idx] = 1
                    idx += 1
                    
        return {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
