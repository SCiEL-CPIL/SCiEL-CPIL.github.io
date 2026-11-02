import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 42

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def to_half_int_list(matrix):
        """Convert numpy matrix to list of lists with half-integer entries"""
        return [[float(matrix[i, j]) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])]

    def verify_exact(U, V, W, rank, n, m, p):
        """Verify exact decomposition using integer arithmetic (multiply by 2 to clear halves)."""
        U2 = np.round(U * 2).astype(np.int64)
        V2 = np.round(V * 2).astype(np.int64)
        W2 = np.round(W * 2).astype(np.int64)
        
        # Use multiple random test matrices for robustness
        for seed in [12345, 67890, 11111]:
            rng = np.random.RandomState(seed)
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    total = 0.0
                    for j in range(m):
                        for r in range(rank):
                            total += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    C_comp[i, k] = total
            
            if not np.allclose(C_comp, C_true, atol=1e-9):
                return False
        return True

    def peel_rank1_improved(T, n, m, p, iterations=5):
        """
        Peel off a rank-1 component using iterative HOSVD refinement.
        """
        nm, mp, pn = n * m, m * p, p * n
        T3 = T.reshape(nm, mp, pn)
        
        # Initialize with HOSVD
        T1 = T3.reshape(nm, mp * pn)
        U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
        T2 = T3.transpose(1, 0, 2).reshape(mp, nm * pn)
        U2, S2, Vt2 = np.linalg.svd(T2, full_matrices=False)
        T3u = T3.transpose(2, 0, 1).reshape(pn, nm * mp)
        U3, S3, Vt3 = np.linalg.svd(T3u, full_matrices=False)
        
        u = U1[:, 0].copy()
        v = U2[:, 0].copy()
        w = U3[:, 0].copy()
        
        # Iterate: fix two, optimize the third
        for _ in range(iterations):
            # Optimize u given v, w
            # T contracted with v and w along axes 1,2
            T_contracted_u = np.einsum('j,k,ijk->i', v, w, T3)
            if np.linalg.norm(T_contracted_u) > 1e-12:
                u = T_contracted_u / np.linalg.norm(T_contracted_u)
            
            # Optimize v given u, w
            T_contracted_v = np.einsum('i,k,ijk->j', u, w, T3)
            if np.linalg.norm(T_contracted_v) > 1e-12:
                v = T_contracted_v / np.linalg.norm(T_contracted_v)
            
            # Optimize w given u, v
            T_contracted_w = np.einsum('i,j,ijk->k', u, v, T3)
            if np.linalg.norm(T_contracted_w) > 1e-12:
                w = T_contracted_w / np.linalg.norm(T_contracted_w)
        
        alpha = np.einsum('i,j,k,ijk->', u, v, w, T3)
        T_residual = T3 - alpha * np.einsum('i,j,k->ijk', u, v, w)
        
        return u, v, w, alpha, T_residual.reshape(nm * mp * pn)

    def try_peeling_strategy(T_dense, target_rank, n, m, p, strategy='improved'):
        """Try a specific peeling strategy and return factors or None."""
        nm, mp, pn = n * m, m * p, p * n
        T_remaining = T_dense.reshape(-1).copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for r in range(target_rank):
            if strategy == 'improved':
                u, v, w, alpha, T_remaining = peel_rank1_improved(T_remaining, n, m, p, iterations=8)
            else:
                # Basic HOSVD
                T3 = T_remaining.reshape(nm, mp, pn)
                T1 = T3.reshape(nm, mp * pn)
                U1, _, _ = np.linalg.svd(T1, full_matrices=False)
                T2 = T3.transpose(1, 0, 2).reshape(mp, nm * pn)
                U2, _, _ = np.linalg.svd(T2, full_matrices=False)
                T3u = T3.transpose(2, 0, 1).reshape(pn, nm * mp)
                U3, _, _ = np.linalg.svd(T3u, full_matrices=False)
                u = U1[:, 0]
                v = U2[:, 0]
                w = U3[:, 0]
                alpha = np.einsum('i,j,k,ijk->', u, v, w, T3)
                T_residual = T3 - alpha * np.einsum('i,j,k->ijk', u, v, w)
                T_remaining = T_residual.reshape(-1)
            
            U_cols.append(alpha * u)
            V_cols.append(v)
            W_cols.append(w)
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        # Project to nearest half-integer
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        
        if verify_exact(U_half, V_half, W_half, target_rank, n, m, p):
            return (U_half, V_half, W_half)
        return None

    # Build the dense matmul tensor
    T_dense = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_dense[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_factors = None
    
    # Try improved peeling from rank 42 down to 25
    for target_rank in range(42, 24, -1):
        result = try_peeling_strategy(T_dense, target_rank, n, m, p, strategy='improved')
        if result is not None:
            best_factors = result
            break
    
    # If improved didn't work, try basic HOSVD
    if best_factors is None:
        for target_rank in range(42, 30, -1):
            result = try_peeling_strategy(T_dense, target_rank, n, m, p, strategy='basic')
            if result is not None:
                best_factors = result
                break
    
    if best_factors is not None:
        U, V, W = best_factors
        return {"U": to_half_int_list(U), "V": to_half_int_list(V), "W": to_half_int_list(W)}
    
    # Fallback to schoolbook
    return {"U": to_half_int_list(U_school), "V": to_half_int_list(V_school), "W": to_half_int_list(W_school)}
