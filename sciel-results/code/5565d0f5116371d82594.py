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
        """Verify exact decomposition using integer arithmetic."""
        # Multiply by 2 to work in integers
        U2 = np.round(U * 2).astype(np.int64)
        V2 = np.round(V * 2).astype(np.int64)
        W2 = np.round(W * 2).astype(np.int64)
        
        # Check: sum_r U2[i*m+j, r] * V2[j*p+k, r] * W2[k*n+i, r] should be 8 * T_dense
        # T_dense has 1s at valid indices, 0 elsewhere.
        
        T_comp = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_comp += np.einsum('i,j,k->ijk', U2[:, r], V2[:, r], W2[:, r])
            
        T_dense = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_dense[i * m + j, j * p + k, k * n + i] = 1.0
                    
        if not np.allclose(T_comp, 8 * T_dense, atol=1e-6):
            return False
        return True

    def find_decomposition(target_rank, n, m, p):
        """
        Attempt to find a rank-target_rank decomposition using iterative SVD peeling
        with immediate projection to half-integers.
        """
        nm, mp, pn = n * m, m * p, p * n
        
        # Initialize residual tensor
        T_res = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_res[i * m + j, j * p + k, k * n + i] = 1.0

        U_cols = []
        V_cols = []
        W_cols = []
        
        for r in range(target_rank):
            # Unfold for SVD
            T1 = T_res.reshape(nm, mp * pn)
            U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
            
            T2 = T_res.transpose(1, 0, 2).reshape(mp, nm * pn)
            U2, S2, Vt2 = np.linalg.svd(T2, full_matrices=False)
            
            T3 = T_res.transpose(2, 0, 1).reshape(pn, nm * mp)
            U3, S3, Vt3 = np.linalg.svd(T3, full_matrices=False)
            
            u = U1[:, 0]
            v = U2[:, 0]
            w = U3[:, 0]
            
            alpha = np.einsum('i,j,k,ijk->', u, v, w, T_res)
            
            # Scale to get candidate factors
            U_c = alpha * u
            V_c = v
            W_c = w
            
            # Project to half-integers
            U_h = np.round(U_c * 2) / 2.0
            V_h = np.round(V_c * 2) / 2.0
            W_h = np.round(W_c * 2) / 2.0
            
            if np.all(U_h == 0) or np.all(V_h == 0) or np.all(W_h == 0):
                return None

            # Compute contribution of this rank-1 term
            T_contrib = np.einsum('i,j,k->ijk', U_h, V_h, W_h)
            
            # Update residual
            T_res = T_res - T_contrib
            
            # Store factors
            U_cols.append(U_h)
            V_cols.append(V_h)
            W_cols.append(W_h)
            
            if np.linalg.norm(T_res) < 1e-9:
                break
        
        if len(U_cols) == 0:
            return None
            
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        # Verify exactness
        if verify_exact(U, V, W, len(U_cols), n, m, p):
            return U, V, W
            
        return None

    # Try to find a decomposition with rank < 42
    for rank in range(39, 42):
        result = find_decomposition(rank, n, m, p)
        if result is not None:
            U, V, W = result
            return {"U": to_half_int_list(U), "V": to_half_int_list(V), "W": to_half_int_list(W)}
            
    # Fallback to schoolbook
    return {"U": to_half_int_list(U_school), "V": to_half_int_list(V_school), "W": to_half_int_list(W_school)}
