import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 2, 4, 4
    
    # Target rank 28
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_u, offset + 0] += 1
        U[a11_u, offset + 0] += 1
        V[b00_v, offset + 0] += 1
        V[b11_v, offset + 0] += 1
        W[c00_w, offset + 0] += 1
        W[c11_w, offset + 0] += 1
        
        # M2 = (a10+a11)b00
        U[a10_u, offset + 1] += 1
        U[a11_u, offset + 1] += 1
        V[b00_v, offset + 1] += 1
        W[c10_w, offset + 1] += 1
        W[c11_w, offset + 1] -= 1
        
        # M3 = a00(b01-b11)
        U[a00_u, offset + 2] += 1
        V[b01_v, offset + 2] += 1
        V[b11_v, offset + 2] -= 1
        W[c01_w, offset + 2] += 1
        W[c11_w, offset + 2] += 1
        
        # M4 = a11(b10-b00)
        U[a11_u, offset + 3] += 1
        V[b10_v, offset + 3] += 1
        V[b00_v, offset + 3] -= 1
        W[c00_w, offset + 3] += 1
        W[c10_w, offset + 3] += 1
        
        # M5 = (a00+a01)b11
        U[a00_u, offset + 4] += 1
        U[a01_u, offset + 4] += 1
        V[b11_v, offset + 4] += 1
        W[c00_w, offset + 4] -= 1
        W[c01_w, offset + 4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_u, offset + 5] += 1
        U[a00_u, offset + 5] -= 1
        V[b00_v, offset + 5] += 1
        V[b01_v, offset + 5] += 1
        W[c11_w, offset + 5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_u, offset + 6] += 1
        U[a11_u, offset + 6] -= 1
        V[b10_v, offset + 6] += 1
        V[b11_v, offset + 6] += 1
        W[c00_w, offset + 6] += 1

    # 4 blocks
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(7, 0, 0, 0, 2, 0, 2)
    add_strassen_block(14, 0, 2, 2, 0, 0, 0)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    # Verify using vectorized operations
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        rng = np.random.RandomState(42)
        for _ in range(3):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            # C_computed[i,j] = sum_r U[:,r] dot A_flat * V[:,r] dot B_flat * W[j*n+i, r]
            # Let UA = U.T @ A_flat (shape rank,)
            # Let VB = V.T @ B_flat (shape rank,)
            # C_flat = W.T @ (UA * VB)
            UA = U_mat.T @ A_flat
            VB = V_mat.T @ B_flat
            C_flat = W_mat.T @ (UA * VB)
            C_computed = C_flat.reshape(p, n).T # W index is k*n+i, so reshape to (p,n) then transpose to (n,p)
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    if not verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    # Build matrix of flattened tensors
    vecs = []
    for r in range(rank):
        vec = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r]).flatten()
        vecs.append(vec)
    mat = np.array(vecs).T # (1024, 28)
    
    try:
        s_vals, _, Vt = np.linalg.svd(mat, full_matrices=False)
        rank_mat = np.sum(s_vals > 1e-10)
        
        if rank_mat < rank:
            # Find the null space vector corresponding to the smallest singular value
            # The null space is spanned by the last (rank - rank_mat) rows of Vt
            # We want to eliminate one term. We pick the term with the largest coefficient in the null vector.
            null_vec = Vt[rank_mat] 
            max_idx = np.argmax(np.abs(null_vec))
            
            U_sub = np.delete(U, max_idx, axis=1)
            V_sub = np.delete(V, max_idx, axis=1)
            W_sub = np.delete(W, max_idx, axis=1)
            
            if verify_decomposition(U_sub, V_sub, W_sub, n, m, p):
                return {"U": U_sub.tolist(), "V": V_sub.tolist(), "W": W_sub.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
