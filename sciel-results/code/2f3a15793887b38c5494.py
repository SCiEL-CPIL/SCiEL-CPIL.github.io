import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _construct_rank_25():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return U, V, W

def _try_find_lower_rank():
    """
    Attempt to find a rank 22 or 21 decomposition using numerical search.
    Returns (U, V, W) if found and verified, else None.
    """
    n, m, p = 2, 2, 7
    target_ranks = [22, 21]
    
    # Construct the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    # A[i,j] B[j,k] -> C[k,i]
    # T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    for rank in target_ranks:
        if _time_remaining() < 5.0:
            return None
            
        # Initialize U, V, W
        # We can fix W to be a standard basis or random to reduce symmetry?
        # Let's use random initialization for all.
        rng = np.random.RandomState(rank)
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Simple gradient descent / alternating least squares
        # Minimize || T - sum(U_r V_r W_r) ||^2
        # We can fix W and solve for U, V? No, it's trilinear.
        # We can use a simple iterative update:
        # U_r = argmin_U || T - sum U_r V_r W_r ||^2 (fixing V, W)
        # This is a linear least squares problem for each r.
        
        for step in range(50):
            if _time_remaining() < 2.0:
                return None
            
            # Update U
            for r in range(rank):
                # T - sum_{r' != r} U_{r'} V_{r'} W_{r'} = U_r V_r W_r
                # Residual R = T - (Total - Term_r)
                # We want U_r such that U_r V_r W_r approx R
                # This is not straightforward because U_r, V_r, W_r are all variables.
                # Instead, we can fix V and W, and update U.
                # R = T - sum_{r' != r} U_{r'} V_{r'} W_{r'}
                # We want to minimize || R - U_r (V_r \otimes W_r)^T ||_F
                # Let M_r = V_r \otimes W_r (size m*p*p*n)
                # Then R_flat = U_r_flat M_r^T
                # U_r_flat = R_flat M_r (M_r^T M_r)^{-1}
                
                # Compute residual
                R = T.copy()
                for r2 in range(rank):
                    if r2 == r: continue
                    R -= np.tensordot(U[:, r2], np.tensordot(V[:, r2], W[:, r2], axes=1), axes=1)
                
                # M_r is (m*p*p*n) x 1? No.
                # V_r is (m*p), W_r is (p*n).
                # Term is U_r (size n*m) * V_r (size m*p) * W_r (size p*n)
                # This is an outer product.
                # We can vectorize T as (n*m, m*p*p*n).
                # T_flat[i, j, k] = T[i, j, k]
                # Term_r_flat[i, j, k] = U_r[i] V_r[j] W_r[k]
                # So R_flat[i, :] = U_r[i] (V_r \otimes W_r)^T
                # Let M_r = V_r \otimes W_r (size m*p*p*n)
                # R_flat is (n*m, m*p*p*n)
                # U_r[i] = (R_flat[i, :] M_r) / (M_r^T M_r)
                
                M_r = np.kron(V[:, r], W[:, r])
                R_flat = R.reshape(n*m, -1)
                
                # Solve for U_r
                denom = np.dot(M_r, M_r)
                if denom < 1e-10:
                    continue
                U[:, r] = np.dot(R_flat, M_r) / denom
                
            # Update V
            for r in range(rank):
                R = T.copy()
                for r2 in range(rank):
                    if r2 == r: continue
                    R -= np.tensordot(U[:, r2], np.tensordot(V[:, r2], W[:, r2], axes=1), axes=1)
                
                # R_flat[j, i, k] = U_r[i] V_r[j] W_r[k]
                # R_flat[j, :] = V_r[j] (U_r \otimes W_r)^T
                M_r = np.kron(U[:, r], W[:, r])
                R_flat = R.reshape(m*p, n*m, p*n).transpose(1, 0, 2).reshape(m*p, -1)
                # Wait, R is (n*m, m*p, p*n).
                # We want R[j, i, k] = U[i] V[j] W[k].
                # For fixed j, R[:, j, :] = U (V[j] W)^T? No.
                # R[i, j, k] = U[i] V[j] W[k].
                # Reshape R to (m*p, n*m, p*n) -> R_transposed[j, i, k]
                # R_transposed[j, :] = V[j] (U \otimes W)^T
                M_r = np.kron(U[:, r], W[:, r])
                R_trans = R.transpose(1, 0, 2) # (m*p, n*m, p*n)
                R_flat = R_trans.reshape(m*p, -1)
                
                denom = np.dot(M_r, M_r)
                if denom < 1e-10:
                    continue
                V[:, r] = np.dot(R_flat, M_r) / denom

            # Update W
            for r in range(rank):
                R = T.copy()
                for r2 in range(rank):
                    if r2 == r: continue
                    R -= np.tensordot(U[:, r2], np.tensordot(V[:, r2], W[:, r2], axes=1), axes=1)
                
                # R[i, j, k] = U[i] V[j] W[k]
                # For fixed k, R[:, :, k] = (U \otimes V) W[k]
                M_r = np.kron(U[:, r], V[:, r])
                R_trans = R.transpose(2, 0, 1) # (p*n, n*m, m*p)
                R_flat = R_trans.reshape(p*n, -1)
                
                denom = np.dot(M_r, M_r)
                if denom < 1e-10:
                    continue
                W[:, r] = np.dot(R_flat, M_r) / denom

        # Check error
        T_recon = np.zeros_like(T)
        for r in range(rank):
            T_recon += np.tensordot(U[:, r], np.tensordot(V[:, r], W[:, r], axes=1), axes=1)
        
        error = np.linalg.norm(T - T_recon)
        if error < 1e-6:
            # Snap to integers/half-integers
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify snapped
            T_recon_snapped = np.zeros_like(T)
            for r in range(rank):
                T_recon_snapped += np.tensordot(U_int[:, r], np.tensordot(V_int[:, r], W_int[:, r], axes=1), axes=1)
            
            if np.allclose(T_recon_snapped, T, atol=1e-9):
                return U_int, V_int, W_int

    return None

def construct():
    if _time_remaining() < 5.0:
        U, V, W = _construct_rank_25()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank
    result = _try_find_lower_rank()
    
    if result is not None:
        U, V, W = result
    else:
        U, V, W = _construct_rank_25()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        C_W += W[:, r] * SA * SB
            
    C_mat = np.zeros((2,7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
            
    if np.allclose(C_mat, C_expected, atol=1e-10):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
