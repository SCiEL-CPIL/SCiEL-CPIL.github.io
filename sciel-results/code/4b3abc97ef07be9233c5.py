import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U_np = np.zeros((n * m, rank))
    V_np = np.zeros((m * p, rank))
    W_np = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U_np[u_idx, col] = 1.0
                V_np[v_idx, col] = 1.0
                W_np[w_idx, col] = 1.0
                col += 1
                
    return U_np, V_np, W_np

def _laderman_decomposition():
    n, m, p = DIMENSIONS
    rank = 23
    U_np = np.zeros((n * m, rank))
    V_np = np.zeros((m * p, rank))
    W_np = np.zeros((p * n, rank))
    
    # Helper to set values
    def set_U(r, i, j, val):
        U_np[i * m + j, r] += val
        
    def set_V(r, j, k, val):
        V_np[j * p + k, r] += val
        
    def set_W(r, k, i, val):
        W_np[k * n + i, r] += val

    # 1. A00 B00
    r = 0
    set_U(r, 0, 0, 1); set_V(r, 0, 0, 1); set_W(r, 0, 0, 1)
    
    # 2. A01 B10
    r = 1
    set_U(r, 0, 1, 1); set_V(r, 1, 0, 1); set_W(r, 0, 0, 1)
    
    # 3. A02 B20
    r = 2
    set_U(r, 0, 2, 1); set_V(r, 2, 0, 1); set_W(r, 0, 0, 1)
    
    # 4. A10 B01
    r = 3
    set_U(r, 1, 0, 1); set_V(r, 0, 1, 1); set_W(r, 1, 1, 1)
    
    # 5. A12 B21
    r = 4
    set_U(r, 1, 2, 1); set_V(r, 2, 1, 1); set_W(r, 1, 1, 1)
    
    # 6. A20 B02
    r = 5
    set_U(r, 2, 0, 1); set_V(r, 0, 2, 1); set_W(r, 2, 2, 1)
    
    # 7. A21 B12
    r = 6
    set_U(r, 2, 1, 1); set_V(r, 1, 2, 1); set_W(r, 2, 2, 1)
    
    # 8. (A00+A11)(B11+B22)
    r = 7
    set_U(r, 0, 0, 1); set_U(r, 1, 1, 1)
    set_V(r, 1, 1, 1); set_V(r, 2, 2, 1)
    set_W(r, 1, 0, 1); set_W(r, 2, 0, 1)
    set_W(r, 1, 1, 1); set_W(r, 2, 1, 1)
    
    # 9. (A01+A10)(B00+B11)
    r = 8
    set_U(r, 0, 1, 1); set_U(r, 1, 0, 1)
    set_V(r, 0, 0, 1); set_V(r, 1, 1, 1)
    set_W(r, 0, 0, 1); set_W(r, 1, 0, 1)
    set_W(r, 0, 1, 1); set_W(r, 1, 1, 1)
    
    # 10. (A02+A20)(B00+B22)
    r = 9
    set_U(r, 0, 2, 1); set_U(r, 2, 0, 1)
    set_V(r, 0, 0, 1); set_V(r, 2, 2, 1)
    set_W(r, 0, 0, 1); set_W(r, 2, 0, 1)
    set_W(r, 0, 2, 1); set_W(r, 2, 2, 1)
    
    # 11. (A10+A21)(B01+B12)
    r = 10
    set_U(r, 1, 0, 1); set_U(r, 2, 1, 1)
    set_V(r, 0, 1, 1); set_V(r, 1, 2, 1)
    set_W(r, 1, 1, 1); set_W(r, 2, 1, 1)
    set_W(r, 1, 2, 1); set_W(r, 2, 2, 1)
    
    # 12. (A11+A22)(B02+B10)
    r = 11
    set_U(r, 1, 1, 1); set_U(r, 2, 2, 1)
    set_V(r, 0, 2, 1); set_V(r, 1, 0, 1)
    set_W(r, 2, 1, 1); set_W(r, 0, 1, 1)
    set_W(r, 2, 2, 1); set_W(r, 0, 2, 1)
    
    # 13. (A12+A20)(B11+B20)
    r = 12
    set_U(r, 1, 2, 1); set_U(r, 2, 0, 1)
    set_V(r, 1, 1, 1); set_V(r, 2, 0, 1)
    set_W(r, 1, 1, 1); set_W(r, 0, 1, 1)
    set_W(r, 1, 2, 1); set_W(r, 0, 2, 1)
    
    # 14. (A00+A22)(B12+B21)
    r = 13
    set_U(r, 0, 0, 1); set_U(r, 2, 2, 1)
    set_V(r, 1, 2, 1); set_V(r, 2, 1, 1)
    set_W(r, 2, 0, 1); set_W(r, 1, 0, 1)
    set_W(r, 2, 2, 1); set_W(r, 1, 2, 1)
    
    # 15. (A01+A22)(B10+B21)
    r = 14
    set_U(r, 0, 1, 1); set_U(r, 2, 2, 1)
    set_V(r, 1, 0, 1); set_V(r, 2, 1, 1)
    set_W(r, 0, 0, 1); set_W(r, 1, 0, 1)
    set_W(r, 2, 2, 1); set_W(r, 1, 2, 1)
    
    # 16. (A02+A11)(B00+B12)
    r = 15
    set_U(r, 0, 2, 1); set_U(r, 1, 1, 1)
    set_V(r, 0, 0, 1); set_V(r, 1, 2, 1)
    set_W(r, 0, 0, 1); set_W(r, 2, 0, 1)
    set_W(r, 0, 1, 1); set_W(r, 2, 1, 1)
    
    # Note: The above 16 composite terms might not be the exact Laderman set.
    # Laderman's set has specific linear forms.
    # If verification fails, we fall back.
    
    return U_np, V_np, W_np

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    # Test with a few random matrices
    for _ in range(3):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        # Compute C_recon
        # C_recon[i, k] = sum_r (sum_ij U[i,j,r] A[i,j]) (sum_jk V[j,k,r] B[j,k]) W[k,i,r]
        # Flatten U, V, W for dot products
        U_flat = U.reshape(n*m, -1)
        V_flat = V.reshape(m*p, -1)
        W_flat = W.reshape(p*n, -1)
        
        # L_r = A_flat dot U_flat
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        L = A_flat @ U_flat
        M = B_flat @ V_flat
        
        # C_recon_flat = sum_r L[r] M[r] W_flat[:, r]
        # This is (L * M) dot W_flat? No.
        # C_recon_flat[k*n + i] = sum_r L[r] M[r] W_flat[k*n+i, r]
        # C_recon_flat = (L * M) @ W_flat.T ?
        # Let's check dimensions.
        # L is (rank,). M is (rank,).
        # L*M is (rank,).
        # W_flat is (p*n, rank).
        # (L*M) @ W_flat.T gives (p*n,).
        
        C_recon_flat = (L * M) @ W_flat.T
        C_recon = C_recon_flat.reshape(p, n)
        
        # The output index in W is k*n + i.
        # So C_recon_flat[k*n + i] corresponds to C[i, k].
        # So C_recon[k, i] in the reshaped (p, n) matrix is C[i, k].
        # We want to compare with C_true[i, k].
        # So C_recon.T should equal C_true.
        
        if not np.allclose(C_recon.T, C_true, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Try Laderman (Rank 23)
    U_l, V_l, W_l = _laderman_decomposition()
    if _verify(U_l, V_l, W_l):
        return {"U": U_l.tolist(), "V": V_l.tolist(), "W": W_l.tolist()}
        
    # Fallback to Schoolbook (Rank 27)
    U_s, V_s, W_s = _schoolbook_decomposition()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
