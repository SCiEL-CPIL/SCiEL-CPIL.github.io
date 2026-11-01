import time
import numpy as np

# EVOLVE-BLOCK-START

def _get_strassen_2x2_factors():
    """
    Returns U, V, W factors for 2x2 Strassen decomposition.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    
    Convention:
    U[i*m + j, r] corresponds to coefficient of A[i,j] in product r.
    V[j*p + k, r] corresponds to coefficient of B[j,k] in product r.
    W[k*n + i, r] corresponds to coefficient of M_r in C[i,k].
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 7 products:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Indices for 2x2:
    # A: [[0,1],[2,3]] -> 0:00, 1:01, 2:10, 3:11
    # B: [[0,1],[2,3]] -> 0:00, 1:01, 2:10, 3:11
    # C: [[0,1],[2,3]] -> 0:00, 1:01, 2:10, 3:11
    
    # M0
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # M1
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1               # B00
    W[2, 1] = 1; W[3, 1] = 1  # C10, C11 (Wait, C10 uses M1, C11 uses M1? 
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    # So W for C10 (idx 2) gets M1. W for C11 (idx 3) gets M1.
    W[2, 1] = 1
    W[3, 1] = -1 # Coefficient in C11 is -1
    
    # M2
    U[0, 2] = 1               # A00
    V[2, 2] = 1; V[3, 2] = -1 # B10 - B11
    W[1, 2] = 1               # C01 (idx 1)
    W[3, 2] = 1               # C11 (idx 3)
    
    # M3
    U[3, 3] = 1               # A11
    V[1, 3] = 1; V[0, 3] = -1 # B01 - B00
    W[0, 3] = 1               # C00 (idx 0)
    W[2, 3] = 1               # C10 (idx 2)
    
    # M4
    U[0, 4] = 1; U[1, 4] = 1  # A00 + A01
    V[3, 4] = 1               # B11
    W[0, 4] = -1              # C00 (idx 0)
    W[1, 4] = 1               # C01 (idx 1)
    
    # M5
    U[2, 5] = 1; U[0, 5] = -1 # A10 - A00
    V[0, 5] = 1; V[1, 5] = 1  # B00 + B01
    W[3, 5] = 1               # C11 (idx 3)
    
    # M6
    U[1, 6] = 1; U[3, 6] = -1 # A01 - A11
    V[2, 6] = 1; V[3, 6] = 1  # B10 + B11
    W[0, 6] = 1               # C00 (idx 0)
    
    return U, V, W

def _build_strassen_4x4():
    """
    Builds rank-49 decomposition for 4x4 using Kronecker products of 2x2 factors.
    """
    U2, V2, W2 = _get_strassen_2x2_factors()
    
    # Kronecker product of factors
    # U4[i,j, r1,r2] = U2[i1,j1, r1] * U2[i2,j2, r2]
    # We need to reshape to 1D indices for the final output.
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Reshape U2 to (2, 2, 7)
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    
    # Compute Kronecker products
    # U4 shape: (4, 4, 49)
    U4 = np.einsum('ijr,kl s->ikjlrs', U2_3d, U2_3d)
    # This creates a tensor with indices i,k,j,l,r,s. 
    # We want to map (i1, i2) -> I, (j1, j2) -> J, (r1, r2) -> R.
    # I = 2*i1 + i2, J = 2*j1 + j2, R = 7*r1 + r2.
    
    # Let's do it explicitly to be safe with memory and indexing.
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            R = 7 * r1 + r2
            for i1 in range(2):
                for i2 in range(2):
                    I = 2 * i1 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            J = 2 * j1 + j2
                            U4[I, J, R] = U2_3d[i1, j1, r1] * U2_3d[i2, j2, r2]
                            
                            V4[I, J, R] = V2_3d[i1, j1, r1] * V2_3d[i2, j2, r2]
                            
                            # W convention: W[k, i, r]
                            # W4[K, I, R] = W2_3d[k1, i1, r1] * W2_3d[k2, i2, r2]
                            # Note: W2 is indexed [k, i, r] where k is row of B (col of A in product?), 
                            # In 2x2 Strassen, W[k, i, r] means coefficient of M_r in C[i,k].
                            # So W2_3d[k, i, r].
                            for k1 in range(2):
                                for k2 in range(2):
                                    K = 2 * k1 + k2
                                    W4[K, I, R] += W2_3d[k1, i1, r1] * W2_3d[k2, i2, r2]
    
    # Reshape to (16, 49)
    # U: (4, 4, 49) -> (16, 49). Index i*m+j.
    U_out = U4.reshape(16, 49)
    V_out = V4.reshape(16, 49)
    W_out = W4.reshape(16, 49)
    
    return U_out, V_out, W_out

def _verify_decomposition(U, V, W, n=4, m=4, p=4):
    """
    Verify C = A @ B using the decomposition.
    """
    rank = U.shape[1]
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    rng = np.random.default_rng(123)
    for _ in range(5):
        A = rng.integers(-10, 10, size=(n, m)).astype(float)
        B = rng.integers(-10, 10, size=(m, p)).astype(float)
        C_target = A @ B
        
        term_u = np.einsum('ijr,ij->r', U4, A)
        term_v = np.einsum('jkr,jk->r', V4, B)
        C_recon = np.einsum('kir,r,r->ik', W4, term_u, term_v)
        
        if not np.allclose(C_recon, C_target, atol=1e-9):
            return False
    return True

def construct():
    start_time = time.time()
    
    # Build Rank 49
    U, V, W = _build_strassen_4x4()
    
    if _verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback Rank 64
        n, m, p = 4, 4, 4
        rank = 64
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
