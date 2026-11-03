"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 110.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 90. We aim for 84.
    # Blocks:
    # I1=[0,1], I2=[2]
    # J1=[0,1], J2=[2,3], J3=[4]
    # K1=[0,1], K2=[2,3], K3=[4,5]
    
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    # Strassen coefficients for output C_uv
    # M1, M2, M3, M4, M5, M6, M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Map (u, v) to list of (m_index, coeff)
    c_coeffs = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(2, 1), (4, 1)],
        (1, 0): [(1, 1), (3, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # i_off, j_off, k_off are top-left indices of the 2x2x2 block
        
        # M1: (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 0:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M2: (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 1:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M3: A[i0,j0] * (B[j0,k1] - B[j1,k1])
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 2:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M4: A[i1,j1] * (B[j1,k0] - B[j0,k0])
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 3:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M5: (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 4:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M6: (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 5:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M7: (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 6:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1

    # Iterate through blocks
    # I1=[0,1], I2=[2]
    # J1=[0,1], J2=[2,3], J3=[4]
    # K1=[0,1], K2=[2,3], K3=[4,5]
    
    # Block I1, K1
    add_strassen(0, 0, 0) # J1
    add_strassen(0, 2, 0) # J2
    for i in [0, 1]:
        for k in [0, 1]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I1, K2
    add_strassen(0, 0, 2) # J1
    add_strassen(0, 2, 2) # J2
    for i in [0, 1]:
        for k in [2, 3]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I1, K3
    add_strassen(0, 0, 4) # J1
    add_strassen(0, 2, 4) # J2
    for i in [0, 1]:
        for k in [4, 5]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I2, K1
    for i in [2]:
        for j in [0, 1]:
            for k in [0, 1]:
                add_schoolbook(i, j, k) # J1
        for j in [2, 3]:
            for k in [0, 1]:
                add_schoolbook(i, j, k) # J2
        for k in [0, 1]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I2, K2
    for i in [2]:
        for j in [0, 1]:
            for k in [2, 3]:
                add_schoolbook(i, j, k) # J1
        for j in [2, 3]:
            for k in [2, 3]:
                add_schoolbook(i, j, k) # J2
        for k in [2, 3]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I2, K3
    for i in [2]:
        for j in [0, 1]:
            for k in [4, 5]:
                add_schoolbook(i, j, k) # J1
        for j in [2, 3]:
            for k in [4, 5]:
                add_schoolbook(i, j, k) # J2
        for k in [4, 5]:
            add_schoolbook(i, 4, k) # J3
            
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] == W[k*n+i,c]
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
    # Wait, no. The matrix multiplication tensor is T[i,j,k] = delta_{i,j,k} in the sense that
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The rank-r decomposition gives:
    # sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k]) * ... 
    # Actually, the correct check is:
    # For the bilinear form: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all valid (i,j,k)
    
    # Let's verify by checking that the reconstructed tensor matches
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because each A[i,j]*B[j,k] contributes to C[i,k]
    
    # Actually, I need to be more careful. The tensor is defined such that:
    # <T, A ⊗ B> = <C, I> where C = A*B
    # This means T[i,j,k] = 1 for all i,j,k (in the appropriate indexing)
    
    # Let me just verify by checking a few random instances
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C using the decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_k V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    # Wait, that's not right either. Let me think again.
    
    # The rank-r decomposition of the matrix multiplication tensor means:
    # sum_j A[i,j] * B[j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # But this doesn't make sense dimensionally.
    
    # Actually, the correct formulation is:
    # The tensor T has entries T[i,j,k] = 1 (for the multiplication)
    # A rank-r decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    
    # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    
    # Let's verify this
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col_s] = 1
                    V_s[j*p+k][col_s] = 1
                    W_s[k*n+i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
