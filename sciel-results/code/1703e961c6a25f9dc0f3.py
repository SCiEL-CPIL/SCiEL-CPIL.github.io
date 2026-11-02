"""Matrix multiplication 2x4 * 4x7 rank-minimal tensor decomposition."""
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication."""
    deadline = time.time() + 110
    
    n, m, p = 2, 4, 7
    
    # Target rank 50 based on 2x(2x2x7) decomposition
    # Each 2x2x7 uses 3 Strassen blocks (rank 21) + 1 schoolbook (rank 4) = 25.
    # Total 50.
    
    R = 50
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    col_idx = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen(i_base, j_base, k_base):
        nonlocal col_idx
        # A indices
        a0 = i_base * m + j_base
        a1 = i_base * m + j_base + 1
        a2 = (i_base + 1) * m + j_base
        a3 = (i_base + 1) * m + j_base + 1
        
        # B indices
        b0 = j_base * p + k_base
        b1 = j_base * p + k_base + 1
        b2 = (j_base + 1) * p + k_base
        b3 = (j_base + 1) * p + k_base + 1
        
        # C indices (W)
        c00 = k_base * n + i_base
        c01 = (k_base + 1) * n + i_base
        c10 = k_base * n + i_base + 1
        c11 = (k_base + 1) * n + i_base + 1
        
        # M1 = (a0+a3)(b0+b3) -> c00 += M1, c11 += M1
        U[a0, col_idx] += 1; U[a3, col_idx] += 1
        V[b0, col_idx] += 1; V[b3, col_idx] += 1
        W[c00, col_idx] += 1; W[c11, col_idx] += 1
        col_idx += 1
        
        # M2 = (a2+a3)b0 -> c10 += M2, c11 -= M2
        U[a2, col_idx] += 1; U[a3, col_idx] += 1
        V[b0, col_idx] += 1
        W[c10, col_idx] += 1; W[c11, col_idx] -= 1
        col_idx += 1
        
        # M3 = a0(b1-b3) -> c01 += M3, c11 += M3
        U[a0, col_idx] += 1
        V[b1, col_idx] += 1; V[b3, col_idx] -= 1
        W[c01, col_idx] += 1; W[c11, col_idx] += 1
        col_idx += 1
        
        # M4 = a3(b2-b0) -> c00 += M4, c10 += M4
        U[a3, col_idx] += 1
        V[b2, col_idx] += 1; V[b0, col_idx] -= 1
        W[c00, col_idx] += 1; W[c10, col_idx] += 1
        col_idx += 1
        
        # M5 = (a0+a1)b3 -> c00 -= M5, c01 += M5
        U[a0, col_idx] += 1; U[a1, col_idx] += 1
        V[b3, col_idx] += 1
        W[c00, col_idx] -= 1; W[c01, col_idx] += 1
        col_idx += 1
        
        # M6 = (a2-a0)(b0+b1) -> c11 += M6
        U[a2, col_idx] += 1; U[a0, col_idx] -= 1
        V[b0, col_idx] += 1; V[b1, col_idx] += 1
        W[c11, col_idx] += 1
        col_idx += 1
        
        # M7 = (a1-a3)(b2+b3) -> c00 += M7
        U[a1, col_idx] += 1; U[a3, col_idx] -= 1
        V[b2, col_idx] += 1; V[b3, col_idx] += 1
        W[c00, col_idx] += 1
        col_idx += 1

    # Helper to add 2x2x1 schoolbook block
    def add_schoolbook_221(i_base, j_base, k_base):
        nonlocal col_idx
        a0 = i_base * m + j_base
        a1 = i_base * m + j_base + 1
        a2 = (i_base + 1) * m + j_base
        a3 = (i_base + 1) * m + j_base + 1
        
        b0 = j_base * p + k_base
        b1 = (j_base + 1) * p + k_base
        
        c0 = k_base * n + i_base
        c1 = k_base * n + i_base + 1
        
        U[a0, col_idx] += 1; V[b0, col_idx] += 1; W[c0, col_idx] += 1; col_idx += 1
        U[a1, col_idx] += 1; V[b1, col_idx] += 1; W[c0, col_idx] += 1; col_idx += 1
        U[a2, col_idx] += 1; V[b0, col_idx] += 1; W[c1, col_idx] += 1; col_idx += 1
        U[a3, col_idx] += 1; V[b1, col_idx] += 1; W[c1, col_idx] += 1; col_idx += 1

    # Process two j-blocks: (0,1) and (2,3)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen(0, j0, k0)
        add_schoolbook_221(0, j0, 6)
        
    # Verify
    def check_validity(U, V, W):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = 0
                        for c in range(R):
                            val += U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
                        if abs(val - (1 if j==jp else 0)) > 1e-9:
                            return False
        return True

    if check_validity(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    R_fallback = n * m * p
    U_f = np.zeros((n * m, R_fallback), dtype=int)
    V_f = np.zeros((m * p, R_fallback), dtype=int)
    W_f = np.zeros((p * n, R_fallback), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i*m+j, idx] = 1
                V_f[j*p+k, idx] = 1
                W_f[k*n+i, idx] = 1
                idx += 1
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
