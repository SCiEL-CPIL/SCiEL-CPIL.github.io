"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W, atol=1e-9):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_true[a,b,c]."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=atol)

    def schoolbook():
        """Return the standard rank-56 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Strassen decomposition for 2x2 * 2x14 by splitting into 7 blocks of 2x2.
        
        For each block q (columns 2q, 2q+1), use the 7-term Strassen algorithm.
        Total rank = 7 blocks * 7 terms = 49.
        
        Strassen for 2x2 * 2x2:
          A = [[a,b],[c,d]], B = [[e,f],[g,h]]
          C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
          
          M1 = (a+d)(e+h)
          M2 = (c+d)e
          M3 = a(f-g)
          M4 = d(h-f)
          M5 = (a+b)h
          M6 = (c-a)(e+f)
          M7 = (b-d)(g+h)
          
          C11 = M1 + M4 - M5 + M7
          C12 = M3 + M5
          C21 = M2 + M4
          C22 = M1 - M2 + M3 + M6
        """
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        
        # A indices: a = i*m + j, so A00=0, A01=1, A10=2, A11=3
        # B indices: b = j*p + k, so for block q: B0_k0=2q, B0_k1=2q+1, B1_k0=14+2q, B1_k1=14+2q+1
        # C indices: c = k*n + i, so C0_k0=4q, C0_k1=4q+2, C1_k0=4q+1, C1_k1=4q+3
        
        for q in range(7):
            k0 = 2 * q
            k1 = 2 * q + 1
            
            # B indices
            v_B00 = 0 * p + k0    # B_{0,k0}
            v_B01 = 0 * p + k1    # B_{0,k1}
            v_B10 = 1 * p + k0    # B_{1,k0}
            v_B11 = 1 * p + k1    # B_{1,k1}
            
            # C indices
            w_C00 = k0 * n + 0    # C_{0,k0}
            w_C01 = k1 * n + 0    # C_{0,k1}
            w_C10 = k0 * n + 1    # C_{1,k0}
            w_C11 = k1 * n + 1    # C_{1,k1}
            
            # A indices
            u_A00 = 0 * m + 0     # A_{0,0}
            u_A01 = 0 * m + 1     # A_{0,1}
            u_A10 = 1 * m + 0     # A_{1,0}
            u_A11 = 1 * m + 1     # A_{1,1}
            
            col_base = q * 7
            
            # M1 = (A00 + A11)(B00 + B11)
            c = col_base + 0
            U[u_A00, c] += 1
            U[u_A11, c] += 1
            V[v_B00, c] += 1
            V[v_B11, c] += 1
            W[w_C00, c] += 1
            W[w_C11, c] += 1
            
            # M2 = (A10 + A11) * B00
            c = col_base + 1
            U[u_A10, c] += 1
            U[u_A11, c] += 1
            V[v_B00, c] += 1
            W[w_C10, c] += 1
            W[w_C11, c] -= 1
            
            # M3 = A00 * (B01 - B11)
            c = col_base + 2
            U[u_A00, c] += 1
            V[v_B01, c] += 1
            V[v_B11, c] -= 1
            W[w_C01, c] += 1
            W[w_C11, c] += 1
            
            # M4 = A11 * (B10 - B00)
            c = col_base + 3
            U[u_A11, c] += 1
            V[v_B10, c] += 1
            V[v_B00, c] -= 1
            W[w_C00, c] += 1
            W[w_C10, c] += 1
            
            # M5 = (A00 + A01) * B11
            c = col_base + 4
            U[u_A00, c] += 1
            U[u_A01, c] += 1
            V[v_B11, c] += 1
            W[w_C00, c] -= 1
            W[w_C01, c] += 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            c = col_base + 5
            U[u_A10, c] += 1
            U[u_A00, c] -= 1
            V[v_B00, c] += 1
            V[v_B01, c] += 1
            W[w_C11, c] += 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            c = col_base + 6
            U[u_A01, c] += 1
            U[u_A11, c] -= 1
            V[v_B10, c] += 1
            V[v_B11, c] += 1
            W[w_C00, c] += 1
        
        return U, V, W

    # Try the rank-49 Strassen decomposition first
    U, V, W = strassen_block_decomposition()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails with strict tolerance, try looser
    if verify(U, V, W, atol=1e-6):
        U_clean = np.round(U * 2) / 2
        V_clean = np.round(V * 2) / 2
        W_clean = np.round(W * 2) / 2
        if verify(U_clean, W_clean, V_clean):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 56
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
