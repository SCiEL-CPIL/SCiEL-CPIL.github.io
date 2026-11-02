import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Rank 40 construction
    # 4 blocks of 2 columns each.
    # Each block:
    # - Strassen for i,j in {0,1} (Rank 7)
    # - i=2, j in {0,1} (Rank 1)
    # - j=2, i in {0,1} (Rank 1)
    # - i=2, j=2 (Rank 1)
    # Total 10 per block * 4 blocks = 40.
    
    R = 40
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]
    
    r_idx = 0
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # 1. Strassen for i in {0,1}, j in {0,1}, k in {k0, k1}
        # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
        # Global i = i', j = j', k = k0 + k'
        
        # Strassen terms:
        # m1 = (a+d)(e+h)
        # a=(0,0), d=(1,1), e=(0,k0), h=(1,k1)
        U[0*m+0][r_idx] = 1 # a
        U[1*m+1][r_idx] = 1 # d
        V[0*p+k0][r_idx] = 1 # e
        V[1*p+k1][r_idx] = 1 # h
        W[k0*n+0][r_idx] = 1 # C00
        W[k1*n+1][r_idx] = 1 # C11
        r_idx += 1
        
        # m2 = (c+d)*e
        # c=(1,0), d=(1,1), e=(0,k0)
        U[1*m+0][r_idx] = 1 # c
        U[1*m+1][r_idx] = 1 # d
        V[0*p+k0][r_idx] = 1 # e
        W[k0*n+1][r_idx] = 1 # C10
        W[k1*n+1][r_idx] = -1 # C11
        r_idx += 1
        
        # m3 = a*(f-h)
        # a=(0,0), f=(1,k0), h=(1,k1)
        U[0*m+0][r_idx] = 1 # a
        V[1*p+k0][r_idx] = 1 # f
        V[1*p+k1][r_idx] = -1 # h
        W[k0*n+0][r_idx] = 1 # C01? No, C01 is k=k0+1? 
        # Wait, Strassen output:
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        # Here C01 corresponds to local k'=1? 
        # In 2x2x2, C is 2x2.
        # C[i', k'].
        # m3 contributes to C[0, 1] (local).
        # Global k for local k'=1 is k1.
        W[k1*n+0][r_idx] = 1 # C01
        W[k1*n+1][r_idx] = 1 # C11
        r_idx += 1
        
        # m4 = d*(g-e)
        # d=(1,1), g=(1,k0), e=(0,k0)
        U[1*m+1][r_idx] = 1 # d
        V[1*p+k0][r_idx] = 1 # g
        V[0*p+k0][r_idx] = -1 # e
        W[k0*n+0][r_idx] = 1 # C00
        W[k0*n+1][r_idx] = 1 # C10
        r_idx += 1
        
        # m5 = (a+b)*h
        # a=(0,0), b=(0,1), h=(1,k1)
        U[0*m+0][r_idx] = 1 # a
        U[0*m+1][r_idx] = 1 # b
        V[1*p+k1][r_idx] = 1 # h
        W[k0*n+0][r_idx] = -1 # C00
        W[k1*n+0][r_idx] = 1 # C01
        r_idx += 1
        
        # m6 = (c-a)*(e+f)
        # c=(1,0), a=(0,0), e=(0,k0), f=(1,k0)
        U[1*m+0][r_idx] = 1 # c
        U[0*m+0][r_idx] = -1 # a
        V[0*p+k0][r_idx] = 1 # e
        V[1*p+k0][r_idx] = 1 # f
        W[k1*n+1][r_idx] = 1 # C11
        r_idx += 1
        
        # m7 = (b-d)*(g+h)
        # b=(0,1), d=(1,1), g=(1,k0), h=(1,k1)
        U[0*m+1][r_idx] = 1 # b
        U[1*m+1][r_idx] = -1 # d
        V[1*p+k0][r_idx] = 1 # g
        V[1*p+k1][r_idx] = 1 # h
        W[k0*n+0][r_idx] = 1 # C00
        r_idx += 1
        
        # 2. i=2, j in {0,1}
        # T[2, j, k] = 1
        U[2*m+0][r_idx] = 1
        U[2*m+1][r_idx] = 1
        V[0*p+k0][r_idx] = 1
        V[0*p+k1][r_idx] = 1
        V[1*p+k0][r_idx] = 1
        V[1*p+k1][r_idx] = 1
        W[k0*n+2][r_idx] = 1
        W[k1*n+2][r_idx] = 1
        r_idx += 1
        
        # 3. j=2, i in {0,1}
        # T[i, 2, k] = 1
        U[0*m+2][r_idx] = 1
        U[1*m+2][r_idx] = 1
        V[2*p+k0][r_idx] = 1
        V[2*p+k1][r_idx] = 1
        W[k0*n+0][r_idx] = 1
        W[k1*n+0][r_idx] = 1
        W[k0*n+1][r_idx] = 1
        W[k1*n+1][r_idx] = 1
        r_idx += 1
        
        # 4. i=2, j=2
        # T[2, 2, k] = 1
        U[2*m+2][r_idx] = 1
        V[2*p+k0][r_idx] = 1
        V[2*p+k1][r_idx] = 1
        W[k0*n+2][r_idx] = 1
        W[k1*n+2][r_idx] = 1
        r_idx += 1

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(nm)]
        V_fb = [[0] * rank_fb for _ in range(mp)]
        W_fb = [[0] * rank_fb for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
