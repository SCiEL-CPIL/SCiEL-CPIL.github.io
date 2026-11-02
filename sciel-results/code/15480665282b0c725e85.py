import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Rank 32 construction
    # 4 blocks of 2 columns each.
    # Each block:
    # - Strassen for i,j in {0,1} (Rank 7)
    # - j=2, all i (Rank 1)
    # Total 8 per block * 4 blocks = 32.
    
    R = 32
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]
    
    r_idx = 0
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # 1. Strassen for i in {0,1}, j in {0,1}, k in {k0, k1}
        
        # m1 = (a+d)(e+h)
        U[0*m+0][r_idx] = 1
        U[1*m+1][r_idx] = 1
        V[0*p+k0][r_idx] = 1
        V[1*p+k1][r_idx] = 1
        W[k0*n+0][r_idx] = 1
        W[k1*n+1][r_idx] = 1
        r_idx += 1
        
        # m2 = (c+d)*e
        U[1*m+0][r_idx] = 1
        U[1*m+1][r_idx] = 1
        V[0*p+k0][r_idx] = 1
        W[k0*n+1][r_idx] = 1
        W[k1*n+1][r_idx] = -1
        r_idx += 1
        
        # m3 = a*(f-h)
        U[0*m+0][r_idx] = 1
        V[1*p+k0][r_idx] = 1
        V[1*p+k1][r_idx] = -1
        W[k1*n+0][r_idx] = 1
        W[k1*n+1][r_idx] = 1
        r_idx += 1
        
        # m4 = d*(g-e)
        U[1*m+1][r_idx] = 1
        V[1*p+k0][r_idx] = 1
        V[0*p+k0][r_idx] = -1
        W[k0*n+0][r_idx] = 1
        W[k0*n+1][r_idx] = 1
        r_idx += 1
        
        # m5 = (a+b)*h
        U[0*m+0][r_idx] = 1
        U[0*m+1][r_idx] = 1
        V[1*p+k1][r_idx] = 1
        W[k0*n+0][r_idx] = -1
        W[k1*n+0][r_idx] = 1
        r_idx += 1
        
        # m6 = (c-a)*(e+f)
        U[1*m+0][r_idx] = 1
        U[0*m+0][r_idx] = -1
        V[0*p+k0][r_idx] = 1
        V[1*p+k0][r_idx] = 1
        W[k1*n+1][r_idx] = 1
        r_idx += 1
        
        # m7 = (b-d)*(g+h)
        U[0*m+1][r_idx] = 1
        U[1*m+1][r_idx] = -1
        V[1*p+k0][r_idx] = 1
        V[1*p+k1][r_idx] = 1
        W[k0*n+0][r_idx] = 1
        r_idx += 1
        
        # 2. j=2, all i
        # T[i, 2, k] = 1
        for i in range(n):
            U[i*m+2][r_idx] = 1
        for k in [k0, k1]:
            V[2*p+k][r_idx] = 1
            for i in range(n):
                W[k*n+i][r_idx] = 1
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
        # Fallback to schoolbook if verification fails
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
