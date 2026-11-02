import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def verify(U, V, W):
        """Check if decomposition is exact."""
        R = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-10)
    
    # --- Rank 26 decomposition: Strassen (7) + 19 border terms ---
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen's 7 multiplications for 2x2 core (i,j,k in {0,1})
    # m1 = (a00+a11)(b00+b11)
    r = 0
    U[0*3+0, r] = 1  # a00
    U[1*3+1, r] = 1  # a11
    V[0*3+0, r] = 1  # b00
    V[1*3+1, r] = 1  # b11
    W[0*3+0, r] = 1  # c00
    W[1*3+1, r] = 1  # c11
    
    # m2 = (a10+a11)b00
    r = 1
    U[1*3+0, r] = 1  # a10
    U[1*3+1, r] = 1  # a11
    V[0*3+0, r] = 1  # b00
    W[0*3+1, r] = 1  # c01... wait, W[k*n+i] so c_{i,j} uses W[k*n+i]
    # Let me be more careful with indices
    # U index: i*m + j = i*3 + j
    # V index: j*p + k = j*3 + k  
    # W index: k*n + i = k*3 + i
    # So W[k, i] contributes to c[i, j]... no wait
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # So for a given (i,j,k), the contribution from rank r is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # This should equal 1 for all (i,j,k)
    
    # Let me redo this more carefully
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    def iu(i, j): return i * m + j
    def iv(j, k): return j * p + k
    def iw(k, i): return k * n + i
    
    # Strassen m1 = (a00+a11)(b00+b11) -> contributes to c00, c11
    r = 0
    U[iu(0,0), r] = 1; U[iu(1,1), r] = 1
    V[iv(0,0), r] = 1; V[iv(1,1), r] = 1
    W[iw(0,0), r] = 1; W[iw(1,1), r] = 1
    
    # m2 = (a10+a11)b00 -> contributes to c10, subtracts from c11
    r = 1
    U[iu(1,0), r] = 1; U[iu(1,1), r] = 1
    V[iv(0,0), r] = 1
    W[iw(0,1), r] = 1; W[iw(1,1), r] = -1
    
    # m3 = a00(b01-b11) -> contributes to c01, c11
    r = 2
    U[iu(0,0), r] = 1
    V[iv(0,1), r] = 1; V[iv(1,1), r] = -1
    W[iw(1,0), r] = 1; W[iw(1,1), r] = 1
    
    # m4 = a11(b10-b00) -> contributes to c00, c10
    r = 3
    U[iu(1,1), r] = 1
    V[iv(1,0), r] = 1; V[iv(0,0), r] = -1
    W[iw(0,0), r] = 1; W[iw(0,1), r] = 1
    
    # m5 = (a00+a01)b11 -> subtracts from c00, contributes to c01
    r = 4
    U[iu(0,0), r] = 1; U[iu(0,1), r] = 1
    V[iv(1,1), r] = 1
    W[iw(0,0), r] = -1; W[iw(1,0), r] = 1
    
    # m6 = (a10-a00)(b00+b01) -> contributes to c11
    r = 5
    U[iu(1,0), r] = 1; U[iu(0,0), r] = -1
    V[iv(0,0), r] = 1; V[iv(0,1), r] = 1
    W[iw(1,1), r] = 1
    
    # m7 = (a01-a11)(b10+b11) -> contributes to c00
    r = 6
    U[iu(0,1), r] = 1; U[iu(1,1), r] = -1
    V[iv(1,0), r] = 1; V[iv(1,1), r] = 1
    W[iw(0,0), r] = 1
    
    # Now add 19 border terms (all (i,j,k) where at least one index is 2)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                U[iu(i,j), col] = 1
                V[iv(j,k), col] = 1
                W[iw(k,i), col] = 1
                col += 1
    
    # Verify the decomposition
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: pure schoolbook rank 27
    R27 = n * m * p
    U27 = np.zeros((n * m, R27))
    V27 = np.zeros((m * p, R27))
    W27 = np.zeros((p * n, R27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[iu(i,j), col] = 1
                V27[iv(j,k), col] = 1
                W27[iw(k,i), col] = 1
                col += 1
    
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
