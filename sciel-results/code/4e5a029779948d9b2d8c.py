"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication using
Kronecker product of 2x2 Strassen factors.

The 4x4 multiplication tensor T[i,j,k] = delta[i,k] decomposes as
T = T_2 (x) T_2 (Kronecker product), where T_2 is the 2x2 multiplication tensor.
The rank-49 factors are U = U_2 (x) U_2, V = V_2 (x) V_2, W = W_2 (x) W_2
with rank index r = 7*r1 + r2.
"""

import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen factor matrices for 2x2 matrix multiplication.
    
    Convention:
      U[i*2+j, r] = coefficient of A[i,j] in multiplication r
      V[j*2+k, r] = coefficient of B[j,k] in multiplication r  
      W[k*2+i, r] = coefficient for C[i,j] in multiplication r (trace convention: W[k*n+i, r])
    
    Strassen's 7 multiplications:
      M1 = (A00+A11)(B00+B11)  -> C00 += 1, C11 += 1
      M2 = (A10+A11)B00        -> C10 += 1
      M3 = A00(B01-B11)        -> C01 += 1, C11 -= 1
      M4 = A11(B10-B00)        -> C00 += 1, C10 += 1
      M5 = (A00+A01)B11        -> C00 -= 1, C01 += 1
      M6 = (A10-A00)(B00+B01)  -> C11 += 1
      M7 = (A01-A11)(B10+B11)  -> C00 += 1, C11 -= 1
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00 -> C10+=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1
    # M3 = A00(B01-B11) -> C01+=1, C11-=1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=-1
    # M4 = A11(B10-B00) -> C00+=1, C10+=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11 -> C00-=1, C01+=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01) -> C11+=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11) -> C00+=1, C11-=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[3,6]=-1
    
    return U, V, W


def _build_rank49_kronecker():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication using
    Kronecker product of 2x2 Strassen factors.
    
    The 4x4 multiplication tensor T[i,j,k] = delta[i,k] where i,j,k in {0,1,2,3}
    decomposes as T = T_2 (x) T_2 where T_2 is the 2x2 multiplication tensor.
    
    Using index decomposition i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2:
      T[i,j,k] = T_2[i1,j1,k1] * T_2[i2,j2,k2]
    
    The rank-49 factors (r = 7*r1 + r2):
      U[i,j,r]   = U_2[i1,j1,r1] * U_2[i2,j2,r2]
      V[j,k,r]   = V_2[j1,k1,r1] * V_2[j2,k2,r2]
      W[k,i,r]   = W_2[k1,i1,r1] * W_2[k2,i2,r2]
    
    U shape: (16, 49), indexed U[i*4+j, r]
    V shape: (16, 49), indexed V[j*4+k, r]
    W shape: (16, 49), indexed W[k*4+i, r]
    """
    U2, V2, W2 = _strassen_2x2()
    
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # Build U[:, r]: U[i*4+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            # where i = 2*i1 + i2, j = 2*j1 + j2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2+j1, r1]
                    if u1 != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                u2 = U2[i2*2+j2, r2]
                                if u2 != 0:
                                    i = 2*i1 + i2
                                    j = 2*j1 + j2
                                    U[i*4+j, r] = u1 * u2
            
            # Build V[:, r]: V[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
            # where j = 2*j1 + j2, k = 2*k1 + k2
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V2[j1*2+k1, r1]
                    if v1 != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                v2 = V2[j2*2+k2, r2]
                                if v2 != 0:
                                    j = 2*j1 + j2
                                    k = 2*k1 + k2
                                    V[j*4+k, r] = v1 * v2
            
            # Build W[:, r]: W[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
            # where k = 2*k1 + k2, i = 2*i1 + i2
            # W2 is indexed W2[local_k*2+local_i, r]
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W2[k1*2+i1, r1]
                    if w1 != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                w2 = W2[k2*2+i2, r2]
                                if w2 != 0:
                                    k = 2*k1 + k2
                                    i = 2*i1 + i2
                                    W[k*4+i, r] = w1 * w2
    
    return U, V, W


def _verify(U, V, W):
    """
    Verify the decomposition is exact.
    T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] should equal 1
    for all i,j,k (the 4x4 matrix multiplication tensor).
    """
    n = m = p = 4
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)  # U[i,j,r]
    V_r = V.reshape(m, p, rank)  # V[j,k,r]
    W_r = W.reshape(p, n, rank)  # W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    # T[i,j,k] should be 1 if i==k, 0 otherwise (delta[i,k])
    expected = np.zeros((4,4,4), dtype=float)
    for i in range(4):
        for k in range(4):
            if i == k:
                for j in range(4):
                    expected[i,j,k] = 1.0
    return np.allclose(T, expected, atol=1e-9)


def _verify_detailed(U, V, W):
    """Detailed verification returning max error."""
    n = m = p = 4
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    expected = np.zeros((4,4,4), dtype=float)
    for i in range(4):
        for k in range(4):
            if i == k:
                for j in range(4):
                    expected[i,j,k] = 1.0
    max_err = np.max(np.abs(T - expected))
    return max_err, T


def _schoolbook():
    """Fallback: rank-64 schoolbook decomposition."""
    n = m = p = 4
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    deadline = time.time() + 115.0
    
    # Primary: rank-49 Kronecker-product Strassen
    try:
        U, V, W = _build_rank49_kronecker()
        if time.time() < deadline:
            max_err, T = _verify_detailed(U, V, W)
            if max_err < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            elif max_err < 1e-6:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: rank-64 schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
