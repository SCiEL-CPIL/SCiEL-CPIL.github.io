import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n, m, p = 4, 4, 4
    
    def verify(U, V, W, timeout=10.0):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] = multiplication tensor."""
        start = time.time()
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        # Build target tensor: T[a_idx, b_idx, c_idx] = 1
        # A[i,k] -> a_idx = i*m + k
        # B[k,j] -> b_idx = k*p + j  
        # C[i,j] -> c_idx = j*n + i  (the k*n+i convention where k=j)
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a_idx = i * m + k
                    b_idx = k * p + j
                    c_idx = j * n + i
                    target[a_idx, b_idx, c_idx] = 1.0
        
        # Reconstruct: recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        
        if time.time() - start > timeout:
            return False
        
        return np.allclose(recon, target, atol=1e-9)
    
    def time_ok():
        return time.time() < deadline
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For 2x2 matrices: A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
    # C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
    # U row: i*2+k, V row: k*2+j, W row: j*2+i
    def strassen_2x2():
        U2 = np.zeros((4, 7), dtype=float)
        V2 = np.zeros((4, 7), dtype=float)
        W2 = np.zeros((4, 7), dtype=float)
        
        # M1 = (A00+A11)(B00+B11): contributes to C00 and C11
        # U: rows 0(A00) and 3(A11), V: rows 0(B00) and 3(B11), W: rows 0(C00) and 3(C11)
        U2[0,0] = 1; U2[3,0] = 1
        V2[0,0] = 1; V2[3,0] = 1
        W2[0,0] = 1; W2[3,0] = 1
        
        # M2 = (A10+A11)(B00): contributes to C10 and C11
        # U: rows 2(A10) and 3(A11), V: row 0(B00), W: rows 1(C10) and 3(C11)
        U2[2,1] = 1; U2[3,1] = 1
        V2[0,1] = 1
        W2[1,1] = 1; W2[3,1] = -1
        
        # M3 = (A00)(B01-B11): contributes to C01 and C11
        # U: row 0(A00), V: rows 1(B01) and 3(B11), W: rows 2(C01) and 3(C11)
        U2[0,2] = 1
        V2[1,2] = 1; V2[3,2] = -1
        W2[2,2] = 1; W2[3,2] = 1
        
        # M4 = (A11)(B10-B00): contributes to C00 and C10
        # U: row 3(A11), V: rows 2(B10) and 0(B00), W: rows 0(C00) and 1(C10)
        U2[3,3] = 1
        V2[2,3] = 1; V2[0,3] = -1
        W2[0,3] = 1; W2[1,3] = 1
        
        # M5 = (A00+A01)(B11): contributes to C00 and C01
        # U: rows 0(A00) and 1(A01), V: row 3(B11), W: rows 0(C00) and 2(C01)
        U2[0,4] = 1; U2[1,4] = 1
        V2[3,4] = 1
        W2[0,4] = -1; W2[2,4] = 1
        
        # M6 = (A10-A00)(B00+B01): contributes to C11
        # U: rows 2(A10) and 0(A00), V: rows 0(B00) and 1(B01), W: row 3(C11)
        U2[2,5] = 1; U2[0,5] = -1
        V2[0,5] = 1; V2[1,5] = 1
        W2[3,5] = 1
        
        # M7 = (A01-A11)(B10+B11): contributes to C00
        # U: rows 1(A01) and 3(A11), V: rows 2(B10) and 3(B11), W: row 0(C00)
        U2[1,6] = 1; U2[3,6] = -1
        V2[2,6] = 1; V2[3,6] = 1
        W2[0,6] = 1
        
        return U2, V2, W2
    
    # Verify Strassen 2x2 first
    U2, V2, W2 = strassen_2x2()
    if not verify(U2.tolist(), V2.tolist(), W2.tolist()):
        # Fallback: naive 2x2
        U2n = np.zeros((4, 4), dtype=float)
        V2n = np.zeros((4, 4), dtype=float)
        W2n = np.zeros((4, 4), dtype=float)
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    r = i*4 + k*2 + j
                    U2n[i*2+k, r] = 1
                    V2n[k*2+j, r] = 1
                    W2n[j*2+i, r] = 1
        if not verify(U2n.tolist(), V2n.tolist(), W2n.tolist()):
            # Should not happen, but return naive 4x4 as last resort
            rank64 = 64
            U64 = np.zeros((16, 64), dtype=float)
            V64 = np.zeros((16, 64), dtype=float)
            W64 = np.zeros((16, 64), dtype=float)
            for i in range(4):
                for k in range(4):
                    for j in range(4):
                        r = i*16 + k*4 + j
                        U64[i*4+k, r] = 1
                        V64[k*4+j, r] = 1
                        W64[j*4+i, r] = 1
            return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}
    
    # Now build 4x4x4 via tensor product of two 2x2x2 Strassen
    # A[i,j] where i = I*2+ii, j = J*2+jj  (I,J: block, ii,jj: inner)
    # B[j,k] where j = J*2+jj, k = K*2+kk
    # C[i,j] where i = I*2+ii, j = J*2+jj, W row = j*4+i = (J*2+jj)*4+(I*2+ii)
    
    # The tensor product: for each pair of Strassen terms (s,t),
    # U_4[a_row, s*7+t] = U2[I*2+J, s] * U2[ii*2+jj, t]
    # V_4[b_row, s*7+t] = V2[J*2+K, s] * V2[jj*2+kk, t]
    # W_4[c_row, s*7+t] = W2[I*2+J, s] * W2[ii*2+jj, t]
    
    rank49 = 49
    U49 = np.zeros((16, 49), dtype=float)
    V49 = np.zeros((16, 49), dtype=float)
    W49 = np.zeros((16, 49), dtype=float)
    
    # Build U: A[i,k] where i=I*2+ii, k=K*2+kk... wait
    # Actually for matrix mult C=A*B: A is n x m = 4x4, B is m x p = 4x4
    # A[i,k]: i in 0..3, k in 0..3. i = I*2+ii, k = K*2+kk
    # But in the block structure: A_block[I,K], A_inner[ii,kk]
    # U2 block level: row I*2+K (for A_block[I,K])
    # U2 inner level: row ii*2+kk (for A_inner[ii,kk])
    
    for I in range(2):
        for K in range(2):
            for ii in range(2):
                for kk in range(2):
                    a_row = (I*2 + ii) * 4 + (K*2 + kk)  # i*4+k
                    for s in range(7):
                        for t in range(7):
                            r = s * 7 + t
                            coeff = U2[I*2+K, s] * U2[ii*2+kk, t]
                            if coeff != 0:
                                U49[a_row, r] = coeff
    
    # V: B[k,j] where k=K*2+kk, j=J*2+jj
    # V2 block level: row K*2+J (for B_block[K,J])
    # V2 inner level: row kk*2+jj (for B_inner[kk,jj])
    
    for K in range(2):
        for J in range(2):
            for kk in range(2):
                for jj in range(2):
                    b_row = (K*2 + kk) * 4 + (J*2 + jj)  # k*4+j
                    for s in range(7):
                        for t in range(7):
                            r = s * 7 + t
                            coeff = V2[K*2+J, s] * V2[kk*2+jj, t]
                            if coeff != 0:
                                V49[b_row, r] = coeff
    
    # W: C[i,j] where i=I*2+ii, j=J*2+jj, W row = j*4+i
    # W2 block level: row I*2+J (for C_block[I,J])
    # W2 inner level: row ii*2+jj (for C_inner[ii,jj])
    
    for I in range(2):
        for J in range(2):
            for ii in range(2):
                for jj in range(2):
                    c_row = (J*2 + jj) * 4 + (I*2 + ii)  # j*4+i
                    for s in range(7):
                        for t in range(7):
                            r = s * 7 + t
                            coeff = W2[I*2+J, s] * W2[ii*2+jj, t]
                            if coeff != 0:
                                W49[c_row, r] = coeff
    
    # Verify the 49-rank decomposition
    if verify(U49.tolist(), V49.tolist(), W49.tolist()):
        return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}
    
    # If verification failed (shouldn't happen), fall back to naive
    rank64 = 64
    U64 = np.zeros((16, 64), dtype=float)
    V64 = np.zeros((16, 64), dtype=float)
    W64 = np.zeros((16, 64), dtype=float)
    for i in range(4):
        for k in range(4):
            for j in range(4):
                r = i*16 + k*4 + j
                U64[i*4+k, r] = 1
                V64[k*4+j, r] = 1
                W64[j*4+i, r] = 1
    
    if verify(U64.tolist(), V64.tolist(), W64.tolist()):
        return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}
    
    # Last resort: return what we have
    return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}
