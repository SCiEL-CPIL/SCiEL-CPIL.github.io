import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    def try_strassen_decomp():
        total_rank = 56
        U_mat = np.zeros((n*m, total_rank))
        V_mat = np.zeros((m*p, total_rank))
        W_mat = np.zeros((p*n, total_rank))
        
        col = 0
        
        for m_block in range(2):
            j_off = m_block * 2
            for n_part in range(2):
                if n_part == 0:
                    i_off = 0
                    n_size = 2
                else:
                    i_off = 2
                    n_size = 1
                
                if n_size == 2:
                    for p_block in range(3):
                        if p_block < 2:
                            k_off = p_block * 2
                            p_size = 2
                        else:
                            k_off = 4
                            p_size = 1
                        
                        if p_size == 2:
                            i0, i1 = i_off, i_off + 1
                            j0, j1 = j_off, j_off + 1
                            k0, k1 = k_off, k_off + 1
                            
                            # M1 = (A00+A11)(B00+B11)
                            U_mat[i0*m+j0, col] = 1
                            U_mat[i1*m+j1, col] = 1
                            V_mat[j0*p+k0, col] = 1
                            V_mat[j1*p+k1, col] = 1
                            W_mat[k0*n+i0, col] = 1
                            W_mat[k1*n+i1, col] = 1
                            col += 1
                            
                            # M2 = (A10+A11)B00
                            U_mat[i1*m+j0, col] = 1
                            U_mat[i1*m+j1, col] = 1
                            V_mat[j0*p+k0, col] = 1
                            W_mat[k0*n+i1, col] = 1
                            col += 1
                            
                            # M3 = A00(B10-B11)
                            U_mat[i0*m+j0, col] = 1
                            V_mat[j1*p+k0, col] = 1
                            V_mat[j1*p+k1, col] = -1
                            W_mat[k0*n+i0, col] = 1
                            W_mat[k1*n+i0, col] = -1
                            col += 1
                            
                            # M4 = A11(B00-B10)
                            U_mat[i1*m+j1, col] = 1
                            V_mat[j0*p+k0, col] = 1
                            V_mat[j0*p+k1, col] = -1
                            W_mat[k0*n+i1, col] = 1
                            W_mat[k1*n+i1, col] = -1
                            col += 1
                            
                            # M5 = (A00+A01)B11
                            U_mat[i0*m+j0, col] = 1
                            U_mat[i0*m+j1, col] = 1
                            V_mat[j1*p+k1, col] = 1
                            W_mat[k1*n+i0, col] = 1
                            col += 1
                            
                            # M6 = (A10-A00)(B00+B01)
                            U_mat[i1*m+j0, col] = 1
                            U_mat[i0*m+j0, col] = -1
                            V_mat[j0*p+k0, col] = 1
                            V_mat[j0*p+k1, col] = 1
                            W_mat[k0*n+i1, col] = 1
                            W_mat[k1*n+i1, col] = 1
                            col += 1
                            
                            # M7 = (A01-A11)(B10+B11)
                            U_mat[i0*m+j1, col] = 1
                            U_mat[i1*m+j1, col] = -1
                            V_mat[j1*p+k0, col] = 1
                            V_mat[j1*p+k1, col] = 1
                            W_mat[k0*n+i0, col] = 1
                            W_mat[k1*n+i0, col] = 1
                            col += 1
                        else:
                            for ii in range(2):
                                for jj in range(2):
                                    U_mat[(i_off+ii)*m + (j_off+jj), col] = 1
                                    V_mat[(j_off+jj)*p + k_off, col] = 1
                                    W_mat[k_off*n + (i_off+ii), col] = 1
                                    col += 1
                else:
                    for jj in range(2):
                        for kk in range(5):
                            U_mat[i_off*m + (j_off+jj), col] = 1
                            V_mat[(j_off+jj)*p + kk, col] = 1
                            W_mat[kk*n + i_off, col] = 1
                            col += 1
        
        # Verify tensor identity
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(total_rank):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        return None
        return U_mat, V_mat, W_mat
    
    result = try_strassen_decomp()
    if result is not None:
        U_mat, V_mat, W_mat = result
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
