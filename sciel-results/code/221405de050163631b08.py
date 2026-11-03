import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) where:
    U[i*2+j, r] = coefficient for A[i,j]
    V[j*2+k, r] = coefficient for B[j,k]  
    W[k*2+i, r] = coefficient for C[k,i] (trace convention W[k*n+i] with n=2)
    """
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = [[0]*7 for _ in range(4)]  # 4 = 2*2
    V = [[0]*7 for _ in range(4)]  # 4 = 2*2
    W = [[0]*7 for _ in range(4)]  # 4 = 2*2 (k*2+i)
    
    # Index helpers: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    def u_idx(i, j): return i*2 + j
    def v_idx(j, k): return j*2 + k
    def w_idx(k, i): return k*2 + i
    
    # M1 = (A00+A11)*(B00+B11)
    U[u_idx(0,0)][0] += 1; U[u_idx(1,1)][0] += 1
    V[v_idx(0,0)][0] += 1; V[v_idx(1,1)][0] += 1
    # C00 += M1, C11 += M1
    W[w_idx(0,0)][0] += 1; W[w_idx(1,1)][0] += 1
    
    # M2 = (A10+A11)*B00
    U[u_idx(1,0)][1] += 1; U[u_idx(1,1)][1] += 1
    V[v_idx(0,0)][1] += 1
    # C10 += M2, C11 -= M2
    W[w_idx(0,1)][1] += 1; W[w_idx(1,1)][1] -= 1
    
    # M3 = A00*(B01-B11)
    U[u_idx(0,0)][2] += 1
    V[v_idx(0,1)][2] += 1; V[v_idx(1,1)][2] -= 1
    # C01 += M3, C11 += M3
    W[w_idx(1,0)][2] += 1; W[w_idx(1,1)][2] += 1
    
    # M4 = A11*(B10-B00)
    U[u_idx(1,1)][3] += 1
    V[v_idx(1,0)][3] += 1; V[v_idx(0,0)][3] -= 1
    # C00 += M4, C10 += M4
    W[w_idx(0,0)][3] += 1; W[w_idx(0,1)][3] += 1
    
    # M5 = (A00+A01)*B11
    U[u_idx(0,0)][4] += 1; U[u_idx(0,1)][4] += 1
    V[v_idx(1,1)][4] += 1
    # C00 -= M5, C01 += M5
    W[w_idx(0,0)][4] -= 1; W[w_idx(1,0)][4] += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[u_idx(1,0)][5] += 1; U[u_idx(0,0)][5] -= 1
    V[v_idx(0,0)][5] += 1; V[v_idx(0,1)][5] += 1
    # C11 += M6
    W[w_idx(1,1)][5] += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[u_idx(0,1)][6] += 1; U[u_idx(1,1)][6] -= 1
    V[v_idx(1,0)][6] += 1; V[v_idx(1,1)][6] += 1
    # C00 += M7
    W[w_idx(0,0)][6] += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Block decomposition strategy:
    # Split middle dimension (m=4) into two groups of 2: J0=[0,1], J1=[2,3]
    # Split first dimension (n=3) into: I0=[0,1], I1=[2]
    # Split last dimension (p=7) into: K0=[0,1], K1=[2,3], K2=[4,5,6]
    
    # For each combination of (I_block, J_block, K_block), we have a sub-problem
    # If all three have size 2, use Strassen (rank 7)
    # Otherwise use schoolbook
    
    # Let's define blocks:
    I_blocks = [(0, 2), (2, 3)]  # sizes 2, 1
    J_blocks = [(0, 2), (2, 4)]  # sizes 2, 2
    K_blocks = [(0, 2), (2, 4), (4, 7)]  # sizes 2, 2, 3
    
    U_list = []
    V_list = []
    W_list = []
    
    for (i0, i1) in I_blocks:
        for (j0, j1) in J_blocks:
            for (k0, k1) in K_blocks:
                ni = i1 - i0
                mj = j1 - j0
                pk = k1 - k0
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    sU, sV, sW = _strassen_2x2x2()
                    for r in range(7):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for ii in range(ni):
                            for jj in range(mj):
                                u_col[(i0+ii)*m + (j0+jj)] = sU[ii*2+jj][r]
                        for jj in range(mj):
                            for kk in range(pk):
                                v_col[(j0+jj)*p + (k0+kk)] = sV[jj*2+kk][r]
                        for kk in range(pk):
                            for ii in range(ni):
                                w_col[(k0+kk)*n + (i0+ii)] = sW[kk*2+ii][r]
                        U_list.append(u_col)
                        V_list.append(v_col)
                        W_list.append(w_col)
                else:
                    # Schoolbook for this block
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                u_col = [0.0] * (n * m)
                                v_col = [0.0] * (m * p)
                                w_col = [0.0] * (p * n)
                                u_col[(i0+ii)*m + (j0+jj)] = 1.0
                                v_col[(j0+jj)*p + (k0+kk)] = 1.0
                                w_col[(k0+kk)*n + (i0+ii)] = 1.0
                                U_list.append(u_col)
                                V_list.append(v_col)
                                W_list.append(w_col)
    
    rank = len(U_list)
    
    # Convert to integer/half-integer if possible
    U = U_list
    V = V_list
    W = W_list
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
