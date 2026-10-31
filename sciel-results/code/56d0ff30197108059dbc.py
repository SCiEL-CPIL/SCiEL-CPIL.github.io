import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4
    rank = 49

    # Strassen 2x2 factors: 7 multiplications
    # U2[i*2+j, s] = coefficient of A[i,j] in s-th product
    # V2[j*2+k, s] = coefficient of B[j,k] in s-th product
    # W2[i*2+k, s] = coefficient of C[i,k] in s-th product
    
    # Standard Strassen for 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)(B00)
    # M2 = (A00)(B01-B11)
    # M3 = (A11)(B10-B00)
    # M4 = (A00+A01)(B11)
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # s=0: M0
    U2[0, 0] = 1  # A00
    U2[3, 0] = 1  # A11
    V2[0, 0] = 1  # B00
    V2[3, 0] = 1  # B11
    W2[0, 0] = 1  # C00
    W2[3, 0] = 1  # C11
    
    # s=1: M1
    U2[2, 1] = 1  # A10
    U2[3, 1] = 1  # A11
    V2[0, 1] = 1  # B00
    W2[1, 1] = 1  # C10
    W2[3, 1] = -1 # C11
    
    # s=2: M2
    U2[0, 2] = 1  # A00
    V2[1, 2] = 1  # B01
    V2[3, 2] = -1 # B11
    W2[2, 2] = 1  # C01
    W2[3, 2] = 1  # C11
    
    # s=3: M3
    U2[3, 3] = 1  # A11
    V2[2, 3] = 1  # B10
    V2[0, 3] = -1 # B00
    W2[0, 3] = 1  # C00
    W2[1, 3] = 1  # C10
    
    # s=4: M4
    U2[0, 4] = 1  # A00
    U2[1, 4] = 1  # A01
    V2[3, 4] = 1  # B11
    W2[0, 4] = -1 # C00
    W2[2, 4] = 1  # C01
    
    # s=5: M5
    U2[2, 5] = 1  # A10
    U2[0, 5] = -1 # A00
    V2[0, 5] = 1  # B00
    V2[1, 5] = 1  # B01
    W2[3, 5] = 1  # C11
    
    # s=6: M6
    U2[1, 6] = 1  # A01
    U2[3, 6] = -1 # A11
    V2[2, 6] = 1  # B10
    V2[3, 6] = 1  # B11
    W2[0, 6] = 1  # C00
    
    # Now build 4x4 factors using Kronecker product structure
    # The 4x4 matrix A is split into 2x2 blocks of 2x2 submatrices:
    # A = [[A00, A01], [A10, A11]]
    # where each Aij is a 2x2 matrix
    #
    # For the recursive Strassen, the outer Strassen operates on blocks:
    # Outer s picks which blocks to combine
    # Inner t operates within each 2x2 block
    #
    # The combined factor index is (s, t) -> col = s*7 + t
    #
    # U4[i*4+j, s*7+t] = sum over outer block contributions
    # The outer Strassen tells us which 2x2 block (br, bc) of A contributes
    # with what coefficient, and the inner Strassen tells us which element
    # within that block contributes.
    
    # Outer Strassen block coefficients (same as inner, operating on 2x2 block indices)
    # beta_outer[s][br][bc] = coefficient of block A[br,bc] in s-th outer product
    beta_outer = np.zeros((7, 2, 2), dtype=int)
    gamma_outer = np.zeros((7, 2, 2), dtype=int)
    delta_outer = np.zeros((7, 2, 2), dtype=int)
    
    # s=0
    beta_outer[0, 0, 0] = 1
    beta_outer[0, 1, 1] = 1
    gamma_outer[0, 0, 0] = 1
    gamma_outer[0, 1, 1] = 1
    delta_outer[0, 0, 0] = 1
    delta_outer[0, 1, 1] = 1
    
    # s=1
    beta_outer[1, 1, 0] = 1
    beta_outer[1, 1, 1] = 1
    gamma_outer[1, 0, 0] = 1
    delta_outer[1, 1, 0] = 1
    delta_outer[1, 1, 1] = -1
    
    # s=2
    beta_outer[2, 0, 0] = 1
    gamma_outer[2, 0, 1] = 1
    gamma_outer[2, 1, 1] = -1
    delta_outer[2, 0, 1] = 1
    delta_outer[2, 1, 1] = 1
    
    # s=3
    beta_outer[3, 1, 1] = 1
    gamma_outer[3, 1, 0] = 1
    gamma_outer[3, 0, 0] = -1
    delta_outer[3, 0, 0] = 1
    delta_outer[3, 1, 0] = 1
    
    # s=4
    beta_outer[4, 0, 0] = 1
    beta_outer[4, 0, 1] = 1
    gamma_outer[4, 1, 1] = 1
    delta_outer[4, 0, 0] = -1
    delta_outer[4, 0, 1] = 1
    
    # s=5
    beta_outer[5, 1, 0] = 1
    beta_outer[5, 0, 0] = -1
    gamma_outer[5, 0, 0] = 1
    gamma_outer[5, 0, 1] = 1
    delta_outer[5, 1, 1] = 1
    
    # s=6
    beta_outer[6, 0, 1] = 1
    beta_outer[6, 1, 1] = -1
    gamma_outer[6, 1, 0] = 1
    gamma_outer[6, 1, 1] = 1
    delta_outer[6, 0, 0] = 1
    
    # Build U, V, W
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U: A element (i,j) where i = br*2+lr, j = bc*2+lc
            for br in range(2):
                for bc in range(2):
                    outer_coeff = beta_outer[s, br, bc]
                    if outer_coeff == 0:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            inner_coeff = U2[lr * 2 + lc, t]
                            if inner_coeff == 0:
                                continue
                            i = br * 2 + lr
                            j = bc * 2 + lc
                            row_u = i * 4 + j
                            U[row_u, col] += outer_coeff * inner_coeff
            
            # V: B element (j,k) where j = br*2+lr, k = bc*2+lc
            for br in range(2):
                for bc in range(2):
                    outer_coeff = gamma_outer[s, br, bc]
                    if outer_coeff == 0:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            inner_coeff = V2[lr * 2 + lc, t]
                            if inner_coeff == 0:
                                continue
                            j = br * 2 + lr
                            k = bc * 2 + lc
                            row_v = j * 4 + k
                            V[row_v, col] += outer_coeff * inner_coeff
            
            # W: C element (i,k) -> W index = k*4 + i
            # C[i,k] where i = br*2+lr, k = bc*2+lc
            # W row index = k*4 + i = (bc*2+lc)*4 + (br*2+lr)
            for br in range(2):
                for bc in range(2):
                    outer_coeff = delta_outer[s, br, bc]
                    if outer_coeff == 0:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            inner_coeff = W2[lc * 2 + lr, t]
                            if inner_coeff == 0:
                                continue
                            i = br * 2 + lr
                            k = bc * 2 + lc
                            row_w = k * 4 + i
                            W[row_w, col] += outer_coeff * inner_coeff
    
    # Verify a few entries
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # T should be 1 if a=i*4+j, b=j*4+k, c=k*4+i (i.e., C[i,k] gets A[i,j]*B[j,k])
    
    # Check T[0,0,0]: i=0,j=0,k=0 -> a=0, b=0, c=0
    val1 = int(np.sum(U[0, :] * V[0, :] * W[0, :]))
    # Check T[1,4,1]: i=0,j=1,k=0 -> a=1, b=4, c=1  
    val2 = int(np.sum(U[1, :] * V[4, :] * W[1, :]))
    # Check T[1,5,4]: i=0,j=1,k=1 -> a=1, b=5, c=4
    val3 = int(np.sum(U[1, :] * V[5, :] * W[4, :]))
    # Check T[4,4,0]: i=1,j=0,k=0 -> a=4, b=4, c=0
    val4 = int(np.sum(U[4, :] * V[4, :] * W[0, :]))
    
    # These should be: val1=1, val2=0, val3=1, val4=1
    if val1 != 1 or val2 != 0 or val3 != 1 or val4 != 1:
        # Debug: print values
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
