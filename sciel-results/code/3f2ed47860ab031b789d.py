"""Tensor decomposition for 4x4x7 matrix multiplication using Kronecker-Strassen."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build Strassen decomposition for 2x2x2
    # T_{2,2,2}: C[i,k] = sum_j A[i,j] B[j,k]
    # U has shape (n*m, rank) = (4, 7)
    # V has shape (m*p, rank) = (4, 7)  
    # W has shape (p*n, rank) = (4, 7)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Strassen's 7 terms for 2x2x2:
    # A = [[a,b],[c,d]] indexed as A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B = [[e,f],[g,h]] indexed as B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C = [[c00,c01],[c10,c11]]
    
    # U[i*2+j, c]: which (i,j) of A
    # V[j*2+k, c]: which (j,k) of B
    # W[k*2+i, c]: which (k,i) of C (output index: W[k*n+i] with n=2)
    
    # Strassen:
    # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: contributes to C00 and C11
    #   C00 gets +1, C11 gets +1
    # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: C10 gets +1
    # M3 = a(f-h): U: a=1; V: f=1,h=-1; W: C01 gets +1
    # M4 = d(g-e): U: d=1; V: g=1,e=-1; W: C10 gets +1, C00 gets -1
    # M5 = (a+b)h: U: a=1,b=1; V: h=1; W: C00 gets +1, C01 gets +1
    # M6 = (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; W: C11 gets +1
    # M7 = (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: C11 gets -1, C00 gets -1
    
    # Let me build U, V, W for 2x2x2 Strassen
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i,j in {0,1} for A; j,k in {0,1} for B; k,i in {0,1} for C
    
    U2 = [[0]*7 for _ in range(4)]  # 4 = 2*2 entries
    V2 = [[0]*7 for _ in range(4)]  # 4 = 2*2 entries
    W2 = [[0]*7 for _ in range(4)]  # 4 = 2*2 entries
    
    # M1: (a+d)(e+h)
    # U: a(0,0)=1, d(1,1)=1
    U2[0][0] = 1  # i=0,j=0 (a)
    U2[3][0] = 1  # i=1,j=1 (d)
    # V: e(0,0)=1, h(1,1)=1
    V2[0][0] = 1  # j=0,k=0 (e)
    V2[3][0] = 1  # j=1,k=1 (h)
    # W: C00(k=0,i=0)=+1, C11(k=1,i=1)=+1
    W2[0][0] = 1  # k=0,i=0 (C00)
    W2[3][0] = 1  # k=1,i=1 (C11)
    
    # M2: (c+d)e
    # U: c(1,0)=1, d(1,1)=1
    U2[2][1] = 1  # i=1,j=0 (c)
    U2[3][1] = 1  # i=1,j=1 (d)
    # V: e(0,0)=1
    V2[0][1] = 1  # j=0,k=0 (e)
    # W: C10(k=0,i=1)=+1
    W2[1][1] = 1  # k=0,i=1 (C10)
    
    # M3: a(f-h)
    # U: a(0,0)=1
    U2[0][2] = 1  # i=0,j=0 (a)
    # V: f(0,1)=1, h(1,1)=-1
    V2[1][2] = 1  # j=0,k=1 (f)
    V2[3][2] = -1  # j=1,k=1 (h)
    # W: C01(k=1,i=0)=+1
    W2[2][2] = 1  # k=1,i=0 (C01)
    
    # M4: d(g-e)
    # U: d(1,1)=1
    U2[3][3] = 1  # i=1,j=1 (d)
    # V: g(1,0)=1, e(0,0)=-1
    V2[2][3] = 1  # j=1,k=0 (g)
    V2[0][3] = -1  # j=0,k=0 (e)
    # W: C10(k=0,i=1)=+1, C00(k=0,i=0)=-1
    W2[1][3] = 1  # k=0,i=1 (C10)
    W2[0][3] = -1  # k=0,i=0 (C00)
    
    # M5: (a+b)h
    # U: a(0,0)=1, b(0,1)=1
    U2[0][4] = 1  # i=0,j=0 (a)
    U2[1][4] = 1  # i=0,j=1 (b)
    # V: h(1,1)=1
    V2[3][4] = 1  # j=1,k=1 (h)
    # W: C00(k=0,i=0)=+1, C01(k=1,i=0)=+1
    W2[0][4] = 1  # k=0,i=0 (C00)
    W2[2][4] = 1  # k=1,i=0 (C01)
    
    # M6: (c-a)(e+f)
    # U: c(1,0)=1, a(0,0)=-1
    U2[2][5] = 1  # i=1,j=0 (c)
    U2[0][5] = -1  # i=0,j=0 (a)
    # V: e(0,0)=1, f(0,1)=1
    V2[0][5] = 1  # j=0,k=0 (e)
    V2[1][5] = 1  # j=0,k=1 (f)
    # W: C11(k=1,i=1)=+1
    W2[3][5] = 1  # k=1,i=1 (C11)
    
    # M7: (b-d)(g+h)
    # U: b(0,1)=1, d(1,1)=-1
    U2[1][6] = 1  # i=0,j=1 (b)
    U2[3][6] = -1  # i=1,j=1 (d)
    # V: g(1,0)=1, h(1,1)=1
    V2[2][6] = 1  # j=1,k=0 (g)
    V2[3][6] = 1  # j=1,k=1 (h)
    # W: C11(k=1,i=1)=-1, C00(k=0,i=0)=-1
    W2[3][6] = -1  # k=1,i=1 (C11)
    W2[0][6] = -1  # k=0,i=0 (C00)
    
    # Verify Strassen decomposition for 2x2x2
    U2_arr = np.array(U2, dtype=float)
    V2_arr = np.array(V2, dtype=float)
    W2_arr = np.array(W2, dtype=float)
    
    # Check: for each (i,j,k), sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] should be 1
    # Actually the tensor decomposition means: T[i,j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    # For matrix multiplication, T[i,j,k] = 1 for all (i,j,k) in the "right" positions
    # Wait, the multiplication tensor has T[i,j,k] = delta_{i,j,k} in the sense that
    # C[i,k] = sum_j A[i,j] B[j,k], so the tensor is T[i,j,k] = 1 for all valid (i,j,k)
    # No, the tensor is: T[i,j,k] = 1 if the indices form a valid multiplication term
    # Actually for the multiplication tensor, T[i,j,k] = 1 for ALL (i,j,k) because
    # C[i,k] = sum_j A[i,j]*B[j,k], so every (i,j,k) triple contributes.
    
    # Let me verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all (i,j,k)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c] for c in range(7))
                if val != 1:
                    print(f"Strassen check failed at ({i},{j},{k}): {val}")
    
    # Now build 4x4x4 via Kronecker product of two Strassen decompositions
    # T_{4,4,4} = T_{2,2,2} ⊗ T_{2,2,2}
    # U4[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    # where i = i1*2 + i2, j = j1*2 + j2
    # V4[j*4+k, c1*7+c2] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
    # where j = j1*2 + j2, k = k1*2 + k2
    # W4[k*4+i, c1*7+c2] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    # where k = k1*2 + k2, i = i1*2 + i2
    
    rank_444 = 7 * 7  # 49
    U444 = [[0] * rank_444 for _ in range(16)]  # 4*4
    V444 = [[0] * rank_444 for _ in range(16)]  # 4*4
    W444 = [[0] * rank_444 for _ in range(16)]  # 4*4
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            uval = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
                            if uval != 0:
                                U444[i*4+j][c] = uval
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            vval = V2[j1*2+0][c1] * V2[j2*2+0][c2]  # k=0 for V indexing? No...
                            # V2[j*2+k, c]: j is the column of A / row of B
                            # For Kronecker: V4[j*4+k, c] where j = j1*2+j2, k = k1*2+k2
                            # V4[j*4+k, c1*7+c2] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
                            # So we need to iterate over k1, k2 as well
                            pass
    
    # Let me redo this more carefully
    U444 = [[0] * rank_444 for _ in range(16)]
    V444 = [[0] * rank_444 for _ in range(16)]
    W444 = [[0] * rank_444 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            uval = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
                            U444[i*4+j][c] = uval
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1 * 7 + c2
                            # V: need to sum over k1, k2... no, V is indexed by (j,k)
                            # V4[j*4+k, c] for fixed j, all k
                            pass
    
    # I'm overcomplicating. Let me use numpy for the Kronecker product.
    U2_arr = np.array(U2, dtype=float)  # shape (4, 7)
    V2_arr = np.array(V2, dtype=float)  # shape (4, 7)
    W2_arr = np.array(W2, dtype=float)  # shape (4, 7)
    
    # Kronecker product for the factor matrices
    # U4[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    # This is equivalent to: reshape U2 to (2,2,7), then for each c1,c2, the (i,j) entry is U2[i1,j1,c1]*U2[i2,j2,c2]
    # Which is the Kronecker product of U2 reshaped appropriately
    
    # U2 has shape (4,7) = (2*2, 7). Reshape to (2,2,7).
    # For Kronecker: U4[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    # where i = i1*2+i2, j = j1*2+j2, c = c1*7+c2
    # This is: for each (c1,c2), the 4x4 matrix is U2_reshaped[:, :, c1] ⊗ U2_reshaped[:, :, c2]
    
    U2_r = U2_arr.reshape(2, 2, 7)  # (i1, j1, c)
    V2_r = V2_arr.reshape(2, 2, 7)  # (j1, k1, c)
    W2_r = W2_arr.reshape(2, 2, 7)  # (k1, i1, c)
    
    U4_arr = np.zeros((16, 49), dtype=float)
    V4_arr = np.zeros((16, 49), dtype=float)
    W4_arr = np.zeros((16, 49), dtype=float)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U4[:, c] = U2_r[:,:,c1] (4x1 col) ⊗ U2_r[:,:,c2] (4x1 col)
            # U2_r[:,:,c] has shape (2,2), flatten to (4,) with index i1*2+i2, j1*2+j2? 
            # No: U2[i*2+j, c] where i = i1*2+i2... wait I need to be careful with indexing.
            
            # U2 is indexed as U2[i*2+j, c] where i in {0,1}, j in {0,1}
            # So U2_r[i1, j1, c] = U2[i1*2+j1, c]
            # For Kronecker: U4[i*4+j, c] where i = i1*2+i2, j = j1*2+j2
            # U4[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
            
            # U2_r[:, :, c1] is a 2x2 matrix with [i1, j1] -> U2[i1*2+j1, c1]
            # U2_r[:, :, c2] is a 2x2 matrix with [i2, j2] -> U2[i2*2+j2, c2]
            # The Kronecker product of these two 2x2 matrices gives a 4x4 matrix
            # with [i1*2+i2, j1*2+j2] = U2_r[i1,j1,c1] * U2_r[i2,j2,c2]
            # Wait, that's not standard Kronecker. Standard Kronecker A⊗B has
            # [i_A*nb + i_B, j_A*mb + j_B] = A[i_A,j_A] * B[i_B,j_B]
            # Here we want [i1*2+i2, j1*2+j2] = U2_r[i1,j1,c1] * U2_r[i2,j2,c2]
            # This IS the Kronecker product of U2_r[:,:,c1] and U2_r[:,:,c2]
            
            kron_U = np.kron(U2_r[:, :, c1], U2_r[:, :, c2])  # 4x4
            U4_arr[:, c] = kron_U.flatten()  # flatten in row-major: [i*4+j]
            
            # For V: V2[j*2+k, c] indexed by (j1, k1) for first, (j2, k2) for second
            # V4[j*4+k, c] where j = j1*2+j2, k = k1*2+k2
            # V4[j*4+k, c1*7+c2] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
            # V2_r[:, :, c] has shape (2,2) with [j, k] -> V2[j*2+k, c]
            kron_V = np.kron(V2_r[:, :, c1], V2_r[:, :, c2])  # 4x4
            V4_arr[:, c] = kron_V.flatten()
            
            # For W: W2[k*2+i, c] indexed by (k1, i1) for first, (k2, i2) for second
            # W4[k*4+i, c] where k = k1*2+k2, i = i1*2+i2
            # W4[k*4+i, c1*7+c2] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
            kron_W = np.kron(W2_r[:, :, c1], W2_r[:, :, c2])  # 4x4
            W4_arr[:, c] = kron_W.flatten()
    
    # Verify 4x4x4 decomposition
    if time.time() > deadline - 30:
        # Just return schoolbook as fallback
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
    
    # Now add schoolbook for the 4x4x3 part (k = 4,5,6)
    # Total rank = 49 + 4*4*3 = 49 + 48 = 97
    rank_total = 49 + 48
    U_total = [[0] * rank_total for _ in range(16)]
    V_total = [[0] * rank_total for _ in range(16)]
    W_total = [[0] * rank_total for _ in range(28)]  # p*n = 7*4 = 28
    
    # Copy the 4x4x4 part (k = 0,1,2,3)
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U_total[i*4+j][c] = U4_arr[i*4+j][c]
                V_total[j*4+0][c] = 0  # will fill below
                # V is indexed by (j, k) where k is 0..6 for the full problem
                # For the 4x4x4 part, k is 0..3
                pass
    
    # Let me be more careful. The full problem:
    # U[i*m+j, c] where m=4, so U[i*4+j, c], shape (16, rank)
    # V[j*p+k, c] where p=7, so V[j*7+k, c], shape (28, rank)
    # W[k*n+i, c] where n=4, so W[k*4+i, c], shape (28, rank)
    
    # The 4x4x4 part uses k=0,1,2,3. The 4x4x3 part uses k=4,5,6.
    
    # For the 4x4x4 part (columns 0..48):
    # U444[i*4+j, c] for i,j in 0..3, c in 0..48
    # V444[j*4+k, c] for j in 0..3, k in 0..3, c in 0..48
    #   But in the full problem, V is indexed by j*7+k, so V[j*7+k, c] for k=0..3
    # W444[k*4+i, c] for k in 0..3, i in 0..3, c in 0..48
    #   In the full problem, W is indexed by k*4+i, so W[k*4+i, c] for k=0..3
    
    U_total = [[0] * rank_total for _ in range(16)]
    V_total = [[0] * rank_total for _ in range(28)]
    W_total = [[0] * rank_total for _ in range(28)]
    
    # 4x4x4 part: columns 0 to 48
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U_total[i*4+j][c] = U4_arr[i*4+j][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V_total[j*7+k][c] = V4_arr[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W_total[k*4+i][c] = W4_arr[k*4+i][c]
    
    # 4x4x3 part: columns 49 to 96 (schoolbook for k=4,5,6)
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U_total[i*4+j][col] = 1
                V_total[j*7+k][col] = 1
                W_total[k*4+i][col] = 1
                col += 1
    
    # Verify the full decomposition
    # For each (i, j, k) with i in 0..3, j in 0..3, k in 0..6:
    # sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*4+i, c] should equal 1
    U_arr = np.array(U_total, dtype=float)
    V_arr = np.array(V_total, dtype=float)
    W_arr = np.array(W_total, dtype=float)
    
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                val = np.sum(U_arr[i*4+j] * V_arr[j*7+k] * W_arr[k*4+i])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        print(f"Verification failed, max error: {max_err}")
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
    
    # Convert to lists of integers (entries are -1, 0, 1)
    U_list = [[int(round(x)) for x in row] for row in U_total]
    V_list = [[int(round(x)) for x in row] for row in V_total]
    W_list = [[int(round(x)) for x in row] for row in W_total]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
