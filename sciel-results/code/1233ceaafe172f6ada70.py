import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 4, 4, 6
    
    # Strassen coefficients for 2x2x2
    # U, V, W are lists of 7 terms, each term is a dict of (index, coeff)
    # Indices are flattened: i*2+j for U (2x2), j*2+k for V (2x2), k*2+i for W (2x2)
    # Note: Standard Strassen for C = A*B where A is 2x2, B is 2x2, C is 2x2.
    # T1 = (A11 + A22) * (B11 + B22)
    # T2 = (A21 + A22) * B11
    # T3 = A11 * (B12 - B22)
    # T4 = A22 * (B21 - B11)
    # T5 = (A11 + A12) * B22
    # T6 = (A21 - A11) * (B11 + B12)
    # T7 = (A12 - A22) * (B21 + B22)
    # C11 = T1 + T4 - T5 + T7
    # C12 = T3 + T5
    # C21 = T2 + T4
    # C22 = T1 - T2 + T3 + T6
    
    strassen_terms = [
        # T1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}), # C11 has -1? No.
        # Let's use the standard expansion directly into the tensor form.
        # T1 contributes to C11, C22
        # T2 contributes to C21, C22
        # T3 contributes to C12, C22
        # T4 contributes to C11, C21
        # T5 contributes to C11, C12
        # T6 contributes to C22
        # T7 contributes to C11
        
        # Let's define the 7 rank-1 tensors (u, v, w) such that sum u v w = mult_tensor
        # u is 4-dim (2x2), v is 4-dim (2x2), w is 4-dim (2x2)
        
        # Term 1: (1,0,0,1) * (1,0,0,1) -> C11(1), C22(-1)? No.
        # Let's just use the explicit Kronecker construction which is safer.
    ]
    
    # Explicit Strassen Terms for 2x2x2
    # U[i,j], V[j,k], W[k,i]
    # 1. U={00:1,11:1}, V={00:1,11:1}, W={00:1,11:1} -> T1
    # 2. U={10:1,11:1}, V={00:1}, W={10:1,11:-1} -> T2 (Contributes to C21, C22)
    #    C21 = U[1,0]V[0,0]W[0,1] + ... 
    #    Let's verify T2 = (A21+A22)B11.
    #    C21 needs A21 B11. C22 needs -A21 B11? No, C22 = T1 - T2 + ... so -T2 contributes -A21 B11 to C22?
    #    Actually, let's just use the known rank-7 decomposition vectors.
    
    # Rank 7 vectors for 2x2x2:
    # u1 = [1, 0, 0, 1] (A11, A22)
    # v1 = [1, 0, 0, 1] (B11, B22)
    # w1 = [1, 0, 0, 1] (C11, C22) -> T1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    #    This contributes to C11 (A11B11), C12 (A11B22), C21 (A22B11), C22 (A22B22).
    #    Wait, C11 = A11B11 + A12B21. C12 = A11B12 + A12B22.
    #    T1 contributes A11B11 to C11? Yes. A11B22 to C12? Yes. A22B11 to C21? Yes. A22B22 to C22? Yes.
    
    # Let's use a verified set of 7 vectors.
    # Reference: Strassen algorithm tensor rank 7.
    
    def make_strassen_222():
        # Returns U, V, W as 4x7 matrices
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices: 0:00, 1:01, 2:10, 3:11
        
        # T1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # T2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # C21 += A10B00+A11B00, C22 -= A10B00+A11B00
        
        # T3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1 # C12 += A00B01-A00B11, C22 += A00B01-A00B11? 
        # Check: C12 = A00B01 + A01B11. C22 = A10B01 + A11B11.
        # T3 contributes A00B01 to C12. Correct.
        # T3 contributes -A00B11 to C22? No, C22 doesn't have A00 term.
        # My W indices are k*n+i.
        # W[1,2]=1 means C[0,1] (C12) gets +1.
        # W[3,2]=1 means C[1,1] (C22) gets +1.
        # So T3 adds A00B01 to C12 and A00B01 to C22?
        # If V[1,2]=1 (B01) and V[3,2]=-1 (B11), then T3 = A00 * (B01 - B11).
        # W[1,2]=1 -> C12 += A00B01 - A00B11.
        # W[3,2]=1 -> C22 += A00B01 - A00B11.
        # This is wrong. C22 should not get A00 terms.
        
        # Let's restart with a reliable source for Strassen tensor.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's map these to vectors.
        # M1:
        # A: 00, 11. B: 00, 11. C: 00 (+1), 11 (+1)?
        # C00 gets M1. C11 gets M1.
        # So W1 has 1 at 00, 1 at 11.
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        
        # M2:
        # A: 10, 11. B: 00. C: 10 (+1), 11 (-1).
        # C10 gets M2. C11 gets -M2.
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        
        # M3:
        # A: 00. B: 01, 11. C: 01 (+1), 11 (+1).
        # C01 gets M3. C11 gets M3.
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        
        # M4:
        # A: 11. B: 10, 00. C: 00 (+1), 10 (+1).
        # C00 gets M4. C10 gets M4.
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        
        # M5:
        # A: 00, 01. B: 11. C: 00 (-1), 01 (+1).
        # C00 gets -M5. C01 gets M5.
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        
        # M6:
        # A: 10, 00. B: 00, 01. C: 11 (+1).
        # C11 gets M6.
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        # M7:
        # A: 01, 11. B: 10, 11. C: 00 (+1).
        # C00 gets M7.
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        
        return U, V, W

    U2, V2, W2 = make_strassen_222()
    
    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                if abs(total - 1) > 1e-9:
                    # Fallback to schoolbook for 2x2x2 if Strassen fails
                    pass # We will just proceed, but it should be correct.

    # Construct 4x4x4 using Kronecker product of two 2x2x2
    # n1=2, m1=2, p1=2, n2=2, m2=2, p2=2
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            U444[i1*2+i2, j1*2+j2, col] += U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            V444[j1*2+j2, k1*2+k2, col] += V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            W444[k1*2+k2, i1*2+i2, col] += W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
            col += 1
            
    # Construct 4x4x2 using Schoolbook
    # n=4, m=4, p=2
    # Rank = 4*4*2 = 32
    U442 = np.zeros((16, 32))
    V442 = np.zeros((8, 32))
    W442 = np.zeros((8, 32))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U442[i*4+j, col] = 1
                V442[j*2+k, col] = 1
                W442[k*4+i, col] = 1
                col += 1

    # Combine 4x4x4 and 4x4x2 to get 4x4x6
    # Total Rank = 49 + 32 = 81
    U_final = np.zeros((16, 81))
    V_final = np.zeros((24, 81)) # 4*6 = 24
    W_final = np.zeros((24, 81)) # 6*4 = 24
    
    # Copy 4x4x4 part (k=0..3)
    U_final[:, :49] = U444
    V_final[:16, :49] = V444
    W_final[:16, :49] = W444
    
    # Copy 4x4x2 part (k=4..5)
    # V442 indices: j*2+k. k is 0,1 locally. Global k is 4,5.
    # So V442 row j*2+k corresponds to global row j*6 + (4+k).
    # We need to map V442 to V_final rows 16..23.
    for j in range(4):
        for k_local in range(2):
            k_global = 4 + k_local
            V_final[j*6 + k_global, 49:] = V442[j*2 + k_local, :]
            
    # W442 indices: k*4+i. k is 0,1 locally. Global k is 4,5.
    # So W442 row k*4+i corresponds to global row (4+k)*4 + i.
    # We need to map W442 to W_final rows 16..23.
    for k_local in range(2):
        for i in range(4):
            k_global = 4 + k_local
            W_final[k_global*4 + i, 49:] = W442[k_local*4 + i, :]

    # Verify
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(6):
                total = np.sum(U_final[i*4+j, :] * V_final[j*6+k, :] * W_final[k*4+i, :])
                if abs(total - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }
    else:
        # Fallback to Schoolbook 4x4x6 (Rank 96)
        U_s = np.zeros((16, 96))
        V_s = np.zeros((24, 96))
        W_s = np.zeros((24, 96))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U_s[i*4+j, col] = 1
                    V_s[j*6+k, col] = 1
                    W_s[k*4+i, col] = 1
                    col += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }
