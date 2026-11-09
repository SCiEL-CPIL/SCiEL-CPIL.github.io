import time
import numpy as np

DIMENSIONS = (6, 6, 6)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Kronecker product decomposition:
    # 6x6 = (3x3) ⊗ (2x2)
    # T_6 = T_3 ⊗ T_2
    # rank(T_6) <= rank(T_3) * rank(T_2)
    # 
    # Use schoolbook for 3x3: rank = 27
    # Use Strassen for 2x2: rank = 7
    # Total rank = 27 * 7 = 189
    #
    # The tensor T for matrix multiplication A(n x m) * B(m x p) -> C(p x n)
    # T[A_idx, B_idx, C_idx] = 1 if A_idx=(i,k), B_idx=(k,j), C_idx=(j,i)
    # In the official trace convention: W[k*n+i, r]
    #
    # For Kronecker product:
    # A has indices (a3, a2) where a3 in [0,8], a2 in [0,3]
    # B has indices (b3, b2) where b3 in [0,8], b2 in [0,3]
    # C has indices (c3, c2) where c3 in [0,8], c2 in [0,3]
    #
    # A_idx global = a3 * 2 + a2 (row-major in 6x6, but we need to map properly)
    # Actually, for 6x6 matrix A, entry A[i,j] has i in [0,5], j in [0,5]
    # i = i3 * 2 + i2, j = j3 * 2 + j2
    # So A_idx (flattened) = i * 6 + j = (i3*2+i2)*6 + (j3*2+j2)
    #
    # Similarly for B: B[k,j] with k = k3*2+k2, j = j3*2+j2
    # B_idx = k * 6 + j = (k3*2+k2)*6 + (j3*2+j2)
    #
    # For C: C[j,i] with j = j3*2+j2, i = i3*2+i2
    # C_idx (in trace convention W[k*n+i]) where k is the row index in C... 
    # Wait, the problem says W[k*n+i, r]. Let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This means W has shape (p*n, rank), and the row index is k*n + i.
    # In our case p=n=6, so row index is k*6 + i where k in [0,5], i in [0,5].
    # This corresponds to C[k, i] if C is stored row-major? Or is it a specific convention?
    # 
    # The tensor decomposition says: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    # where a indexes A, b indexes B, c indexes C.
    # The "official trace convention" W[k*n+i] likely means that the index c = k*n + i
    # corresponds to the entry C[i, k] or C[k, i]. 
    #
    # For matrix multiplication C = A * B, C[i, j] = sum_k A[i, k] B[k, j].
    # So the tensor T[A(i,k), B(k,j), C(i,j)] = 1.
    # If W uses convention k*n + i, and C[i,j] is what we compute, then 
    # the index in W for C[i,j] would be j*n + i (since k=j, i=i).
    # So W[j*6 + i, r] corresponds to C[i, j].
    
    # Let's define the 2x2 Strassen tensors U2, V2, W2
    # For 2x2: A is 2x2, B is 2x2, C is 2x2
    # U2 shape (4, 7), V2 shape (4, 7), W2 shape (4, 7)
    # A[i,k] flattened: i*2+k
    # B[k,j] flattened: k*2+j
    # C[i,j] in trace convention: j*2+i
    
    U2 = np.zeros((4, 7), dtype=np.int64)
    V2 = np.zeros((4, 7), dtype=np.int64)
    W2 = np.zeros((4, 7), dtype=np.int64)
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M2 + M3
    # C10 = M4 + M5
    # C11 = M1 - M2 + M3 + M5
    
    # Term 0: M1
    U2[0, 0] = 1; U2[3, 0] = 1  # A00, A11
    V2[0, 0] = 1; V2[3, 0] = 1  # B00, B11
    W2[0, 0] = 1; W2[3, 0] = 1  # C00 (0*2+0), C11 (1*2+1)
    
    # Term 1: M2
    U2[2, 1] = 1; U2[3, 1] = 1  # A10, A11
    V2[0, 1] = 1                # B00
    W2[1, 1] = 1; W2[3, 1] = -1 # C01 (1*2+0), C11 (1*2+1)
    
    # Term 2: M3
    U2[0, 2] = 1                # A00
    V2[1, 2] = 1; V2[3, 2] = -1 # B01, B11
    W2[1, 2] = 1; W2[3, 2] = 1  # C01, C11
    
    # Term 3: M4
    U2[3, 3] = 1                # A11
    V2[2, 3] = 1; V2[0, 3] = -1 # B10, B00
    W2[0, 3] = -1; W2[2, 3] = 1 # C00, C10
    
    # Term 4: M5
    U2[0, 4] = 1; U2[1, 4] = 1  # A00, A01
    V2[3, 4] = 1                # B11
    W2[2, 4] = 1; W2[3, 4] = 1  # C10, C11
    
    # Term 5: M6
    U2[2, 5] = 1; U2[0, 5] = -1 # A10, A00
    V2[0, 5] = 1; V2[1, 5] = 1  # B00, B01
    W2[0, 5] = 1                # C00
    
    # Term 6: M7
    U2[1, 6] = 1; U2[3, 6] = -1 # A01, A11
    V2[2, 6] = 1; V2[3, 6] = 1  # B10, B11
    W2[0, 6] = 1                # C00
    
    # Define 3x3 schoolbook tensors U3, V3, W3
    # For 3x3: A is 3x3, B is 3x3, C is 3x3
    # U3 shape (9, 27), V3 shape (9, 27), W3 shape (9, 27)
    # A[i,k] flattened: i*3+k
    # B[k,j] flattened: k*3+j
    # C[i,j] in trace convention: j*3+i
    
    U3 = np.zeros((9, 27), dtype=np.int64)
    V3 = np.zeros((9, 27), dtype=np.int64)
    W3 = np.zeros((9, 27), dtype=np.int64)
    
    for i in range(3):
        for k in range(3):
            for j in range(3):
                r = i * 9 + k * 3 + j
                U3[i * 3 + k, r] = 1
                V3[k * 3 + j, r] = 1
                W3[j * 3 + i, r] = 1
    
    # Now construct 6x6 tensors via Kronecker product
    # rank = 27 * 7 = 189
    rank = 189
    U = np.zeros((36, rank), dtype=np.int64)
    V = np.zeros((36, rank), dtype=np.int64)
    W = np.zeros((36, rank), dtype=np.int64)
    
    for r3 in range(27):
        for r2 in range(7):
            col = r3 * 7 + r2
            
            # U: A6[i,j] with i=i3*2+i2, j=j3*2+j2
            # A6_idx = i*6 + j = (i3*2+i2)*6 + (j3*2+j2)
            # A3_idx = i3*3 + j3
            # A2_idx = i2*2 + j2
            for i3 in range(3):
                for j3 in range(3):
                    a3_idx = i3 * 3 + j3
                    for i2 in range(2):
                        for j2 in range(2):
                            a2_idx = i2 * 2 + j2
                            a6_idx = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                            U[a6_idx, col] = U3[a3_idx, r3] * U2[a2_idx, r2]
            
            # V: B6[k,j] with k=k3*2+k2, j=j3*2+j2
            # B6_idx = k*6 + j = (k3*2+k2)*6 + (j3*2+j2)
            # B3_idx = k3*3 + j3
            # B2_idx = k2*2 + j2
            for k3 in range(3):
                for j3 in range(3):
                    b3_idx = k3 * 3 + j3
                    for k2 in range(2):
                        for j2 in range(2):
                            b2_idx = k2 * 2 + j2
                            b6_idx = (k3 * 2 + k2) * 6 + (j3 * 2 + j2)
                            V[b6_idx, col] = V3[b3_idx, r3] * V2[b2_idx, r2]
            
            # W: C6[i,j] with i=i3*2+i2, j=j3*2+j2
            # W_idx = j*6 + i (trace convention: k*n+i where k=j, n=6)
            # W3_idx = j3*3 + i3
            # W2_idx = j2*2 + i2
            for i3 in range(3):
                for j3 in range(3):
                    w3_idx = j3 * 3 + i3
                    for i2 in range(2):
                        for j2 in range(2):
                            w2_idx = j2 * 2 + i2
                            w6_idx = (j3 * 2 + j2) * 6 + (i3 * 2 + i2)
                            W[w6_idx, col] = W3[w3_idx, r3] * W2[w2_idx, r2]
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
