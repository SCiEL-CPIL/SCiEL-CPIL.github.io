import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    # 1. Rank-7 Strassen decomposition for 2x2 matrix multiplication
    # U, V, W are 4x7 matrices
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # Term 1: P1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1  # C00
    W2[3, 0] = 1  # C11

    # Term 2: P2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1  # C10
    W2[3, 1] = -1 # C11

    # Term 3: P3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1  # C01
    W2[3, 2] = 1  # C11

    # Term 4: P4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1  # C00
    W2[2, 3] = 1  # C10

    # Term 5: P5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C00
    W2[1, 4] = 1  # C01

    # Term 6: P6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1  # C11

    # Term 7: P7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1  # C00

    # Verify 2x2 Strassen decomposition
    # T2[i,j,k] = sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
    # Should be 1 for all i,j,k
    T2 = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    T2[i, j, k] += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
    if not np.allclose(T2, np.ones((2,2,2)), atol=1e-9):
        # Fallback to schoolbook for 2x2
        U2 = np.zeros((4, 4))
        V2 = np.zeros((4, 4))
        W2 = np.zeros((4, 4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = i*4 + j*2 + k
                    U2[i*2+j, r] = 1
                    V2[j*2+k, r] = 1
                    W2[k*2+i, r] = 1
        # Re-verify
        T2 = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for r in range(4):
                        T2[i, j, k] += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]

    # 2. Schoolbook rank-27 decomposition for 3x3 matrix multiplication
    # U3, V3, W3 are 9x27 matrices
    # Term t corresponds to (i_t, j_t, k_t) where t = i_t*9 + j_t*3 + k_t
    # U3[i,j,t] = 1 if i==i_t and j==j_t
    # V3[j,k,t] = 1 if j==j_t and k==k_t
    # W3[k,i,t] = 1 if k==k_t and i==i_t
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                t = i*9 + j*3 + k
                U3[i*3+j, t] = 1
                V3[j*3+k, t] = 1
                W3[k*3+i, t] = 1

    # Verify 3x3 schoolbook decomposition
    T3 = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(27):
                    T3[i, j, k] += U3[i*3+j, t] * V3[j*3+k, t] * W3[k*3+i, t]
    if not np.allclose(T3, np.ones((3,3,3)), atol=1e-9):
        # This shouldn't happen for schoolbook, but just in case
        pass

    # 3. Construct 6x6 decomposition
    # Partition 6x6 into 2x2 blocks of 3x3 submatrices
    # Global index: i = 3*i2 + i3, j = 3*j2 + j3, k = 3*k2 + k3
    # where i2,j2,k2 in {0,1} (2x2 block indices) and i3,j3,k3 in {0,1,2} (3x3 submatrix indices)
    
    # For each Strassen term s (0..6) and each schoolbook term t (0..26):
    # r = s*27 + t
    # U6[i,j,r] = U2[i2,j2,s] * U3[i3,j3,t]
    # V6[j,k,r] = V2[j2,k2,s] * V3[j3,k3,t]
    # W6[k,i,r] = W2[k2,i2,s] * W3[k3,i3,t]
    
    # The rank is 7 * 27 = 189
    total_rank = 7 * 27
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    for s in range(7):
        for t in range(27):
            r = s * 27 + t
            
            # Get the 2x2 matrices for this Strassen term
            u2_mat = U2[:, s].reshape(2, 2)
            v2_mat = V2[:, s].reshape(2, 2)
            w2_mat = W2[:, s].reshape(2, 2)
            
            # Get the 3x3 matrices for this schoolbook term
            u3_mat = U3[:, t].reshape(3, 3)
            v3_mat = V3[:, t].reshape(3, 3)
            w3_mat = W3[:, t].reshape(3, 3)
            
            # Compute Kronecker products
            # U6[:, r] is the Kronecker product of u3_mat and u2_mat
            # (A ⊗ B)_{i,j} where i=3*i2+i3, j=3*j2+j3 is A[i3,j3] * B[i2,j2]
            u6_mat = np.kron(u3_mat, u2_mat)
            v6_mat = np.kron(v3_mat, v2_mat)
            w6_mat = np.kron(w3_mat, w2_mat)
            
            U6[:, r] = u6_mat.flatten()
            V6[:, r] = v6_mat.flatten()
            W6[:, r] = w6_mat.flatten()

    # Verify the 6x6 decomposition
    if time.time() < deadline:
        try:
            U3g = U6.reshape(6, 6, -1)
            V3g = V6.reshape(6, 6, -1)
            W3g = W6.reshape(6, 6, -1)
            
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            target = np.ones((6, 6, 6))
            if np.allclose(reconstructed, target, atol=1e-9):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        except Exception as e:
            pass

    # Fallback to schoolbook if verification failed
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                idx = i*6 + j
                jk = j*6 + k
                ki = k*6 + i
                col = i*36 + j*6 + k
                U_f[idx, col] = 1.0
                V_f[jk, col] = 1.0
                W_f[ki, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
