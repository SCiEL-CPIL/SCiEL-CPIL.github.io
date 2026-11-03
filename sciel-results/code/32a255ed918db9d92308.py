import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen factors for 2x2x2
    # U[i,j,s] = coefficient of A[i,j] in summand s
    # V[j,k,s] = coefficient of B[j,k] in summand s
    # W[k,i,s] = coefficient of C[k,i] in summand s
    #
    # Strassen's algorithm for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 + M7
    
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # s=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1  # C00 += 1*M1, C11 += 1*M1
    
    # s=1: M2 = (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = -1; W2[1, 1, 1] = -1  # C00 += -1*M2, C11 += -1*M2... wait
    # C00 = M1 - M4 + M5 + M7, so M2 doesn't appear in C00
    # C10 = M2 + M3, so C10 += 1*M2
    # C11 = M1 - M2 + M6 + M7, so C11 += -1*M2
    W2[0, 1, 1] = 0  # C01 doesn't use M2
    W2[1, 0, 1] = 1  # C10 += M2
    W2[1, 1, 1] = -1  # C11 += -M2
    
    # s=2: M3 = A00(B10-B11)
    U2[0, 0, 2] = 1
    V2[1, 0, 2] = 1; V2[1, 1, 2] = -1
    W2[0, 1, 2] = 1  # C01 += M3
    W2[1, 0, 2] = 1  # C10 += M3
    
    # s=3: M4 = A11(B01-B10)
    U2[1, 1, 3] = 1
    V2[0, 1, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = -1  # C00 += -M4
    W2[0, 1, 3] = 1  # C01 += M4
    
    # s=4: M5 = (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = 1  # C00 += M5
    
    # s=5: M6 = (A10-A00)(B00+B10)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[1, 0, 5] = 1
    W2[1, 1, 5] = 1  # C11 += M6
    
    # s=6: M7 = (A01-A11)(B01+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[0, 1, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1  # C00 += M7
    W2[1, 1, 6] = 1  # C11 += M7
    
    # Build 4x4x4 factors using Kronecker recursion
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]
    
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2*j1 + j0
                    k = 2*k1 + k0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]
    
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2*k1 + k0
                    i = 2*i1 + i0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]
    
    # Verify 4x4x4 decomposition
    # C = A * B, where A is 4x4, B is 4x4, C is 4x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = delta[i,k]... no, T[i,j,k] such that C[i,k] = sum_j A[i,j]B[j,k]
    # The multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] where for each rank-1 term, we have U[i,j,s]*V[j,k,s]*W[k,i,s] summing to give the coefficient
    # The correct tensor for 4x4*4x4 -> 4x4 is: for each (i,j,k), the coefficient is 1 if j is the shared index
    # T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
    # So the "true" tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that each (i,j,k) triple is valid)
    # Wait, let me think more carefully.
    
    # Matrix multiplication A(4x4) * B(4x4) = C(4x4)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor M has M[i,j,k] = 1 for all i,j,k in 0..3
    # (because each term A[i,j]*B[j,k] contributes to C[i,k])
    
    # The decomposition says: M[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
    # (using the W[k*n+i,r] convention where n=4, so W[k,i,s])
    
    # Let's verify
    M_true = np.ones((4, 4, 4))
    M_decomp = np.zeros((4, 4, 4))
    for s in range(49):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    M_decomp[i, j, k] += U4[i, j, s] * V4[j, k, s] * W4[k, i, s]
    
    if not np.allclose(M_true, M_decomp):
        # Fallback to identity decomposition (rank 64)
        rank = 64
        U = np.zeros((16, rank))
        V = np.zeros((32, rank))
        W = np.zeros((32, rank))
        for i in range(4):
            for j in range(4):
                s = i * 4 + j
                U[i*4 + j, s] = 1
                for k in range(4):
                    V[j*8 + k, s] = 1  # B[j,k]
                    W[k*4 + i, s] = 1  # C[i,k] -> W[k*4+i]
        # But we need 4x4x8, so we need two blocks
        # For 4x4x8: A(4x4)*B(4x8)=C(4x8)
        # C[i,k] = sum_j A[i,j]*B[j,k] for k=0..7
        # We can use two copies of the 4x4x4 decomposition
        rank = 128
        U = np.zeros((16, rank))
        V = np.zeros((32, rank))
        W = np.zeros((32, rank))
        # Block 1: k=0..3
        for s in range(64):
            for i in range(4):
                for j in range(4):
                    # Extract from identity
                    pass
        # Just use simple identity for both blocks
        for block in range(2):
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        s = block * 64 + i * 16 + j * 4 + k
                        U[i*4 + j, s] = 1
                        V[j*8 + block*4 + k, s] = 1
                        W[(block*4 + k)*4 + i, s] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 4x4x8 factors: two blocks of 4x4x4
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Block 1: k in 0..3
    for i in range(4):
        for j in range(4):
            U[i*4 + j, 0:49] = U4[i, j, :]
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j, k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k, i, :]
    
    # Block 2: k in 4..7
    for i in range(4):
        for j in range(4):
            U[i*4 + j, 49:98] = U4[i, j, :]
    for j in range(4):
        for kp in range(4):
            k = 4 + kp
            V[j*8 + k, 49:98] = V4[j, kp, :]
    for kp in range(4):
        k = 4 + kp
        for i in range(4):
            W[k*4 + i, 49:98] = W4[kp, i, :]
    
    # Verify the full 4x4x8 decomposition
    # C[i,k] = sum_j A[i,j]*B[j,k] for k=0..7
    # Tensor: T[i,j,k] = 1 for all i,j in 0..3, k in 0..7
    # Decomposition: T[i,j,k] = sum_s U[i,j,s]*V[j,k,s]*W[k,i,s]
    
    elapsed = time.time() - start_time
    if elapsed > 130:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Quick spot check
    np.random.seed(42)
    A_test = np.random.randint(-2, 3, size=(4, 4)).astype(float)
    B_test = np.random.randint(-2, 3, size=(4, 8)).astype(float)
    C_expected = A_test @ B_test
    
    C_decomp = np.zeros((4, 8))
    for s in range(rank):
        us = np.array([U[i*4+j, s] for i in range(4) for j in range(4)]).reshape(4, 4)
        vs = np.array([V[j*8+k, s] for j in range(4) for k in range(8)]).reshape(4, 8)
        ws = np.array([W[k*4+i, s] for k in range(8) for i in range(4)]).reshape(8, 4)
        # C[i,k] += sum_s us[i,j]*vs[j,k]*ws[k,i]... 
        # Actually the decomposition is: C[i,k] = sum_s (sum_j U[i,j,s]*A[i,j]) * V_part... 
        # No, the tensor decomposition means:
        # For each s: factor_s = U[:,s] * V[:,s] * W[:,s] as a rank-1 tensor
        # C[i,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s] summed over j... 
        # Wait, no. The standard form is:
        # T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
        # where T[i,j,k] = 1 for valid (i,j,k)
        # This means: the (i,j,k) coefficient in the tensor is sum_s U[i,j,s]*V[j,k,s]*W[k,i,s]
        # And we need this to equal 1 for all i,j,k (with k in 0..7)
        
        # So to compute C from A and B:
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # = sum_j sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] * A[i,j] * B[j,k]
        # = sum_s (sum_j A[i,j]*U[i,j,s]) * (sum_j B[j,k]*V[j,k,s]) * W[k,i,s]
        
        a_s = np.zeros((4,))
        for i in range(4):
            for j in range(4):
                a_s[i] += U[i*4+j, s] * A_test[i, j]
        b_s = np.zeros((8,))
        for k in range(8):
            for j in range(4):
                b_s[k] += V[j*8+k, s] * B_test[j, k]
        for i in range(4):
            for k in range(8):
                C_decomp[i, k] += a_s[i] * b_s[k] * W[k*4+i, s]
    
    if not np.allclose(C_expected, C_decomp, atol=1e-10):
        # Something wrong, return safe fallback
        rank_fb = 128
        U_fb = np.zeros((16, rank_fb))
        V_fb = np.zeros((32, rank_fb))
        W_fb = np.zeros((32, rank_fb))
        for block in range(2):
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        s = block * 64 + i * 16 + j * 4 + k
                        U_fb[i*4 + j, s] = 1
                        V_fb[j*8 + block*4 + k, s] = 1
                        W_fb[(block*4 + k)*4 + i, s] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
