import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 8

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build the rank-7 Strassen decomposition for 2x2x2.
    # Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # The decomposition satisfies: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}
    #
    # Strassen's 7 products:
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

    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M1: A00+A11, B00+B11 -> C00+=1, C11+=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M2: A10+A11, B00 -> C10+=1, C11-=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M3: A00, B01-B11 -> C01+=1, C11+=1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M4: A11, B10-B00 -> C00+=1, C10+=1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M5: A00+A01, B11 -> C00-=1, C01+=1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M6: A10-A00, B00+B01 -> C11+=1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M7: A01-A11, B10+B11 -> C00+=1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Verify the 2x2x2 Strassen decomposition
    def verify_222(U2, V2, W2):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U2[i*2+j, t] * V2[j*2+k, t] * W2[k*2+i, t] for t in range(7))
                    if s != 1:
                        return False
        return True

    if not verify_222(U2, V2, W2):
        return schoolbook()

    # Build the 4x4x4 decomposition via the Kronecker/tensor-product structure.
    # The idea: treat the 4x4x4 MM as a 2x2x2 MM of 2x2 blocks, where each
    # "scalar" multiplication in the block-level Strassen is itself a 2x2x2 MM.
    # This gives a rank-49 decomposition (7*7) for 4x4x4.
    #
    # The tensor product structure:
    # At the block level, Strassen gives 7 products M_r (r=0..6) of 2x2 matrices.
    # Each M_r = A_r * B_r where A_r and B_r are 2x2 matrices formed from block entries.
    # Each 2x2x2 multiplication can be done with rank 7 (Strassen again).
    # Total: 7 * 7 = 49 rank terms.
    #
    # For the tensor product decomposition:
    # U4[i*4+j, r*7+s] = U2[i*2+ip, r] * U2[ip*2+jp, s]  (for appropriate indices)
    # Actually, the correct tensor product for MM is:
    # If T4 = T2 x T2 (tensor product), then:
    # U4[(i,ip),(j,jp), (r,s)] = U2[i,j,r] * U2[ip,jp,s]
    # V4[(j,jp),(k,kp), (r,s)] = V2[j,k,r] * V2[jp,kp,s]
    # W4[(k,kp),(i,ip), (r,s)] = W2[k,i,r] * W2[kp,ip,s]
    #
    # In flat indices:
    # U4[i*4+j, r*7+s] = U2[i*2+ip, r] * U2[ip*2+jp, s] where j = jp... 
    # Wait, I need to be more careful. Let me use the direct construction.
    #
    # Actually, the tensor product of two MM tensors T2 ⊗ T2 gives the MM tensor for
    # 4x4x4 if we arrange the indices properly. The rank of T2 ⊗ T2 is rank(T2)*rank(T2) = 49.
    #
    # Let me construct it directly:
    # For 4x4x4: A is 4x4, B is 4x4, C is 4x4.
    # U4[i*4+j, col], V4[j*4+k, col], W4[k*4+i, col]
    # col = r*7 + s for r,s in 0..6
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            # U4[i*4+j, col] = sum over all ways... 
            # Actually the tensor product is:
            # U4[i*4+j, r*7+s] = U2[i*2 + (j//2)... 
            # No. Let me think differently.
            #
            # The 4x4 matrix A has entries A[i,j] for i,j in 0..3.
            # Write i = 2*I + I2, j = 2*J + J2 where I,J,I2,J2 in {0,1}.
            # Similarly for B: j = 2*J + J2, k = 2*K + K2
            # And C: k = 2*K + K2, i = 2*I + I2
            #
            # C[k,i] = sum_j A[i,j] * B[j,k]
            # = sum_{J,J2} A[2I+I2, 2J+J2] * B[2J+J2, 2K+K2]
            #
            # The tensor product decomposition says:
            # sum_{r,s} U2[I*2+I2, r] * U2[J2*2+... 
            # Hmm, this is getting confusing. Let me just use the explicit formula.
            #
            # For the tensor product T2 ⊗ T2:
            # The first factor acts on (I, J, K) and the second on (I2, J2, K2).
            # U4[(I,I2),(J,J2), (r,s)] = U2[I,J,r] * U2[I2,J2,s]
            # V4[(J,J2),(K,K2), (r,s)] = V2[J,K,r] * V2[J2,K2,s]
            # W4[(K,K2),(I,I2), (r,s)] = W2[K,I,r] * W2[K2,I2,s]
            #
            # In flat indices:
            # i = 2*I + I2, j = 2*J + J2, k = 2*K + K2
            # U4[i*4+j, r*7+s] = U2[I*2+J, r] * U2[I2*2+J2, s]
            # V4[j*4+k, r*7+s] = V2[J*2+K, r] * V2[J2*2+K2, s]
            # W4[k*4+i, r*7+s] = W2[K*2+I, r] * W2[K2*2+I2, s]

            for I in range(2):
                for I2 in range(2):
                    for J in range(2):
                        for J2 in range(2):
                            i = 2*I + I2
                            j = 2*J + J2
                            U4[i*4+j, col] = U2[I*2+J, r] * U2[I2*2+J2, s]

            for J in range(2):
                for J2 in range(2):
                    for K in range(2):
                        for K2 in range(2):
                            j = 2*J + J2
                            k = 2*K + K2
                            V4[j*4+k, col] = V2[J*2+K, r] * V2[J2*2+K2, s]

            for K in range(2):
                for K2 in range(2):
                    for I in range(2):
                        for I2 in range(2):
                            k = 2*K + K2
                            i = 2*I + I2
                            W4[k*4+i, col] = W2[K*2+I, r] * W2[K2*2+I2, s]

    # Verify the 4x4x4 decomposition
    def verify_444(U4, V4, W4):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = sum(U4[i*4+j, t] * V4[j*4+k, t] * W4[k*4+i, t] for t in range(49))
                    if s != 1:
                        return False
        return True

    if not verify_444(U4, V4, W4):
        # Fallback: use the simpler block approach
        return schoolbook()

    # Now construct 4x4x8 from two 4x4x4 decompositions.
    # B is 4x8: split into B_left (cols 0-3) and B_right (cols 4-7), each 4x4.
    # C is 8x4: split into C_top (rows 0-3) and C_bottom (rows 4-7), each 4x4.
    # C_top = A * B_left, C_bottom = A * B_right
    #
    # U has 16 rows (4*4), V has 32 rows (4*8), W has 32 rows (8*4).
    # Rank = 49 + 49 = 98.
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)

    # Left half: columns 0-3 of B, rows 0-3 of C
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
            W[k*4 + j, 0:49] = W4[k*4 + j, :]

    # Right half: columns 4-7 of B, rows 4-7 of C
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + (4+k), 49:98] = V4[j*4 + k, :]
            W[(4+k)*4 + j, 49:98] = W4[k*4 + j, :]

    # Final verification
    def verify_full(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if s != 1:
                        return False
        return True

    if verify_full(U, V, W, n, m, p, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return schoolbook()
