import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0

    # Strassen 2x2 decomposition (rank 7)
    # U2[i2,j2,r], V2[j2,k2,r], W2[k2,i2,r] with T[i,j,k]=sum_r U[i,r]V[j,r]W[k,r]=delta(i,k)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # r=0: (A00+A11)(B00+B11) -> C00+, C11+
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,3]=1
    # r=1: (A10+A11)B00 -> C10+, C11-
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,3]=-1
    # r=2: A00(B01-B11) -> C01+, C11+
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,0]=1; W2[3,3]=1
    # r=3: A11(B10-B00) -> C00+, C10+
    U2[3,3]=1
    V2[0,3]=-1; V2[2,3]=1
    W2[0,0]=1; W2[2,1]=1
    # r=4: (A00+A01)B11 -> C00-, C01+
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,0]=-1; W2[1,0]=1
    # r=5: (A10-A00)(B00+B01) -> C11+
    U2[0,5]=-1; U2[2,5]=1
    V2[0,5]=1; V2[1,5]=1
    W2[3,3]=1
    # r=6: (A01-A11)(B10+B11) -> C00+
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,0]=1

    # Schoolbook 3x3 decomposition (rank 27)
    # U3[i3,j3,r]=1, V3[j3,k3,r]=1, W3[k3,i3,r]=1 for r=(i3,j3,k3)
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                r = i3*9 + j3*3 + k3
                U3[i3*3+j3, r] = 1
                V3[j3*3+k3, r] = 1
                W3[k3*3+i3, r] = 1

    # Kronecker product: 6x6 = 2x2 blocks of 3x3
    # Index mapping: i6 = 3*i2 + i3, j6 = 3*j2 + j3, k6 = 3*k2 + k3
    rank = 7 * 27  # 189
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)

    # Precompute non-zero entries
    nz_u2 = [(i2, j2, U2[i2*2+j2, r]) for r in range(7) for i2 in range(2) for j2 in range(2) if U2[i2*2+j2, r] != 0]
    nz_v2 = [(j2, k2, V2[j2*2+k2, r]) for r in range(7) for j2 in range(2) for k2 in range(2) if V2[j2*2+k2, r] != 0]
    nz_w2 = [(k2, i2, W2[k2*2+i2, r]) for r in range(7) for k2 in range(2) for i2 in range(2) if W2[k2*2+i2, r] != 0]
    nz_u3 = [(i3, j3, 1) for i3 in range(3) for j3 in range(3)]  # all ones
    nz_v3 = [(j3, k3, 1) for j3 in range(3) for k3 in range(3)]
    nz_w3 = [(k3, i3, 1) for k3 in range(3) for i3 in range(3)]

    for r2 in range(7):
        for r3 in range(27):
            r = r2*27 + r3

            for (i2, j2, vu) in nz_u2:
                if vu == 0 or U2[i2*2+j2, r2] == 0:
                    continue
                for (i3, j3, _) in nz_u3:
                    if U3[i3*3+j3, r3] == 0:
                        continue
                    i6 = 3*i2 + i3
                    j6 = 3*j2 + j3
                    U6[i6*6+j6, r] = U2[i2*2+j2, r2] * U3[i3*3+j3, r3]

            for (j2, k2, vv) in nz_v2:
                if V2[j2*2+k2, r2] == 0:
                    continue
                for (j3, k3, _) in nz_v3:
                    if V3[j3*3+k3, r3] == 0:
                        continue
                    j6 = 3*j2 + j3
                    k6 = 3*k2 + k3
                    V6[j6*6+k6, r] = V2[j2*2+k2, r2] * V3[j3*3+k3, r3]

            for (k2, i2, vw) in nz_w2:
                if W2[k2*2+i2, r2] == 0:
                    continue
                for (k3, i3, _) in nz_w3:
                    if W3[k3*3+i3, r3] == 0:
                        continue
                    k6 = 3*k2 + k3
                    i6 = 3*i2 + i3
                    W6[k6*6+i6, r] = W2[k2*2+i2, r2] * W3[k3*3+i3, r3]

    # Fast verification using matrix multiplication
    if time.time() < deadline:
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] should equal delta(i,k)
        # Compute via: for each r, outer product of U[:,r], V[:,r], W[:,r]
        # More efficiently: T = sum_r (U[:,r] * V[:,r]) outer with W[:,r] reshaped
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # = sum_r (U @ V.T)[i,j] * W[k,r]... no
        # Let's use: T_ijk = sum_r U_ir V_jr W_kr
        # = sum_r (U_ir W_kr) V_jr
        # Build: M[i,k,r] = U[i,r]*W[k,r], then T[i,j,k] = sum_r M[i,k,r]*V[j,r]
        # Or simply: T = np.einsum('ir,jr,kr->ijk', U6, V6, W6)
        T = np.einsum('ir,jr,kr->ijk', U6, V6, W6)
        expected = np.zeros((6,6,6), dtype=int)
        for i in range(6):
            expected[i, i, i] = 1  # T[i,j,k] = 1 iff i==k... wait
        # Actually T[i,j,k] = delta(i,k) means T[i,j,k]=1 when i==k for all j
        # For matrix mult C[i,k]=sum_j A[i,j]B[j,k], tensor T[i,j,k]=1 for all i,j,k? No.
        # The multiplication tensor: T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
        # It always contributes, so T[i,j,k] = 1 for ALL i,j,k? That can't be decomposed with rank < 216.
        # Wait - the standard tensor for matmul is: T[i,j,k] = delta(i,k) is WRONG.
        # Correct: C[i,k] = sum_j A[i,j] B[j,k], so the tensor is T[i,j,k] = 1 for all i,j,k.
        # No! That's the "all-ones" tensor which has rank n*m*p = 216.
        # The actual matmul tensor: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k, i.e., T[i,j,k]=1 for all i,j,k.
        # Rank of all-ones 6x6x6 tensor is 216 (schoolbook).
        # But Strassen gives rank 7 for 2x2: T2[i,j,k]=1 for all i,j,k in 2x2x2.
        # Let me verify Strassen: sum_r U2[i,r]*V2[j,r]*W2[k,r] should = 1 for all i,j,k.
        T2 = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
        if not np.all(T2 == 1):
            # Fallback: pure schoolbook 6x6
            U6 = np.zeros((36, 216), dtype=int)
            V6 = np.zeros((36, 216), dtype=int)
            W6 = np.zeros((36, 216), dtype=int)
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        r = i*36 + j*6 + k
                        U6[i*6+j, r] = 1
                        V6[j*6+k, r] = 1
                        W6[k*6+i, r] = 1
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

        # Now verify full 6x6
        if not np.all(T == 1):
            # Fallback: pure schoolbook 6x6
            U6 = np.zeros((36, 216), dtype=int)
            V6 = np.zeros((36, 216), dtype=int)
            W6 = np.zeros((36, 216), dtype=int)
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        r = i*36 + j*6 + k
                        U6[i*6+j, r] = 1
                        V6[j*6+k, r] = 1
                        W6[k*6+i, r] = 1
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
