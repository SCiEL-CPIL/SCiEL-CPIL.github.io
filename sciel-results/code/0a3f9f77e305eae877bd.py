import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Index convention:
    # U[i, r] for i = row * m + col  (A has shape 2x3, indices 0..5)
    # V[j, r] for j = row * p + col  (B has shape 3x3, indices 0..8)
    # W[k, r] for k = col * n + row  (C has shape 3x2, indices 0..5)
    # W[k*n+i, r] means output C[i, j] where i=row (0..1), j=col (0..2)
    # So C[i,j] index in W is j*n + i = 2*j + i

    # Strassen for the 2x2 block: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[0,2], 3=A[1,0], 4=A[1,1], 5=A[1,2]
    # B indices: 0=B[0,0], 1=B[0,1], 2=B[0,2], 3=B[1,0], 4=B[1,1], 5=B[1,2], 6=B[2,0], 7=B[2,1], 8=B[2,2]
    # C indices (W): C[i,j] -> j*2+i
    #   C[0,0]->0, C[0,1]->2, C[0,2]->4
    #   C[1,0]->1, C[1,1]->3, C[1,2]->5

    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2: (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4: A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6: (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Schoolbook: A[:, 2] * B[2, :2] -> C[:, :2]
    # C[0,0] += A[0,2]*B[2,0]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    # C[0,1] += A[0,2]*B[2,1]
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    # C[1,0] += A[1,2]*B[2,0]
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    # C[1,1] += A[1,2]*B[2,1]
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # Schoolbook: A[:, 0:3] * B[:, 2] -> C[:, 2]
    # C[0,2] += A[0,j]*B[j,2] for j=0,1,2
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    # C[1,2] += A[1,j]*B[j,2] for j=0,1,2
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
