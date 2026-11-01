"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Rank-35 decomposition: Strassen (rank 7) for each of 5 independent 2x2 column blocks.
    # The 2x2x10 tensor is the tensor product of the 2x2x2 MM tensor and a 1x1x5 identity.
    # We use the known rank-7 decomposition of 2x2 matrix multiplication (Strassen).
    #
    # Conventions:
    #   A is 2x2: A[i,j] for i in {0,1}, j in {0,1}
    #   B is 2x10: B[j,k] for j in {0,1}, k in {0..9}
    #   C = A * B is 2x10: C[i,k]
    #
    # U shape: (n*m, rank) = (4, 35), U[i*m+j, r]
    # V shape: (m*p, rank) = (20, 35), V[j*p+k, r]
    # W shape: (p*n, rank) = (20, 35), W[k*n+i, r]  (official trace convention)
    #
    # For each block k in {0..4} (columns 2k and 2k+1 of B), we apply Strassen.
    # Rank = 5 * 7 = 35.

    n, m, p = DIMENSIONS
    rank = 35

    # U: 4 x 35
    U = [[0] * rank for _ in range(4)]
    # V: 20 x 35
    V = [[0] * rank for _ in range(20)]
    # W: 20 x 35
    W = [[0] * rank for _ in range(20)]

    # Strassen's 7 products for 2x2 * 2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # For block k, B's columns are B[0, 2k], B[0, 2k+1], B[1, 2k], B[1, 2k+1]
    # So e=B[0,2k], f=B[0,2k+1], g=B[1,2k], h=B[1,2k+1]
    #
    # U entries (row = i*m + j, i.e., A[i,j]):
    # M1: A[0,0]=1, A[1,1]=1
    # M2: A[1,0]=1, A[1,1]=1
    # M3: A[0,0]=1
    # M4: A[1,1]=1
    # M5: A[0,0]=1, A[0,1]=1
    # M6: A[1,0]=1, A[0,0]=-1
    # M7: A[0,1]=1, A[1,1]=-1
    #
    # V entries (row = j*p + k):
    # M1: V[0*20+2k]=1, V[1*20+(2k+1)]=1
    # M2: V[0*20+2k]=1
    # M3: V[0*20+(2k+1)]=1, V[1*20+(2k+1)]=-1
    # M4: V[1*20+(2k+1)]=1, V[1*20+2k]=-1
    # M5: V[1*20+(2k+1)]=1
    # M6: V[0*20+2k]=1, V[0*20+(2k+1)]=1
    # M7: V[1*20+(2k+1)]=1, V[1*20+2k]=1
    #
    # W entries (row = k*n + i, i.e., C[i,k] -> row = k*2 + i):
    # For block k, the two output columns are 2k and 2k+1.
    # C[0,2k] -> W row = 2k*2 + 0 = 4k
    # C[1,2k] -> W row = 2k*2 + 1 = 4k+1
    # C[0,2k+1] -> W row = (2k+1)*2 + 0 = 4k+2
    # C[1,2k+1] -> W row = (2k+1)*2 + 1 = 4k+3
    #
    # M1: W[4k]=1, W[4k+3]=1
    # M2: W[4k+1]=1, W[4k+3]=-1
    # M3: W[4k+2]=1, W[4k+3]=1
    # M4: W[4k]=1, W[4k+1]=1
    # M5: W[4k]=-1, W[4k+2]=1
    # M6: W[4k+3]=1
    # M7: W[4k]=1

    # U contributions per Strassen term (independent of block k)
    U_contrib = [
        {0: 1, 3: 1},       # M1: A[0,0], A[1,1]
        {2: 1, 3: 1},       # M2: A[1,0], A[1,1]
        {0: 1},             # M3: A[0,0]
        {3: 1},             # M4: A[1,1]
        {0: 1, 1: 1},       # M5: A[0,0], A[0,1]
        {2: 1, 0: -1},      # M6: A[1,0], A[0,0]
        {1: 1, 3: -1},      # M7: A[0,1], A[1,1]
    ]

    # V contributions per Strassen term (depends on block k)
    def V_contrib(s, k):
        col0 = 2 * k
        col1 = 2 * k + 1
        # Row = j * p + col
        if s == 0:  # M1
            return {0 * 20 + col0: 1, 1 * 20 + col1: 1}
        elif s == 1:  # M2
            return {0 * 20 + col0: 1}
        elif s == 2:  # M3
            return {0 * 20 + col1: 1, 1 * 20 + col1: -1}
        elif s == 3:  # M4
            return {1 * 20 + col1: 1, 1 * 20 + col0: -1}
        elif s == 4:  # M5
            return {1 * 20 + col1: 1}
        elif s == 5:  # M6
            return {0 * 20 + col0: 1, 0 * 20 + col1: 1}
        elif s == 6:  # M7
            return {1 * 20 + col1: 1, 1 * 20 + col0: 1}
        return {}

    # W contributions per Strassen term (depends on block k)
    def W_contrib(s, k):
        # Output rows: 4k (C[0,2k]), 4k+1 (C[1,2k]), 4k+2 (C[0,2k+1]), 4k+3 (C[1,2k+1])
        if s == 0:  # M1: C[0,2k] += 1, C[1,2k+1] += 1
            return {4 * k: 1, 4 * k + 3: 1}
        elif s == 1:  # M2: C[1,2k] += 1, C[1,2k+1] -= 1
            return {4 * k + 1: 1, 4 * k + 3: -1}
        elif s == 2:  # M3: C[0,2k+1] += 1, C[1,2k+1] += 1
            return {4 * k + 2: 1, 4 * k + 3: 1}
        elif s == 3:  # M4: C[0,2k] += 1, C[1,2k] += 1
            return {4 * k: 1, 4 * k + 1: 1}
        elif s == 4:  # M5: C[0,2k] -= 1, C[0,2k+1] += 1
            return {4 * k: -1, 4 * k + 2: 1}
        elif s == 5:  # M6: C[1,2k+1] += 1
            return {4 * k + 3: 1}
        elif s == 6:  # M7: C[0,2k] += 1
            return {4 * k: 1}
        return {}

    for k in range(5):
        for s in range(7):
            col = 7 * k + s
            # Fill U
            for urow, val in U_contrib[s].items():
                U[urow][col] = val
            # Fill V
            for vrow, val in V_contrib(s, k).items():
                V[vrow][col] = val
            # Fill W
            for wrow, val in W_contrib(s, k).items():
                W[wrow][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
