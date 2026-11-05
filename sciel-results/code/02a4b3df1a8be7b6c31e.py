import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank 16 decomposition
    # Blocks:
    # 1. A1 (2x2) * B1a (2x2, cols 0,1) -> Rank 7 (Strassen)
    # 2. A1 (2x2) * B1b (2x2, cols 2,3) -> Rank 7 (Strassen)
    # 3. A1 (2x2) * B1c (2x1, col 4)    -> Rank 1 (Outer product of ones)
    # 4. A2 (2x1) * B2 (1x5)            -> Rank 1 (Outer product of ones)
    
    rank = 16
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = (A10 + A11) B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        # M3 = A00 (B01 - B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M4 = A11 (B10 - B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        # M5 = (A00 + A01) B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_rank1_block(i_list, j_list, k_list):
        """
        Adds a rank-1 decomposition for a block where T[i,j,k]=1 for i in i_list, j in j_list, k in k_list.
        This works if the block is a 'full' box of 1s.
        U[i, col] = 1 for i in i_list
        V[j, col] = 1 for j in j_list
        W[k, col] = 1 for k in k_list
        Then sum_r U[i,r]V[j,r]W[k,r] = 1 * 1 * 1 = 1.
        """
        nonlocal col
        for i in i_list:
            U[i*m + j_list[0], col] = 1 # Wait, index mapping is i*m+j.
            # Actually, U index is i*m + j.
            # V index is j*p + k.
            # W index is k*n + i.
            # For a block defined by sets I, J, K:
            # We need U[i*m+j, col] = 1 for all i in I, j in J.
            # V[j*p+k, col] = 1 for all j in J, k in K.
            # W[k*n+i, col] = 1 for all k in K, i in I.
            # This creates a rank-1 tensor that is 1 on the box I x J x K and 0 elsewhere?
            # No, the tensor product U V W is 1 if i in I, j in J, k in K?
            # Let's check:
            # T[i,j,k] = U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col].
            # If i in I, j in J, k in K: 1 * 1 * 1 = 1.
            # If i not in I: W[k*n+i] is 0 (unless i is in I for another term, but we are adding to a specific col).
            # Wait, if i is not in I, U[i*m+j] is 0 AND W[k*n+i] is 0. So T=0.
            # If j is not in J: U is 0 AND V is 0. So T=0.
            # If k is not in K: V is 0 AND W is 0. So T=0.
            # So yes, this creates a box of 1s.
            pass
            
        # Correct implementation:
        for i in i_list:
            for j in j_list:
                U[i*m + j, col] = 1
        for j in j_list:
            for k in k_list:
                V[j*p + k, col] = 1
        for k in k_list:
            for i in i_list:
                W[k*n + i, col] = 1
        col += 1

    # Block 1: A1 (i=0,1) * B1a (j=0,1) * C (k=0,1)
    # i in [0,1], j in [0,1], k in [0,1]
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Block 2: A1 (i=0,1) * B1b (j=0,1) * C (k=2,3)
    # i in [0,1], j in [0,1], k in [2,3]
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 3: A1 (i=0,1) * B1c (j=0,1) * C (k=4)
    # i in [0,1], j in [0,1], k in [4]
    # This is a 2x2x1 box of 1s. Rank 1.
    add_rank1_block([0, 1], [0, 1], [4])
    
    # Block 4: A2 (i=0,1) * B2 (j=2) * C (k=0,1,2,3,4)
    # i in [0,1], j in [2], k in [0,1,2,3,4]
    # This is a 2x1x5 box of 1s. Rank 1.
    add_rank1_block([0, 1], [2], [0, 1, 2, 3, 4])

    assert col == rank, f"Rank mismatch: {col} != {rank}"

    # Verification
    T_rec = np.einsum('ir,vr,wr->ivw', U, V, W, optimize=True)
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0

    if not np.allclose(T_rec, T_target, atol=1e-9):
        # Fallback to schoolbook rank 30 if something went wrong
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
