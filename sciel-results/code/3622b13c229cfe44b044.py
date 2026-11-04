"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor product (Kronecker product) of the rank-7 Strassen 
    decomposition for 2x2 matrices.
    
    The rank of the 4x4 matrix multiplication tensor is known to be 49 
    (best known explicit bound). This construction achieves this rank 
    using integer coefficients, satisfying the exactness requirement.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen decomposition matrices
    # U2 is 4x7 (indices 0..3 for 2x2 matrix elements, 0..6 for rank)
    # V2 is 4x7
    # W2 is 4x7
    
    # We use the standard Strassen coefficients.
    # M0: (A00+A11)(B00+B11)
    # M1: (A10+A11)B00
    # M2: A00(B01-B11)
    # M3: A11(B10-B00)
    # M4: (A00+A01)B11
    # M5: (A10-A00)(B00+B01)
    # M6: (A01-A11)(B10+B11)
    
    # Note: The mapping of indices to U, V, W must be consistent with the 
    # output trace convention W[k*n+i, r].
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Construct U: (n*m) x rank
    # U[i, j, c] where i,j are 4x4 indices, c is rank index (c1*7+c2)
    # U[i,j, c1*7+c2] = U2[i1,j1,c1] * U2[i2,j2,c2]
    # i = 2*i1 + i2, j = 2*j1 + j2
    
    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        for j in range(m):
            j1 = j // 2
            j2 = j % 2
            for c1 in range(7):
                u1 = U2[i1 * 2 + j1, c1]
                if u1 == 0:
                    continue
                for c2 in range(7):
                    u2 = U2[i2 * 2 + j2, c2]
                    if u2 == 0:
                        continue
                    U[i * m + j, c1 * 7 + c2] = u1 * u2

    # Construct V: (m*p) x rank
    # V[j, k, c] where j,k are 4x4 indices
    # j = 2*j1 + j2, k = 2*k1 + k2
    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 2
            k2 = k % 2
            for c1 in range(7):
                v1 = V2[j1 * 2 + k1, c1]
                if v1 == 0:
                    continue
                for c2 in range(7):
                    v2 = V2[j2 * 2 + k2, c2]
                    if v2 == 0:
                        continue
                    V[j * p + k, c1 * 7 + c2] = v1 * v2

    # Construct W: (p*n) x rank
    # W[k, i, c] where k,i are 4x4 indices
    # k = 2*k1 + k2, i = 2*i1 + i2
    # Note: The trace convention is W[k*n+i, r].
    for k in range(p):
        k1 = k // 2
        k2 = k % 2
        for i in range(n):
            i1 = i // 2
            i2 = i % 2
            for c1 in range(7):
                w1 = W2[k1 * 2 + i1, c1]
                if w1 == 0:
                    continue
                for c2 in range(7):
                    w2 = W2[k2 * 2 + i2, c2]
                    if w2 == 0:
                        continue
                    W[k * n + i, c1 * 7 + c2] = w1 * w2

    if time.time() > deadline:
        # Fallback should not be needed for this fixed construction
        pass

    # Convert to lists as per interface requirement if necessary, 
    # though numpy arrays are often accepted. 
    # The prompt says "factor matrices of shapes...". 
    # Usually lists of lists are safest for strict type checking.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
