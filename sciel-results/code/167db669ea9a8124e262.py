import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    via the tensor (Kronecker) product of the rank-7 Strassen scheme.
    """
    start = time.time()
    limit = start + 110  # Safety margin

    n, m, p = DIMENSIONS

    # Strassen 2x2x2 factor matrices (4,7)
    # Rows correspond to flattened 2x2 matrix indices (0,1,2,3)
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

    # Compute Kronecker products
    U_kron = np.kron(U2, U2)
    V_kron = np.kron(V2, V2)
    W_kron = np.kron(W2, W2)
    
    # Shapes are (16, 49)
    
    # Generate permutations to match standard row-major flattening of 4x4 matrices
    # For U (input A):
    # Standard index i_std = 4*r + c
    # Kron index i_kron = 8*r1 + 4*r2 + 2*s1 + s2
    # where r = 2*r1 + s1, c = 2*r2 + s2
    # We need perm_U such that U_std[i_std] = U_kron[perm_U[i_std]]
    
    perm_U = np.zeros(16, dtype=int)
    for r1 in range(2):
        for r2 in range(2):
            for s1 in range(2):
                for s2 in range(2):
                    r = 2 * r1 + s1
                    c = 2 * r2 + s2
                    i_std = 4 * r + c
                    i_kron = 8 * r1 + 4 * r2 + 2 * s1 + s2
                    perm_U[i_std] = i_kron
                    
    # For V (input B):
    # V corresponds to B. Standard index j_std = 4*c + k (col c, row k of B? No, B is m x p = 4x4)
    # In matrix mult C = A B, A is n x m, B is m x p.
    # Flattened B is usually row-major: B_{j,k} at j*p + k.
    # In the 2x2 block view of B (4x4), blocks are B_{00}, B_{01}, B_{10}, B_{11}.
    # The Strassen decomposition for 2x2 B uses indices for B_{ij}.
    # The Kronecker product structure for V should follow the same index logic as U, 
    # because the tensor structure is symmetric in the "input" roles if we consider the indices.
    # However, let's verify.
    # U acts on A. V acts on B.
    # A_{ij} -> index 4*i + j.
    # B_{jk} -> index 4*j + k.
    # The Kronecker expansion of V2 (which is 4x7) gives V_kron.
    # The row index of V_kron corresponds to the flattened index of the 2x2 B sub-blocks.
    # The mapping from standard 4x4 B index to Kron index is identical to A's mapping.
    perm_V = perm_U.copy()
    
    # For W (output C):
    # C_{ik} -> index 4*i + k.
    # W2 corresponds to C_{ik}.
    # The mapping is identical.
    perm_W = perm_U.copy()

    U = U_kron[perm_U, :]
    V = V_kron[perm_V, :]
    W = W_kron[perm_W, :]

    if time.time() > limit:
        raise TimeoutError()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
