"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

# Strassen's rank-7 decomposition for 2x2 matrix multiplication
# The 7 products are:
# M1 = (A11 + A22) * (B11 + B22)
# M2 = (A21 + A22) * B11
# M3 = A11 * (B12 - B22)
# M4 = A22 * (B21 - B11)
# M5 = (A11 + A12) * B22
# M6 = (A21 - A11) * (B11 + B12)
# M7 = (A12 - A22) * (B21 + B22)
#
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6
#
# Strassen U, V, W for 2x2 (indices: i*2+j for U, j*2+k for V, k*2+i for W)
# U[i*2+j, c] = coefficient of A[i,j] in product c
# V[j*2+k, c] = coefficient of B[j,k] in product c
# W[k*2+i, c] = coefficient of product c in C[i,k]

def _strassen_2x2():
    """Return U, V, W for Strassen 2x2 (rank 7)."""
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Product 0: M1 = (A11 + A22) * (B11 + B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11 (k=0, i=0)
    W[3, 0] = 1  # C22 (k=1, i=1)
    
    # Product 1: M2 = (A21 + A22) * B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21 (k=1, i=0)
    
    # Product 2: M3 = A11 * (B12 - B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12 (k=0, i=1)
    
    # Product 3: M4 = A22 * (B21 - B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1 # B11
    W[2, 3] = 1  # C21 (k=1, i=0)
    W[0, 3] = 1  # C11 (k=0, i=0)
    
    # Product 4: M5 = (A11 + A12) * B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1 # C11 (k=0, i=0)
    W[1, 4] = 1  # C12 (k=0, i=1)
    
    # Product 5: M6 = (A21 - A11) * (B11 + B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22 (k=1, i=1)
    
    # Product 6: M7 = (A12 - A22) * (B21 + B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1 # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11 (k=0, i=0)
    
    return U, V, W

def _kron_combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two tensor decompositions via Kronecker product.
    If T1 = sum_c U1[:,c] V1[:,c] W1[:,c] and T2 = sum_c' U2[:,c'] V2[:,c'] W2[:,c'],
    then T1 x T2 (block tensor) has rank rank1*rank2 with factors:
    U = kron(U1, U2), V = kron(V1, V2), W = kron(W1, W2)
    """
    # U has shape (n1*m1, r1) and (n2*m2, r2) -> (n1*m1*n2*m2, r1*r2)
    # We need to be careful about the ordering.
    # The 4x4 matrix A is partitioned into 2x2 blocks of 2x2 matrices.
    # A[i1*2+i2, j1*2+j2] where i1,j1 in {0,1} are block indices, i2,j2 in {0,1} are within-block.
    # U for the combined tensor: index (i1*2+i2, j1*2+j2) -> combined index
    # The Kronecker product structure: U_combined[(i1*2+i2)*4 + (j1*2+j2), c1*7+c2]
    # = U1[i1*2+j1, c1] * U2[i2*2+j2, c2]
    
    n1m1, r1 = U1.shape
    n2m2, r2 = U2.shape
    r = r1 * r2
    U_comb = np.zeros((n1m1 * n2m2, r), dtype=np.float64)
    V_comb = np.zeros((V1.shape[0] * V2.shape[0], r), dtype=np.float64)
    W_comb = np.zeros((W1.shape[0] * W2.shape[0], r), dtype=np.float64)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            U_comb[:, c] = np.kron(U1[:, c1], U2[:, c2])
            V_comb[:, c] = np.kron(V1[:, c1], V2[:, c2])
            W_comb[:, c] = np.kron(W1[:, c1], W2[:, c2])
    
    return U_comb, V_comb, W_comb

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product of two Strassen rank-7 decompositions.
    """
    U1, V1, W1 = _strassen_2x2()
    U2, V2, W2 = _strassen_2x2()
    
    U, V, W = _kron_combine(U1, V1, W1, U2, V2, W2)
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
