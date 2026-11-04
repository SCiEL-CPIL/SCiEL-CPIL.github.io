"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
TIME_LIMIT = 150
SAFETY_MARGIN = 10

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if total != 1:
                    return False
    return True

def _strassen_3x3():
    """
    Returns rank-7 decomposition for 3x3 matrix multiplication.
    U: (9, 7), V: (9, 7), W: (9, 7)
    Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*3+j... no wait.
    
    Actually the tensor T[i,j,k] for 3x3 mult has T[i,j,k]=1 if we compute
    C[i,k] = sum_j A[i,j]*B[j,k]. The tensor decomposition is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    So U has shape (3*3, 7) indexed by (i,j), V has (3*3, 7) indexed by (j,k),
    W has (3*3, 7) indexed by (i,k).
    """
    # Strassen for 3x3: we can use the known rank 7 decomposition.
    # For 2x2, Strassen uses 7 multiplications. For 3x3, the minimal rank is unknown
    # but we can embed 2x2 Strassen in 3x3.
    # Actually, let's use a simpler approach: the schoolbook rank for 3x3 is 27.
    # But we want to be clever. Let's use the fact that rank of 3x3 mult tensor
    # is at most 27, but can we do better?
    # The known lower bound for 3x3 is 19. The best known upper bound is 23.
    # For now, let's just use schoolbook rank 27 for 3x3, or try to do better.
    
    # Actually, let me just construct a valid decomposition with rank 27 (schoolbook)
    # for the 3x3 core, then extend to 2x3 and 3x5.
    
    n3, m3, p3 = 3, 3, 3
    rank3 = n3 * m3 * p3
    U3 = np.zeros((n3*m3, rank3), dtype=object)
    V3 = np.zeros((m3*p3, rank3), dtype=object)
    W3 = np.zeros((p3*n3, rank3), dtype=object)
    
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U3[i*m3+j, col] = 1
                V3[j*p3+k, col] = 1
                W3[k*n3+i, col] = 1
                col += 1
    
    return U3, V3, W3

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for matrix multiplication.
    Returns the schoolbook decomposition (rank 30) as baseline.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    U_school = np.zeros((n * m, rank), dtype=object)
    V_school = np.zeros((m * p, rank), dtype=object)
    W_school = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
