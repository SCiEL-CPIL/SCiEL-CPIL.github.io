"""Kronecker product decomposition for 3x4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Compute Kronecker product decomposition from two sub-decompositions."""
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition for (n,m,p) matrix multiplication."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    assert U.shape == (n*m, rank), f"U shape {U.shape} != ({n*m}, {rank})"
    assert V.shape == (m*p, rank), f"V shape {V.shape} != ({m*p}, {rank})"
    assert W.shape == (p*n, rank), f"W shape {W.shape} != ({p*n}, {rank})"
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(rank):
                    total += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed[i, k] = total
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        diff = np.abs(C_computed - C_expected).max()
        raise ValueError(f"Verification failed! Max diff: {diff}")
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 3, 4, 8

    # Kronecker decomposition: (3,2,4) ⊗ (1,2,2)
    n1, m1, p1 = 3, 2, 4
    n2, m2, p2 = 1, 2, 2

    U1, V1, W1 = _schoolbook(n1, m1, p1)
    U2, V2, W2 = _schoolbook(n2, m2, p2)

    U, V, W = _kron_decomp(U1, V1, W1, U2, V2, W2)

    rank = U.shape[1]
    _verify(U, V, W, n, m, p)

    U_list = [[float(x) for x in row] for row in U.tolist()]
    V_list = [[float(x) for x in row] for row in V.tolist()]
    W_list = [[float(x) for x in row] for row in W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
