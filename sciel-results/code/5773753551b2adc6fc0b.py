"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_rank7():
    """Return the classic Strassen rank-7 decomposition of 2x2x2 matrix multiplication.

    Conventions:
      U shape (4, 7): row index a = i*2 + j
      V shape (4, 7): row index b = j*2 + k
      W shape (4, 7): row index c = k*2 + i   (trace convention W[k*n + i, r])

    The 7 Strassen products:
      P1 = (A00 + A11)(B00 + B11)
      P2 = (A10 + A11) B00
      P3 = A00 (B10 - B11)
      P4 = A11 (B01 - B00)
      P5 = (A00 + A01) B11
      P6 = (A10 - A00)(B00 + B01)
      P7 = (A01 - A11)(B10 + B11)

    C00 = P1 + P4 - P5 - P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P7
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # Row indices for U: (i, j) -> i*2 + j
    #   (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Row indices for V: (j, k) -> j*2 + k
    #   (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Row indices for W: (k, i) -> k*2 + i
    #   (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

    # P1 = (A00 + A11)(B00 + B11)
    U[0, 0] += 1; U[3, 0] += 1   # A00, A11
    V[0, 0] += 1; V[3, 0] += 1   # B00, B11
    W[0, 0] += 1; W[3, 0] += 1   # C00 (k=0,i=0), C11 (k=1,i=1)

    # P2 = (A10 + A11) B00
    U[2, 1] += 1; U[3, 1] += 1   # A10, A11
    V[0, 1] += 1                 # B00
    W[2, 1] += 1                 # C10 (k=1,i=0)

    # P3 = A00 (B10 - B11)
    U[0, 2] += 1                 # A00
    V[2, 2] += 1; V[3, 2] -= 1   # B10, -B11
    W[1, 2] += 1; W[3, 2] += 1   # C01 (k=0,i=1), C11 (k=1,i=1)

    # P4 = A11 (B01 - B00)
    U[3, 3] += 1                 # A11
    V[1, 3] += 1; V[0, 3] -= 1   # B01, -B00
    W[0, 3] += 1; W[2, 3] += 1   # C00 (k=0,i=0), C10 (k=1,i=0)

    # P5 = (A00 + A01) B11
    U[0, 4] += 1; U[1, 4] += 1   # A00, A01
    V[3, 4] += 1                 # B11
    W[0, 4] -= 1; W[1, 4] += 1   # -C00, C01

    # P6 = (A10 - A00)(B00 + B01)
    U[2, 5] += 1; U[0, 5] -= 1   # A10, -A00
    V[0, 5] += 1; V[1, 5] += 1   # B00, B01
    W[2, 6] += 1                 # C10 (k=1,i=0)

    # P7 = (A01 - A11)(B10 + B11)
    U[1, 6] += 1; U[3, 6] -= 1   # A01, -A11
    V[2, 6] += 1; V[3, 6] += 1   # B10, B11
    W[0, 6] -= 1; W[3, 6] += 1   # -C00, C11

    return U, V, W


def _kron_decompose_4x4x4():
    """Build rank-49 decomposition of 4x4x4 MM via Kronecker product of two Strassen rank-7.

    The 4x4x4 MM tensor T4[i,j,k] = 1 is the Kronecker product of T2 with itself:
      T4[(i1,i2),(j1,j2),(k1,k2)] = T2[i1,j1,k1] * T2[i2,j2,k2]

    If T2 = sum_{r=1}^{7} U2[:,r] ⊗ V2[:,r] ⊗ W2[:,r], then
      T4 = sum_{r,s=1}^{7} (U2[:,r] ⊗ U2[:,s]) ⊗ (V2[:,r] ⊗ V2[:,s]) ⊗ (W2[:,r] ⊗ W2[:,s])

    This gives rank 49 with entries in {-1, 0, 1}.
    """
    U2, V2, W2 = _strassen_rank7()
    # Kronecker product: U4[a, r*7+s] = U2[a2, r] * U2[a1, s]
    # where a = a2*4 + a1, r*7+s is the rank index
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)

    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            U4[:, col] = np.kron(U2[:, r], U2[:, s])
            V4[:, col] = np.kron(V2[:, r], V2[:, s])
            W4[:, col] = np.kron(W2[:, r], W2[:, s])

    return U4, V4, W4


def _build_mm_tensor_4x4x4():
    """Build the 4x4x4 matrix multiplication tensor."""
    n = m = p = 4
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, tol=1e-9):
    """Verify exact decomposition of 4x4x4 MM tensor."""
    T = _build_mm_tensor_4x4x4()
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol, rtol=0)


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 150s limit

    # Deterministic Kronecker construction: rank 49, exact, O(1) time
    U, V, W = _kron_decompose_4x4x4()

    # Verify correctness
    if not _verify(U, V, W):
        # Fallback to schoolbook rank 64 (should never trigger)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Convert to list of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
