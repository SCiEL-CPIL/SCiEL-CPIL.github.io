"""2x5x6 matrix multiplication tensor decomposition.

Strategy:
  1. Build a Strassen-based rank-54 decomposition (split m=5 as 2+1+2, p=6 as 2+2+2).
  2. Verify it; if it fails, fall back to schoolbook rank 60.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul using Strassen.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1

    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1

    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1

    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _verify_strassen_222(sU, sV, sW):
    """Verify the 2x2x2 Strassen decomposition."""
    n = m = p = 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(7):
                    s += sU[i * m + j, r] * sV[j * p + k, r] * sW[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _build_strassen_256():
    """Build the 2x5x6 decomposition.

    Strategy: Split m=5 as 2+1+2, p=6 as 2+2+2.
    - A (2x5) = [A_a (2x2) | A_b (2x1) | A_c (2x2)]
    - B (5x6) = [B_a (2x6); B_b (1x6); B_c (2x6)]

    Each 2x2x6 block uses 3 Strassen 2x2x2 (rank 7 each) = rank 21
    The 2x1x6 block uses schoolbook = rank 12

    Total: 21 + 12 + 21 = 54
    """
    n, m, p = 2, 5, 6
    total_rank = 54

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    sU, sV, sW = _strassen_222()
    col = 0

    # Part 1: A_a * B_a (2x2 x 6), three Strassen 2x2x2 blocks
    # A_a occupies rows 0,1 of A (i.e., columns j=0,1)
    # B_a occupies rows 0,1 of B (i.e., columns j=0,1)
    for block in range(3):
        k_off = block * 2
        for r in range(7):
            # U: i in {0,1}, j in {0,1}
            U[0, col] = sU[0, r]   # i=0, j=0
            U[1, col] = sU[1, r]   # i=0, j=1
            U[5, col] = sU[2, r]   # i=1, j=0
            U[6, col] = sU[3, r]   # i=1, j=1

            # V: j in {0,1}, k in {k_off, k_off+1}
            V[0 * 6 + k_off + 0, col] = sV[0, r]  # j=0, k=k_off
            V[0 * 6 + k_off + 1, col] = sV[1, r]  # j=0, k=k_off+1
            V[1 * 6 + k_off + 0, col] = sV[2, r]  # j=1, k=k_off
            V[1 * 6 + k_off + 1, col] = sV[3, r]  # j=1, k=k_off+1

            # W: k in {k_off, k_off+1}, i in {0,1}
            W[(k_off + 0) * 2 + 0, col] = sW[0, r]  # k=k_off, i=0
            W[(k_off + 0) * 2 + 1, col] = sW[1, r]  # k=k_off, i=1
            W[(k_off + 1) * 2 + 0, col] = sW[2, r]  # k=k_off+1, i=0
            W[(k_off + 1) * 2 + 1, col] = sW[3, r]  # k=k_off+1, i=1
            col += 1
    # col = 21

    # Part 2: A_b * B_b (2x1 x 6), schoolbook rank 12
    # A_b occupies column j=2 of A
    # B_b occupies row j=2 of B
    for i in range(2):
        for k in range(6):
            U[i * 5 + 2, col] = 1   # i, j=2
            V[2 * 6 + k, col] = 1   # j=2, k
            W[k * 2 + i, col] = 1   # k, i
            col += 1
    # col = 33

    # Part 3: A_c * B_c (2x2 x 6), three Strassen 2x2x2 blocks
    # A_c occupies columns j=3,4 of A
    # B_c occupies rows j=3,4 of B
    for block in range(3):
        k_off = block * 2
        for r in range(7):
            # U: i in {0,1}, j in {3,4}
            U[0 * 5 + 3, col] = sU[0, r]  # i=0, j=3
            U[0 * 5 + 4, col] = sU[1, r]  # i=0, j=4
            U[1 * 5 + 3, col] = sU[2, r]  # i=1, j=3
            U[1 * 5 + 4, col] = sU[3, r]  # i=1, j=4

            # V: j in {3,4}, k in {k_off, k_off+1}
            V[3 * 6 + k_off + 0, col] = sV[0, r]  # j=3, k=k_off
            V[3 * 6 + k_off + 1, col] = sV[1, r]  # j=3, k=k_off+1
            V[4 * 6 + k_off + 0, col] = sV[2, r]  # j=4, k=k_off
            V[4 * 6 + k_off + 1, col] = sV[3, r]  # j=4, k=k_off+1

            # W: k in {k_off, k_off+1}, i in {0,1}
            W[(k_off + 0) * 2 + 0, col] = sW[0, r]  # k=k_off, i=0
            W[(k_off + 0) * 2 + 1, col] = sW[1, r]  # k=k_off, i=1
            W[(k_off + 1) * 2 + 0, col] = sW[2, r]  # k=k_off+1, i=0
            W[(k_off + 1) * 2 + 1, col] = sW[3, r]  # k=k_off+1, i=1
            col += 1
    # col = 54

    assert col == total_rank
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=5, p=6):
    """Verify the full decomposition. U, V, W are numpy arrays."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s = float(np.dot(u_row, v_row * w_row))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _check_half_integer(U, V, W):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-12):
            return False
    return True


def _schoolbook():
    """Schoolbook decomposition, rank = n*m*p = 60."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    t0 = time.time()
    deadline = t0 + 120  # safety margin before 150s kill

    # Primary: Strassen-based rank-54
    try:
        result = _build_strassen_256()
        if time.time() < deadline:
            if _verify_decomposition(result['U'], result['V'], result['W']):
                if _check_half_integer(result['U'], result['V'], result['W']):
                    return {
                        "U": result['U'].tolist(),
                        "V": result['V'].tolist(),
                        "W": result['W'].tolist()
                    }
    except Exception:
        pass

    # Fallback: schoolbook rank-60
    sb = _schoolbook()
    return {
        "U": sb['U'].tolist(),
        "V": sb['V'].tolist(),
        "W": sb['W'].tolist()
    }

# EVOLVE-BLOCK-END
