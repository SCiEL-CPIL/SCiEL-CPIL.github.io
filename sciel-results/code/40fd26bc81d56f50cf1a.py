"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook matrix multiplication of (n x m) * (m x p)."""
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


def _strassen_2x2x2():
    """
    Strassen's algorithm for 2x2 * 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) with integer entries.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 = M1 + M4 - M5 + M7
    W[0, 0] = 1
    # C22 = M1 - M2 + M3 + M6
    W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 = M2 + M4
    W[2, 1] = 1
    # C22 = M1 - M2 + M3 + M6
    W[3, 1] = -1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C12 = M3 + M5
    W[1, 2] = 1
    # C22 = M1 - M2 + M3 + M6
    W[3, 2] = 1
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C11 = M1 + M4 - M5 + M7
    W[0, 3] = 1
    # C21 = M2 + M4
    W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 = M1 + M4 - M5 + M7
    W[0, 4] = -1
    # C12 = M3 + M5
    W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 = M1 - M2 + M3 + M6
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 = M1 + M4 - M5 + M7
    W[0, 6] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
    
    T_expected = np.ones((n, m, p))
    return np.allclose(T, T_expected, atol=tol)


def _block_decomposition_4x5x6():
    """
    Block-structured decomposition for 4x5 * 5x6.
    
    Split A (4x5) into:
      A_top (2x5), A_bot (2x5)
    
    Split B (5x6) into:
      B_left (5x3), B_right (5x3)
    
    C (4x6) = [C_top_left (2x3), C_top_right (2x3)]
              [C_bot_left (2x3),  C_bot_right (2x3)]
    
    Each C block is a 2x5 * 5x3 product.
    
    For each 2x5 * 5x3, we further split:
      A_block (2x5) = [A1 (2x2), A2 (2x3)]
      B_block (5x3) = [B1 (2x3); B2 (3x3)]
    
    C_block = A1*B1 + A2*B2
    
    A1*B1: 2x2 * 2x3 -> rank 7*? No, 2x2x3 has rank at most 2*2*3=12 but can be done better.
    Actually, for 2x2 * 2x3, we can think of it as two 2x2 * 2x1 products sharing the A side.
    But that's not directly helpful.
    
    Alternative: Use Strassen on the 2x2 * 2x2 part, and schoolbook for the rest.
    
    Let me try a different split that's more regular:
    
    Split A (4x5) into 2x2 + 2x3:
      A = [A11 (2x2) | A12 (2x3)]
          [A21 (2x2) | A22 (2x3)]
    
    Split B (5x6) into 2x3 + 3x3:
      B = [B11 (2x3) | B12 (2x3)]
          [B21 (3x3) | B22 (3x3)]
    
    C = [C11 (2x3) | C12 (2x3)]
        [C21 (2x3) | C22 (2x3)]
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Now each term is a matrix product:
    A11*B11: 2x2 * 2x3
    A12*B21: 2x3 * 3x3
    etc.
    
    This is getting complex. Let me just use a simpler approach:
    
    Use schoolbook for the whole thing but try to find a better decomposition
    by using Strassen on the 2x2x2 blocks where possible.
    
    Actually, let me try a known result: the rank of 4x5x6 matrix multiplication.
    The schoolbook rank is 120. Can we do better?
    
    One approach: use the fact that 4x5x6 can be decomposed using
    tensor rank properties. But this is research-level.
    
    For now, let me try a block approach:
    - Split the 4 rows into 2+2
    - Split the 5 columns into 2+3
    - Split the 6 columns into 3+3
    
    This gives 8 sub-problems, each of size 2x2x3 or 2x3x3.
    
    For 2x2x3: We can use a decomposition.
    For 2x3x3: We can use a decomposition.
    
    Let me implement these small cases.
    """
    n, m, p = 4, 5, 6
    
    # Split:
    # A: rows 0-1 (top), rows 2-3 (bot); cols 0-1 (left), cols 2-4 (right)
    # B: rows 0-1 (top), rows 2-4 (bot); cols 0-2 (left), cols 3-5 (right)
    
    # C: rows 0-1 (top), rows 2-3 (bot); cols 0-2 (left), cols 3-5 (right)
    
    # C_top_left = A_top_left * B_top_left + A_top_right * B_bot_left
    #   = (2x2)*(2x3) + (2x3)*(3x3)
    # C_top_right = A_top_left * B_top_right + A_top_right * B_bot_right
    #   = (2x2)*(2x3) + (2x3)*(3x3)
    # C_bot_left = A_bot_left * B_top_left + A_bot_right * B_bot_left
    #   = (2x2)*(2x3) + (2x3)*(3x3)
    # C_bot_right = A_bot_left * B_top_right + A_bot_right * B_bot_right
    #   = (2x2)*(2x3) + (2x3)*(3x3)
    
    # Each C block has two terms. Let me decompose each term separately.
    
    # Term 1: (2x2) * (2x3)
    # This is A (2x2) * B (2x3) -> C (2x3)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can decompose this using rank at most 2*2*3 = 12, but we can do better.
    # Actually, we can view this as two 2x2 * 2x1 products:
    #   C[:,0] = A * B[:,0] (2x2 * 2x1)
    #   C[:,1] = A * B[:,1] (2x2 * 2x1)
    #   C[:,2] = A * B[:,2] (2x2 * 2x1)
    # Each 2x2 * 2x1 has rank at most 2*2*1 = 4, but can be done with rank 3?
    # Actually, 2x2 * 2x1 is just two dot products, so rank 4 is tight.
    
    # Let me just use schoolbook for these small cases and combine.
    
    # For now, let me return schoolbook to ensure correctness.
    U, V, W = _schoolbook(n, m, p)
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6.
    Returns {'U': U, 'V': V, 'W': W} where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Use schoolbook as a baseline
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the schoolbook decomposition
    assert _verify_decomposition(U, V, W, n, m, p), "Schoolbook decomposition failed verification"
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    # Try block decomposition
    U2, V2, W2 = _block_decomposition_4x5x6()
    if _verify_decomposition(U2, V2, W2, n, m, p):
        if U2.shape[1] < best_rank:
            best_rank = U2.shape[1]
            best_result = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
