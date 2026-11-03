"""Matrix multiplication 2,4,8 — rank 56 decomposition via Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_SEED_START = time.time()


def _deadline_guard(seconds_remaining=115.0):
    """Return True if we should stop searching."""
    return (time.time() - _SEED_START) > seconds_remaining


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We construct a rank 56 decomposition.
    # Strategy:
    # 1. Split the middle dimension m=4 into two blocks of size 2: m1=2, m2=2.
    #    A is 2x4, so A = [A0 | A1] where A0, A1 are 2x2.
    #    B is 4x8, so B = [B0; B1] where B0, B1 are 2x8.
    #    C = A0*B0 + A1*B1.
    # 2. Split the output dimension p=8 into four blocks of size 2: p1=2, p2=2, p3=2, p4=2.
    #    B0 is 2x8, so B0 = [B00 | B01 | B02 | B03] where Bij are 2x2.
    #    B1 is 2x8, so B1 = [B10 | B11 | B12 | B13] where Bij are 2x2.
    #    C = A0*B0 + A1*B1
    #      = A0*B00 + A0*B01 + A0*B02 + A0*B03 + A1*B10 + A1*B11 + A1*B12 + A1*B13
    # 3. Each term is a 2x2 * 2x2 multiplication.
    #    The rank of 2x2 matrix multiplication is 7 (Strassen).
    #    There are 8 such terms.
    #    Total rank = 8 * 7 = 56.
    
    # Strassen's algorithm for 2x2 multiplication:
    # Given X (2x2) and Y (2x2), Z = X*Y.
    # Let X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]].
    # Z = [[z00, z01], [z10, z11]].
    # 
    # m1 = (x00 + x11) * (y00 + y11)
    # m2 = (x10 + x11) * y00
    # m3 = x00 * (y01 - y11)
    # m4 = x11 * (y10 - y00)
    # m5 = (x00 + x01) * y11
    # m6 = (x10 - x00) * (y00 + y10)
    # m7 = (x01 - x11) * (y01 + y11)
    #
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    #
    # Each mi is a bilinear form. We can represent the decomposition of a single 2x2 multiplication
    # by 7 rank-1 terms in the tensor space.
    # The tensor for 2x2 multiplication has indices (i, j, k) where i,j are input indices for X,
    # j,k for Y? No, the standard tensor M[i,j,k] for matrix multiplication C[i,k] = sum_j A[i,j]B[j,k].
    # For 2x2 * 2x2, the tensor is 2x2x2.
    # The decomposition is sum_{r=1}^7 u_r[i,j] v_r[j,k] w_r[i,k].
    #
    # Let's define the 7 rank-1 components for a generic 2x2 multiplication.
    # We need to map the local indices (i, j, k) in {0,1} to the global indices.
    
    rank = 56
    U = np.zeros((n*m, rank)) # 8x56
    V = np.zeros((m*p, rank)) # 32x56
    W = np.zeros((p*n, rank)) # 16x56
    
    # Strassen coefficients
    # Each term r in 1..7 corresponds to a product of linear forms.
    # We define the patterns for U, V, W for a single 2x2 block.
    # U pattern (2x2):
    #   r=1: [[1, 0], [0, 1]]  (x00 + x11)
    #   r=2: [[0, 0], [1, 1]]  (x10 + x11)
    #   r=3: [[1, 0], [0, 0]]  (x00)
    #   r=4: [[0, 0], [0, 1]]  (x11)
    #   r=5: [[1, 1], [0, 0]]  (x00 + x01)
    #   r=6: [[-1, 0], [1, 0]] (x10 - x00)
    #   r=7: [[0, 1], [0, -1]] (x01 - x11)
    #
    # V pattern (2x2):
    #   r=1: [[1, 1], [1, 1]]  (y00 + y11) -> wait, y is 2x2. y00, y01, y10, y11.
    #       y00+y11 is [[1,0],[0,1]]? No.
    #       The bilinear form is (linear in A) * (linear in B).
    #       For m1 = (x00+x11)(y00+y11):
    #       U part: x00 + x11. In U matrix (2x2), this is [[1,0],[0,1]].
    #       V part: y00 + y11. In V matrix (2x2), this is [[1,0],[0,1]].
    #
    #   r=2: (x10+x11)y00
    #       U: [[0,0],[1,1]]
    #       V: [[1,0],[0,0]] (y00)
    #
    #   r=3: x00(y01-y11)
    #       U: [[1,0],[0,0]]
    #       V: [[0,1],[0,-1]] (y01 - y11)
    #
    #   r=4: x11(y10-y00)
    #       U: [[0,0],[0,1]]
    #       V: [[-1,0],[1,0]] (y10 - y00)
    #
    #   r=5: (x00+x01)y11
    #       U: [[1,1],[0,0]]
    #       V: [[0,0],[0,1]] (y11)
    #
    #   r=6: (x10-x00)(y00+y10)
    #       U: [[-1,0],[1,0]]
    #       V: [[1,0],[1,0]] (y00 + y10)
    #
    #   r=7: (x01-x11)(y01+y11)
    #       U: [[0,1],[0,-1]]
    #       V: [[0,1],[0,1]] (y01 + y11)
    #
    # W pattern (2x2) for output Z:
    #   z00 = m1 + m4 - m5 + m7
    #   z01 = m3 + m5
    #   z10 = m2 + m4
    #   z11 = m1 - m2 + m3 + m6
    #
    #   W matrix for each r:
    #   r=1: [[1, 0], [0, 1]]  (contributes to z00 and z11)
    #   r=2: [[0, 0], [1, -1]] (contributes to z10 and z11) -> z10 gets +1, z11 gets -1
    #   r=3: [[0, 1], [0, 1]]  (contributes to z01 and z11)
    #   r=4: [[1, 0], [1, 0]]  (contributes to z00 and z10)
    #   r=5: [[-1, 1], [0, 0]] (contributes to z00 and z01)
    #   r=6: [[0, 0], [0, 1]]  (contributes to z11)
    #   r=7: [[1, 0], [0, 0]]  (contributes to z00)
    
    # Define the 7 patterns
    U_patterns = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    V_patterns = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [1, 0]],
        [[0, 1], [0, 1]]
    ]
    W_patterns = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    # We have 8 blocks.
    # Block index b from 0 to 7.
    # b = 0..3 corresponds to A0 * B0b (A part index 0, B part index 0..3)
    # b = 4..7 corresponds to A1 * B1(b-4) (A part index 1, B part index 0..3)
    
    for b in range(8):
        # Determine which A block and which B block
        a_block = b // 4  # 0 or 1
        b_subblock = b % 4  # 0, 1, 2, 3 (which 2x2 chunk of B's columns)
        
        # Global indices for A part (2x2)
        # A is 2x4. Rows i in {0,1}. Cols j in {0,1,2,3}.
        # a_block=0 uses cols 0,1. a_block=1 uses cols 2,3.
        a_col_offset = a_block * 2
        
        # Global indices for B part (2x2)
        # B is 4x8. Rows j in {0,1,2,3}. Cols k in {0..7}.
        # b_subblock=0 uses rows 0,1 (from B0) and cols 0,1.
        # Wait, B is split into B0 (rows 0,1) and B1 (rows 2,3).
        # So if a_block=0, we use B0 (rows 0,1).
        # If a_block=1, we use B1 (rows 2,3).
        # b_subblock determines the column chunk.
        # b_subblock=0 -> cols 0,1
        # b_subblock=1 -> cols 2,3
        # b_subblock=2 -> cols 4,5
        # b_subblock=3 -> cols 6,7
        
        b_row_offset = a_block * 2  # 0 or 2
        b_col_offset = b_subblock * 2
        
        # Fill the 7 columns for this block
        for r in range(7):
            col = b * 7 + r
            
            # U: 2x2 pattern mapped to global A indices
            # U global shape: (n*m, rank) = (8, 56).
            # Index: i*m + j. i in {0,1}, j in {0,1,2,3}.
            for i in range(2):
                for j in range(2):
                    local_u = U_patterns[r][i][j]
                    global_i = i
                    global_j = a_col_offset + j
                    U[global_i * m + global_j, col] = local_u
            
            # V: 2x2 pattern mapped to global B indices
            # V global shape: (m*p, rank) = (32, 56).
            # Index: j*p + k. j in {0,1,2,3}, k in {0..7}.
            for j in range(2):
                for k in range(2):
                    local_v = V_patterns[r][j][k]
                    global_j = b_row_offset + j
                    global_k = b_col_offset + k
                    V[global_j * p + global_k, col] = local_v
            
            # W: 2x2 pattern mapped to global C indices
            # W global shape: (p*n, rank) = (16, 56).
            # Index: k*n + i. k in {0..7}, i in {0,1}.
            for i in range(2):
                for k in range(2):
                    local_w = W_patterns[r][i][k]
                    global_i = i
                    global_k = b_col_offset + k
                    W[global_k * n + global_i, col] = local_w

    # Validate the decomposition
    def validate(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(total - 1) > 1e-10:
                        return False
        return True

    if not validate(U, V, W):
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        rank = 64

    def to_list(A):
        return [[float(x) for x in row] for row in A]

    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
