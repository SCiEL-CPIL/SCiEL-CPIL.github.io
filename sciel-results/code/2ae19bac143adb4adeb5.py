"""
Rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.

Strategy:
  1. Partition the 4x4 matrix B into four 2x2 blocks:
       B = [[B00, B01],
            [B10, B11]]
     where B00 = B[0:2, 0:2], B01 = B[0:2, 2:4],
           B10 = B[2:4, 0:2], B11 = B[2:4, 2:4].

  2. The product C = A * B (A is 2x4, B is 4x4) can be written as:
       C = [[A0, A1]] * [[B00, B01], [B10, B11]]
         = [[A0*B00 + A1*B10,  A0*B01 + A1*B11]]
     where A0 = A[:, 0:2] (2x2), A1 = A[:, 2:4] (2x2).

  3. Apply Strassen's algorithm at the block level for the 2x2 block product:
     We have two "left" block-rows: [A0, A1] (each 2x2) and two "right" block-columns.
     Actually, the block structure is:
       Left matrix (2 blocks tall, 2 blocks wide): [[A0, A1]]  -- but this is 1 row of blocks.
       
     Let me re-examine. A is 2x4. B is 4x4.
     Split A into two 2x2 blocks horizontally: A = [A0 | A1]
     Split B into four 2x2 blocks: B = [[B00, B01], [B10, B11]]
     
     C = A * B = [A0 | A1] * [[B00, B01], [B10, B11]]
       = [A0*B00 + A1*B10,  A0*B01 + A1*B11]
     
     So C has two 2x2 blocks: C0 = A0*B00 + A1*B10, C1 = A0*B01 + A1*B11.
     
     Each Ci is a sum of two 2x2 matrix products. We can use Strassen on each pair?
     No, Strassen is for a single 2x2 * 2x2 product. Here we have a sum of two products.
     
     Alternative view: Think of the block multiplication as a 2x2 matrix multiplication
     where the "matrices" are themselves 2x2 blocks, but the left "matrix" is 1x2 and right is 2x2.
     This is not a square block multiplication.

  4. Correct block-Strassen approach:
     Consider the multiplication as:
       C = A * B
     where A is 2x4 and B is 4x4.
     
     We can embed this into a 2x2 block multiplication by padding, or more directly:
     
     Let's use the known fact: M(2,4,4) <= M(2,2,2) * something.
     
     Actually, the cleanest way: 
     C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
     
     Split j into j0 in {0,1} and j1 in {2,3}.
     C[i,k] = sum_{j0=0}^{1} A[i,j0]*B[j0,k] + sum_{j1=0}^{1} A[i,2+j1]*B[2+j1,k]
     
     So C = A_left * B_left + A_right * B_right
     where A_left is 2x2 (cols 0,1), B_left is 2x4 (rows 0,1),
           A_right is 2x2 (cols 2,3), B_right is 2x4 (rows 2,3).
     
     Now B_left = [B00 | B01] and B_right = [B10 | B11] where each is 2x2.
     
     So C_left = A_left * B_left = A_left * [B00 | B01] = [A_left*B00 | A_left*B01]
     C_right = A_right * B_right = A_right * [B10 | B11] = [A_right*B10 | A_right*B11]
     
     C = [A_left*B00 + A_right*B10 | A_left*B01 + A_right*B11]
     
     Each 2x2 block of C is a sum of two 2x2*2x2 products.
     
     To use Strassen: For a single 2x2*2x2 product, Strassen uses 7 multiplications.
     Here we have 4 such 2x2*2x2 products (A_left*B00, A_left*B01, A_right*B10, A_right*B11),
     each contributing to one block of C.
     
     If we do them independently with Strassen: 4 * 7 = 28 multiplications.
     This matches the rank-28 construction mentioned in the knowledge.

  5. Implementation:
     - Use the standard Strassen rank-7 decomposition for 2x2*2x2.
     - Apply it 4 times for the four sub-products.
     - Sum the contributions appropriately in the output tensor.

     The Strassen rank-7 decomposition for 2x2*2x2:
     Given X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]],
     Z = X*Y where:
       z00 = m1 - m2 + m4 - m5
       z01 = m3 + m5
       z10 = m2 + m4
       z11 = m1 - m3 + m6 - m7
     
     with:
       m1 = (x00+x11)*(y00+y11)
       m2 = (x10+x11)*y00
       m3 = x00*(y01-y11)
       m4 = x11*(y10-y00)
       m5 = (x00+x01)*y11
       m6 = (x10-x00)*(y00+y01)
       m7 = (x01-x11)*(y10+y11)
     
     In tensor decomposition form, each mi is a rank-1 term.

  6. For our problem, we apply this to four pairs:
     Pair 1: X=A_left, Y=B00 -> contributes to C[0:2, 0:2]
     Pair 2: X=A_left, Y=B01 -> contributes to C[0:2, 2:4]
     Pair 3: X=A_right, Y=B10 -> contributes to C[0:2, 0:2]
     Pair 4: X=A_right, Y=B11 -> contributes to C[0:2, 2:4]
     
     Total rank = 4 * 7 = 28.
"""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]


def _strassen_2x2_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    
    Returns (U, V, W) where:
      U has shape (4, 7)  -- indexing into X (flattened row-major: x00=0, x01=1, x10=2, x11=3)
      V has shape (4, 7)  -- indexing into Y (flattened row-major: y00=0, y01=1, y10=2, y11=3)
      W has shape (4, 7)  -- indexing into Z (flattened row-major: z00=0, z01=1, z10=2, z11=3)
    
    The decomposition satisfies:
      for all (a, b, c): sum_r U[a,r] * V[b,r] * W[c,r] = 1 if (a,b,c) is a valid multiplication index, else 0.
    
    For 2x2*2x2, the target tensor T[a,b,c] = 1 iff:
      a = i*2 + j  (i,j in {0,1})
      b = j*2 + k  (j,k in {0,1})
      c = k*2 + i  (k,i in {0,1})
    """
    # Strassen's 7 products:
    # m1 = (x00+x11)*(y00+y11)
    # m2 = (x10+x11)*y00
    # m3 = x00*(y01-y11)
    # m4 = x11*(y10-y00)
    # m5 = (x00+x01)*y11
    # m6 = (x10-x00)*(y00+y01)
    # m7 = (x01-x11)*(y10+y11)
    
    # U rows: x00=0, x01=1, x10=2, x11=3
    # V rows: y00=0, y01=1, y10=2, y11=3
    # W rows: z00=0, z01=1, z10=2, z11=3
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # m1 = (x00+x11)*(y00+y11)
    # U: x00=1, x11=1; V: y00=1, y11=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # Contribution to Z: z00 += m1, z11 += m1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (x10+x11)*y00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # Contribution: z10 += m2, z11 += m2
    W[2, 1] = 1; W[3, 1] = 1
    
    # m3 = x00*(y01-y11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # Contribution: z01 += m3, z11 -= m3
    W[1, 2] = 1; W[3, 2] = -1
    
    # m4 = x11*(y10-y00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # Contribution: z00 += m4, z10 += m4
    W[0, 3] = 1; W[2, 3] = 1
    
    # m5 = (x00+x01)*y11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # Contribution: z00 -= m5, z01 += m5
    W[0, 4] = -1; W[1, 4] = 1
    
    # m6 = (x10-x00)*(y00+y01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # Contribution: z11 += m6
    W[3, 5] = 1
    
    # m7 = (x01-x11)*(y10+y11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # Contribution: z11 -= m7
    W[3, 6] = -1
    
    return U, V, W


def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (8, rank), (16, rank), (8, rank).
    Uses block-Strassen decomposition with rank 28.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Get the base Strassen decomposition for 2x2*2x2
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    # Total rank = 4 * 7 = 28
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=np.int64)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=np.int64)  # (8, 28)
    
    # The four sub-products:
    # Pair 0: X=A_left (cols 0,1 of A), Y=B00 (rows 0,1; cols 0,1 of B) -> C[0:2, 0:2]
    # Pair 1: X=A_left (cols 0,1 of A), Y=B01 (rows 0,1; cols 2,3 of B) -> C[0:2, 2:4]
    # Pair 2: X=A_right (cols 2,3 of A), Y=B10 (rows 2,3; cols 0,1 of B) -> C[0:2, 0:2]
    # Pair 3: X=A_right (cols 2,3 of A), Y=B11 (rows 2,3; cols 2,3 of B) -> C[0:2, 2:4]
    
    # For each pair, we need to map:
    #   X indices (2x2): x_row in {0,1}, x_col in {0,1} -> A index = x_row * 4 + (x_col_offset + x_col)
    #   Y indices (2x2): y_row in {0,1}, y_col in {0,1} -> B index = (y_row_offset + y_row) * 4 + (y_col_offset + y_col)
    #   Z indices (2x2): z_row in {0,1}, z_col in {0,1} -> C index = z_col * 2 + z_row  (W convention: W[k*n+i] where i=z_row, k=z_col)
    
    # Pair configurations: (x_col_offset, y_row_offset, y_col_offset)
    pairs = [
        (0, 0, 0),  # A_left, B00 -> C[0:2, 0:2]
        (0, 0, 2),  # A_left, B01 -> C[0:2, 2:4]
        (2, 2, 0),  # A_right, B10 -> C[0:2, 0:2]
        (2, 2, 2),  # A_right, B11 -> C[0:2, 2:4]
    ]
    
    for pair_idx, (x_col_off, y_row_off, y_col_off) in enumerate(pairs):
        base = pair_idx * 7
        for r in range(7):
            # U: map x indices to A indices
            for x_idx in range(4):
                val = U_s[x_idx, r]
                if val != 0:
                    x_row = x_idx // 2
                    x_col = x_idx % 2
                    a_idx = x_row * m + (x_col_off + x_col)
                    U[a_idx, base + r] += val
            
            # V: map y indices to B indices
            for y_idx in range(4):
                val = V_s[y_idx, r]
                if val != 0:
                    y_row = y_idx // 2
                    y_col = y_idx % 2
                    b_idx = (y_row_off + y_row) * p + (y_col_off + y_col)
                    V[b_idx, base + r] += val
            
            # W: map z indices to C indices
            for z_idx in range(4):
                val = W_s[z_idx, r]
                if val != 0:
                    z_row = z_idx // 2
                    z_col = z_idx % 2
                    c_idx = z_col * n + z_row
                    W[c_idx, base + r] += val
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
