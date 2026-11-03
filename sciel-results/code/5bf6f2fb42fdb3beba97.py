import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135.0
    
    # Strassen components for 2x2x2
    # local_a: 0=A11, 1=A12, 2=A21, 3=A22
    # local_b: 0=B11, 1=B12, 2=B21, 3=B22
    # local_w: 0=C11, 1=C12, 2=C21, 3=C22
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),           # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7
    ]
    
    # Blocks: (a_rows, a_cols, b_rows, b_cols)
    # For 2x4 * 4x6:
    # A is 2x4, B is 4x6
    # We can decompose this into blocks:
    # Top-left 2x2 block of A * Top-left 2x2 block of B (cols 0,1)
    # Top-left 2x2 block of A * Middle 2x2 block of B (cols 2,3)
    # Top-left 2x2 block of A * Bottom 2x2 block of B (cols 4,5)
    # Bottom-left 2x2 block of A * Top-left 2x2 block of B (cols 0,1)
    # Bottom-left 2x2 block of A * Middle 2x2 block of B (cols 2,3)
    # Bottom-left 2x2 block of A * Bottom 2x2 block of B (cols 4,5)
    # Note: A is 2x4, so we split A into two 2x2 blocks (top rows 0,1; bottom rows 0,1)
    # Wait, A is 2x4. So A has rows [0,1] and cols [0,1,2,3].
    # We can split A into:
    #   A_top: rows [0,1], cols [0,1]
    #   A_bottom: rows [0,1], cols [2,3]
    # B is 4x6. Rows [0,1,2,3], cols [0,1,2,3,4,5].
    # We can split B into:
    #   B1: rows [0,1], cols [0,1]
    #   B2: rows [0,1], cols [2,3]
    #   B3: rows [0,1], cols [4,5]
    #   B4: rows [2,3], cols [0,1]
    #   B5: rows [2,3], cols [2,3]
    #   B6: rows [2,3], cols [4,5]
    # But Strassen works on 2x2 blocks. So we need 2x2 blocks from A and 2x2 blocks from B.
    # A_top is 2x2, A_bottom is 2x2.
    # B1 is 2x2, B2 is 2x2, B3 is 2x2, B4 is 2x2, B5 is 2x2, B6 is 2x2.
    # So we have 6 combinations:
    # (A_top, B1), (A_top, B2), (A_top, B3), (A_bottom, B4), (A_bottom, B5), (A_bottom, B6)
    # Wait, that's not right. The matrix product is A @ B.
    # C[i,j] = sum_k A[i,k] B[k,j]
    # C is 2x6.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] + A[0,3]*B[3,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2] + A[0,3]*B[3,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3] + A[0,2]*B[2,3] + A[0,3]*B[3,3]
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4] + A[0,3]*B[3,4]
    # C[0,5] = A[0,0]*B[0,5] + A[0,1]*B[1,5] + A[0,2]*B[2,5] + A[0,3]*B[3,5]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0] + A[1,3]*B[3,0]
    # ...
    
    # Let's group by 2x2 blocks of B (columns) and 2x2 blocks of A (columns).
    # B columns [0,1]: B1 = B[0:2, 0:2]
    # B columns [2,3]: B2 = B[0:2, 2:4]
    # B columns [4,5]: B3 = B[0:2, 4:6]
    # B rows [0,1] are used for A columns [0,1].
    # B rows [2,3] are used for A columns [2,3].
    
    # So:
    # C[0:2, 0:2] = A[0:2, 0:2] @ B[0:2, 0:2] + A[0:2, 2:4] @ B[2:4, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2] @ B[0:2, 2:4] + A[0:2, 2:4] @ B[2:4, 2:4]
    # C[0:2, 4:6] = A[0:2, 0:2] @ B[0:2, 4:6] + A[0:2, 2:4] @ B[2:4, 4:6]
    
    # Each of these is a sum of two 2x2 matrix multiplications.
    # We can use Strassen for each 2x2 multiplication.
    # But Strassen combines the multiplications.
    # Actually, we can just use Strassen for each 2x2 block multiplication.
    # There are 6 such multiplications:
    # 1. A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2] (part 1)
    # 2. A[0:2, 2:4] @ B[2:4, 0:2] -> C[0:2, 0:2] (part 2)
    # 3. A[0:2, 0:2] @ B[0:2, 2:4] -> C[0:2, 2:4] (part 1)
    # 4. A[0:2, 2:4] @ B[2:4, 2:4] -> C[0:2, 2:4] (part 2)
    # 5. A[0:2, 0:2] @ B[0:2, 4:6] -> C[0:2, 4:6] (part 1)
    # 6. A[0:2, 2:4] @ B[2:4, 4:6] -> C[0:2, 4:6] (part 2)
    
    # Each 2x2 multiplication uses 7 scalar multiplications (Strassen).
    # So total rank = 6 * 7 = 42.
    
    # The blocks are:
    # (a_rows, a_cols, b_rows, b_cols)
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1]),  # A[0:2,0:2] @ B[0:2,0:2]
        ([0, 1], [2, 3], [2, 3], [0, 1]),  # A[0:2,2:4] @ B[2:4,0:2]
        ([0, 1], [0, 1], [0, 1], [2, 3]),  # A[0:2,0:2] @ B[0:2,2:4]
        ([0, 1], [2, 3], [2, 3], [2, 3]),  # A[0:2,2:4] @ B[2:4,2:4]
        ([0, 1], [0, 1], [0, 1], [4, 5]),  # A[0:2,0:2] @ B[0:2,4:6]
        ([0, 1], [2, 3], [2, 3], [4, 5]),  # A[0:2,2:4] @ B[2:4,4:6]
    ]
    
    rank = len(blocks) * len(strassen)  # 6 * 7 = 42
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*6 = 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 6*2 = 12 rows
    
    col = 0
    for a_rows, a_cols, b_rows, b_cols in blocks:
        if not time.time() < deadline:
            break
        
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            if not time.time() < deadline:
                break
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx]][col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx]][col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx]][col] += coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
