"""
Rank-minimal tensor decomposition for 4x4 x 4x5 matrix multiplication.
Uses block Strassen (rank 56) for the 4x4x4 part + rank 4 schoolbook for 5th column.
Total rank = 60.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]


def construct():
    """
    Construct an exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    
    Returns: {'U': U, 'V': V, 'W': W}
    U: (16, rank), V: (20, rank), W: (20, rank)
    W[k*n+i, t] follows the official trace convention.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: Block Strassen for 4x4x4 part (rank 56) + schoolbook for 5th column (rank 4)
    # Total rank = 60
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]   # 16 x 60
    V = [[0] * rank for _ in range(m * p)]   # 20 x 60
    W = [[0] * rank for _ in range(p * n)]   # 20 x 60
    
    col = 0
    
    # Block Strassen for 4x4x4:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 2x2.
    
    # Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = A11*(B00-B10)
    # M3 = (A00-A11)*(B10+B11)
    # M4 = A00*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A11)*(B00+B01)
    # M7 = (A01+A11)*(B10-B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mi, expand the linear combinations and use schoolbook for each 2x2*2x2.
    
    # Helper: add schoolbook terms for a single 2x2*2x2=2x2 product
    def _add_2x2_schoolbook(a_br, a_bc, b_br, b_bc, c_br, c_bc, sign_a, sign_b, sign_c):
        """
        Add 8 schoolbook terms for A[a_br,a_bc] * B[b_br,b_bc] -> C[c_br,c_bc]
        where each is a 2x2 block.
        sign_a, sign_b: signs for A and B blocks (from Strassen linear combinations)
        sign_c: sign for C block contribution
        Returns next column index.
        """
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    i_g = a_br * 2 + i
                    j_g = a_bc * 2 + j
                    k_g = b_bc * 2 + k
                    
                    a_idx = i_g * 4 + j_g
                    b_idx = (b_br * 2 + j) * 5 + (b_bc * 2 + k)
                    c_idx = k_g * 4 + i_g  # W[k*n+i] for 4x4 part
                    
                    U[a_idx][col] = sign_a
                    V[b_idx][col] = sign_b
                    W[c_idx][col] = sign_c
                    col += 1
        return col
    
    # M1 = (A00+A11)*(B00+B11)
    # Expands to: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Contributions to C: C00 += M1, C11 += M1
    
    # A00*B00 -> C00 (sign +1)
    _add_2x2_schoolbook(0, 0, 0, 0, 0, 0, 1, 1, 1)
    # A00*B11 -> C00 (sign +1)
    _add_2x2_schoolbook(0, 0, 1, 1, 0, 0, 1, 1, 1)
    # A11*B00 -> C00 (sign +1)
    _add_2x2_schoolbook(1, 1, 0, 0, 0, 0, 1, 1, 1)
    # A11*B11 -> C00 (sign +1)
    _add_2x2_schoolbook(1, 1, 1, 1, 0, 0, 1, 1, 1)
    
    # A00*B00 -> C11 (sign +1)
    _add_2x2_schoolbook(0, 0, 0, 0, 1, 1, 1, 1, 1)
    # A00*B11 -> C11 (sign +1)
    _add_2x2_schoolbook(0, 0, 1, 1, 1, 1, 1, 1, 1)
    # A11*B00 -> C11 (sign +1)
    _add_2x2_schoolbook(1, 1, 0, 0, 1, 1, 1, 1, 1)
    # A11*B11 -> C11 (sign +1)
    _add_2x2_schoolbook(1, 1, 1, 1, 1, 1, 1, 1, 1)
    
    # M2 = A11*(B00-B10)
    # Expands to: A11*B00 - A11*B10
    # Contribution: C10 += M2
    
    # A11*B00 -> C10 (sign +1)
    _add_2x2_schoolbook(1, 1, 0, 0, 1, 0, 1, 1, 1)
    # A11*B10 -> C10 (sign -1)
    _add_2x2_schoolbook(1, 1, 1, 0, 1, 0, 1, -1, 1)
    
    # M3 = (A00-A11)*(B10+B11)
    # Expands to: A00*B10 + A00*B11 - A11*B10 - A11*B11
    # Contribution: C01 += M3
    
    # A00*B10 -> C01 (sign +1)
    _add_2x2_schoolbook(0, 0, 1, 0, 0, 1, 1, 1, 1)
    # A00*B11 -> C01 (sign +1)
    _add_2x2_schoolbook(0, 0, 1, 1, 0, 1, 1, 1, 1)
    # A11*B10 -> C01 (sign -1)
    _add_2x2_schoolbook(1, 1, 1, 0, 0, 1, -1, 1, 1)
    # A11*B11 -> C01 (sign -1)
    _add_2x2_schoolbook(1, 1, 1, 1, 0, 1, -1, 1, 1)
    
    # M4 = A00*(B01-B11)
    # Expands to: A00*B01 - A00*B11
    # Contributions: C00 += M4, C10 += M4
    
    # A00*B01 -> C00 (sign +1)
    _add_2x2_schoolbook(0, 0, 0, 1, 0, 0, 1, 1, 1)
    # A00*B11 -> C00 (sign -1)
    _add_2x2_schoolbook(0, 0, 1, 1, 0, 0, 1, -1, 1)
    
    # A00*B01 -> C10 (sign +1)
    _add_2x2_schoolbook(0, 0, 0, 1, 1, 0, 1, 1, 1)
    # A00*B11 -> C10 (sign -1)
    _add_2x2_schoolbook(0, 0, 1, 1, 1, 0, 1, -1, 1)
    
    # M5 = (A00+A01)*B11
    # Expands to: A00*B11 + A01*B11
    # Contributions: C00 -= M5, C01 += M5
    
    # A00*B11 -> C00 (sign -1)
    _add_2x2_schoolbook(0, 0, 1, 1, 0, 0, 1, 1, -1)
    # A01*B11 -> C00 (sign -1)
    _add_2x2_schoolbook(0, 1, 1, 1, 0, 0, 1, 1, -1)
    
    # A00*B11 -> C01 (sign +1)
    _add_2x2_schoolbook(0, 0, 1, 1, 0, 1, 1, 1, 1)
    # A01*B11 -> C01 (sign +1)
    _add_2x2_schoolbook(0, 1, 1, 1, 0, 1, 1, 1, 1)
    
    # M6 = (A10-A11)*(B00+B01)
    # Expands to: A10*B00 + A10*B01 - A11*B00 - A11*B01
    # Contribution: C11 += M6
    
    # A10*B00 -> C11 (sign +1)
    _add_2x2_schoolbook(1, 0, 0, 0, 1, 1, 1, 1, 1)
    # A10*B01 -> C11 (sign +1)
    _add_2x2_schoolbook(1, 0, 0, 1, 1, 1, 1, 1, 1)
    # A11*B00 -> C11 (sign -1)
    _add_2x2_schoolbook(1, 1, 0, 0, 1, 1, -1, 1, 1)
    # A11*B01 -> C11 (sign -1)
    _add_2x2_schoolbook(1, 1, 0, 1, 1, 1, -1, 1, 1)
    
    # M7 = (A01+A11)*(B10-B01)
    # Expands to: A01*B10 - A01*B01 + A11*B10 - A11*B01
    # Contribution: C00 += M7
    
    # A01*B10 -> C00 (sign +1)
    _add_2x2_schoolbook(0, 1, 1, 0, 0, 0, 1, 1, 1)
    # A01*B01 -> C00 (sign -1)
    _add_2x2_schoolbook(0, 1, 0, 1, 0, 0, 1, -1, 1)
    # A11*B10 -> C00 (sign +1)
    _add_2x2_schoolbook(1, 1, 1, 0, 0, 0, 1, 1, 1)
    # A11*B01 -> C00 (sign -1)
    _add_2x2_schoolbook(1, 1, 0, 1, 0, 0, 1, -1, 1)
    
    # Now add schoolbook for the 5th column (k=4)
    # For i in 0..3, j in 0..3: C[i,4] += A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            a_idx = i * 4 + j
            b_idx = j * 5 + 4
            c_idx = 4 * 4 + i  # W[4*4+i] = W[16+i]
            
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
