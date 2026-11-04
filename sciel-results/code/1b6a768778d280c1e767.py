"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """Construct a rank-44 exact decomposition for 2x3 @ 3x8 matrix multiplication.

    Strategy: Split the 3 columns of A (and 3 rows of B) into a 2-wide part and a 1-wide part.
    For the 2x2x8 sub-problem, apply Strassen's algorithm per 2-column block of the 8 columns.
    For the 1x1x8 tail, use schoolbook (rank 8).
    
    Total rank = 4 * 7 (Strassen) + 16 (schoolbook for 2x1x8) = 28 + 16 = 44.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit

    n, m, p = 2, 3, 8

    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (16, rank)

    rank = 44
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]

    col = 0

    # Strassen decomposition for 2x2 matrices, applied per 2-column block of B.
    # A is 2x2 (rows i=0,1; cols j=0,1), B is 2x2 (rows j=0,1; cols k=0,1)
    # C = A*B
    # Strassen:
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A10)*B11
    # m6 = (A11-A10)*(B00+B10)
    # m7 = (A01-A11)*(B01+B11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6

    # For the 2x2x8 sub-problem, we split the 8 columns of B into 4 blocks of 2 columns each.
    # For each block (k_start, k_start+1), we apply Strassen.

    # U indices: row index = i*m + j where i in [0,1], j in [0,1] for the Strassen part
    # V indices: row index = j*p + k where j in [0,1], k in [0,7]
    # W indices: row index = k*n + i where k in [0,7], i in [0,1]

    # Strassen part: columns j=0,1 of A (which is columns 0,1 of the 3-column A)
    # A rows: i=0,1
    # B rows: j=0,1 (which correspond to the 2x2 part)
    # B cols: k=0,7

    for k_block in range(4):
        k0 = 2 * k_block
        k1 = 2 * k_block + 1

        # m1 = (A00+A11)*(B00+B11)
        # A00: U[0*3+0] = U[0], A11: U[1*3+1] = U[4]
        # B00: V[0*8+k0] = V[k0], B11: V[1*8+k1] = V[8+k1]
        # C00: W[k0*2+0] = W[2*k0], C11: W[k1*2+1] = W[2*k1+1]
        
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[8+k1][col] = 1
        W[2*k0][col] = 1; W[2*k1+1][col] = 1
        col += 1

        # m2 = (A10+A11)*B00
        # A10: U[1*3+0] = U[3], A11: U[4]
        # B00: V[k0]
        # C10: W[2*k0+1], C11: W[2*k1+1]
        
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[2*k0+1][col] = 1; W[2*k1+1][col] = -1
        col += 1

        # m3 = A00*(B01-B11)
        # A00: U[0]
        # B01: V[k1], B11: V[8+k1]
        # C01: W[2*k0+1], C11: W[2*k1+1]
        
        U[0][col] = 1
        V[k1][col] = 1; V[8+k1][col] = -1
        W[2*k0+1][col] = 1; W[2*k1+1][col] = 1
        col += 1

        # m4 = A11*(B10-B00)
        # A11: U[4]
        # B10: V[8+k0], B00: V[k0]
        # C00: W[2*k0], C10: W[2*k0+1]
        
        U[4][col] = 1
        V[8+k0][col] = 1; V[k0][col] = -1
        W[2*k0][col] = 1; W[2*k0+1][col] = 1
        col += 1

        # m5 = (A00+A10)*B11
        # A00: U[0], A10: U[3]
        # B11: V[8+k1]
        # C00: W[2*k0], C01: W[2*k0+1]
        
        U[0][col] = 1; U[3][col] = 1
        V[8+k1][col] = 1
        W[2*k0][col] = -1; W[2*k0+1][col] = 1
        col += 1

        # m6 = (A11-A10)*(B00+B10)
        # A11: U[4], A10: U[3]
        # B00: V[k0], B10: V[8+k0]
        # C11: W[2*k1+1]
        
        U[4][col] = 1; U[3][col] = -1
        V[k0][col] = 1; V[8+k0][col] = 1
        W[2*k1+1][col] = 1
        col += 1

        # m7 = (A01-A11)*(B01+B11)
        # A01: U[0*3+1] = U[1], A11: U[4]
        # B01: V[k1], B11: V[8+k1]
        # C00: W[2*k0], C01: W[2*k0+1]
        
        U[1][col] = 1; U[4][col] = -1
        V[k1][col] = 1; V[8+k1][col] = 1
        W[2*k0][col] = 1; W[2*k0+1][col] = -1
        col += 1

    # Now the schoolbook part for the remaining column j=2 of A.
    # A has rows i=0,1, so A02 and A12.
    # B has rows j=2, so B2k for k=0..7.
    # C has entries C_i,k = A_i2 * B_2k for i=0,1 and k=0..7.
    # U indices: U[0*3+2] = U[2], U[1*3+2] = U[5]
    # V indices: V[2*8+k] = V[16+k]
    # W indices: W[k*2+i]

    for i in range(2):
        for k in range(8):
            U[i*3+2][col] = 1
            V[16+k][col] = 1
            W[k*2+i][col] = 1
            col += 1

    # Convert Fraction to int where possible, else keep as float for the interface
    # The interface expects integer/half-integer entries.
    # Our entries are all integers (0, 1, -1).
    
    U_out = [[int(x) for x in row] for row in U]
    V_out = [[int(x) for x in row] for row in V]
    W_out = [[int(x) for x in row] for row in W]

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
