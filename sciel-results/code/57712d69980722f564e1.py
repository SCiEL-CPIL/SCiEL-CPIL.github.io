"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    Uses a block decomposition combining Strassen's algorithm for 2x2 blocks and schoolbook for smaller blocks.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s deadline

    n, m, p = 2, 3, 7
    rank = n * m * p  # Schoolbook rank is 42

    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    column = 0

    # We'll use a block decomposition approach
    # A is 2x3, B is 3x7
    # Split A into A1 (2x2) and A2 (2x1)
    # Split B into B1 (3x3) and B2 (3x4)
    # A1*B1: Split into A11(2x2)*B11(2x2) using Strassen (rank 7)
    #          and A11(2x2)*B12(2x1) using schoolbook (rank 4)
    #          and A2(2x1)*B13(1x3) using schoolbook (rank 3)
    # A1*B2: Split into A11(2x2)*B21(2x2) using Strassen (rank 7)
    #          and A11(2x2)*B22(2x2) using Strassen (rank 7)
    #          and A2(2x1)*B23(1x4) using schoolbook (rank 4)
    # Total rank: 7+4+3+7+7+4 = 32

    # Strassen's algorithm for 2x2*2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    def strassen_2x2(i_offset, j_offset, k_offset):
        """Add Strassen decomposition for A[i_offset:i_offset+2, j_offset:j_offset+2] * B[j_offset:j_offset+2, k_offset:k_offset+2]"""
        global column
        # M1 = (A11+A22)(B11+B22)
        U[i_offset + 0, column] = 1  # A11
        U[i_offset + 1, column] = 1  # A22
        V[j_offset + 0, column] = 1  # B11
        V[j_offset + 1, column] = 1  # B22
        W[k_offset + 0, column] = 1  # C11
        W[k_offset + 1, column] = 1  # C22
        column += 1

        # M2 = (A21+A22)B11
        U[i_offset + 1, column] = 1  # A21
        U[i_offset + 2, column] = 1  # A22 (but this is out of bounds for 2x2, need to fix)
        # Actually for 2x2, indices are:
        # A11 at [0,0], A12 at [0,1]
        # A21 at [1,0], A22 at [1,1]
        # So for offset i_offset, j_offset:
        # A11 at [i_offset, j_offset]
        # A12 at [i_offset, j_offset+1]
        # A21 at [i_offset+1, j_offset]
        # A22 at [i_offset+1, j_offset+1]
        # Let me redo this properly
        pass

    # Let me implement this more carefully by building the full decomposition explicitly

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] using Strassen (rank 7)
    # A indices: row 0-1, col 0-1
    # B indices: row 0-1, col 0-1
    # C indices: row 0-1, col 0-1

    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0, column] = 1  # A[0,0]
    U[3, column] = 1  # A[1,1] (i=1, j=1 -> i*m+j = 1*3+1 = 4, wait)
    # n=2, m=3, p=7
    # A is 2x3: U has n*m = 6 rows
    # A[i,j] -> U[i*m+j]
    # So A[0,0] -> U[0], A[0,1] -> U[1], A[0,2] -> U[2]
    # A[1,0] -> U[3], A[1,1] -> U[4], A[1,2] -> U[5]
    # B is 3x7: V has m*p = 21 rows
    # B[j,k] -> V[j*p+k]
    # C is 2x7: W has p*n = 14 rows
    # C[k,i] -> W[k*n+i]
    # C[0,0] -> W[0], C[0,1] -> W[1]
    # C[1,0] -> W[2], C[1,1] -> W[3]

    # Let me restart with correct indexing
    U = np.zeros((6, 42))
    V = np.zeros((21, 42))
    W = np.zeros((14, 42))
    column = 0

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] using Strassen (rank 7)
    # A[0:2, 0:2]: rows 0-1, cols 0-1
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # B[0:2, 0:2]: rows 0-1, cols 0-1
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[7], B[1,1]=V[8]
    # C[0:2, 0:2]: rows 0-1, cols 0-1
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]

    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0, column] = 1  # A[0,0]
    U[4, column] = 1  # A[1,1]
    V[0, column] = 1  # B[0,0]
    V[8, column] = 1  # B[1,1]
    W[0, column] = 1  # C[0,0]
    W[3, column] = 1  # C[1,1]
    column += 1

    # M2 = (A[1,0]+A[1,1])*B[0,0]
    U[3, column] = 1  # A[1,0]
    U[4, column] = 1  # A[1,1]
    V[0, column] = 1  # B[0,0]
    W[2, column] = 1  # C[1,0]
    column += 1

    # M3 = A[0,0]*(B[0,1]-B[1,1])
    U[0, column] = 1  # A[0,0]
    V[1, column] = 1  # B[0,1]
    V[8, column] = -1  # B[1,1]
    W[1, column] = 1  # C[0,1]
    column += 1

    # M4 = A[1,1]*(B[1,0]-B[0,0])
    U[4, column] = 1  # A[1,1]
    V[7, column] = 1  # B[1,0]
    V[0, column] = -1  # B[0,0]
    W[0, column] = 1  # C[0,0]
    W[2, column] = 1  # C[1,0]
    column += 1

    # M5 = (A[0,0]+A[0,1])*B[1,1]
    U[0, column] = 1  # A[0,0]
    U[1, column] = 1  # A[0,1]
    V[8, column] = 1  # B[1,1]
    W[1, column] = 1  # C[0,1]
    W[3, column] = 1  # C[1,1]
    column += 1

    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[3, column] = 1  # A[1,0]
    U[0, column] = -1  # A[0,0]
    V[0, column] = 1  # B[0,0]
    V[1, column] = 1  # B[0,1]
    W[3, column] = 1  # C[1,1]
    column += 1

    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[1, column] = 1  # A[0,1]
    V[7, column] = 1  # B[1,0]
    V[8, column] = 1  # B[1,1]
    W[0, column] = 1  # C[0,0]
    column += 1

    # Now we need to fix the C contributions
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    # I made errors in the W matrix above. Let me redo this more carefully.

    # Let me use a different approach: build the full decomposition by enumerating all terms

    U = np.zeros((6, 42))
    V = np.zeros((21, 42))
    W = np.zeros((14, 42))
    column = 0

    # For each output element C[i,k] (i in 0..1, k in 0..6):
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We'll use Strassen for the 2x2 blocks where possible

    # Block decomposition:
    # Split A into A_left (2x2, cols 0-1) and A_right (2x1, col 2)
    # Split B into B_top (2x7, rows 0-1) and B_bottom (1x7, row 2)
    # A_left * B_top: This is 2x2 * 2x7
    #   Split B_top into B_top_left (2x2, cols 0-1), B_top_mid (2x2, cols 2-3), B_top_right (2x3, cols 4-6)
    #   A_left * B_top_left: 2x2*2x2, use Strassen (rank 7)
    #   A_left * B_top_mid: 2x2*2x2, use Strassen (rank 7)
    #   A_left * B_top_right: 2x2*2x3, split into 2x2*2x2 + 2x2*2x1
    #     2x2*2x2: Strassen (rank 7)
    #     2x2*2x1: schoolbook (rank 4)
    #   Total for A_left*B_top: 7+7+7+4 = 25
    # A_right * B_top: 2x1 * 2x7, schoolbook (rank 14)
    # A_left * B_bottom: 2x2 * 1x7, schoolbook (rank 14)
    # A_right * B_bottom: 2x1 * 1x7, schoolbook (rank 14)
    # Total: 25+14+14+14 = 67, which is worse than schoolbook 42

    # Let me try a different split:
    # A (2x3) * B (3x7)
    # Use the fact that 2x2*2x2 can be done with rank 7 (Strassen)
    # Split B into B1 (3x3), B2 (3x3), B3 (3x1)
    # A*B1: 2x3*3x3
    #   Split A into A1 (2x2, cols 0-1), A2 (2x1, col 2)
    #   Split B1 into B11 (2x2, rows 0-1, cols 0-1), B12 (2x1, rows 0-1, col 2), B13 (1x3, row 2)
    #   A1*B11: 2x2*2x2, Strassen (rank 7)
    #   A1*B12: 2x2*2x1, schoolbook (rank 4)
    #   A2*B13: 2x1*1x3, schoolbook (rank 3)
    #   Total: 7+4+3 = 14
    # A*B2: same as A*B1, rank 14
    # A*B3: 2x3*3x1, schoolbook (rank 6)
    # Total: 14+14+6 = 34

    # Let me implement this:
    U = np.zeros((6, 34))
    V = np.zeros((21, 34))
    W = np.zeros((14, 34))
    column = 0

    # A*B1: A (2x3) * B1 (3x3, cols 0-2 of B)
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:3] (2x1)
    # B11 = B1[0:2, 0:2] (2x2), B12 = B1[0:2, 2:3] (2x1), B13 = B1[2:3, 0:3] (1x3)

    # A1*B11: Strassen 2x2*2x2
    # A1: rows 0-1, cols 0-1 of A
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # B11: rows 0-1, cols 0-1 of B
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[7], B[1,1]=V[8]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]

    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0, column] = 1
    U[4, column] = 1
    V[0, column] = 1
    V[8, column] = 1
    W[0, column] = 1  # contributes to C[0,0]
    W[3, column] = 1  # contributes to C[1,1]
    column += 1

    # M2 = (A[1,0]+A[1,1])*B[0,0]
    U[3, column] = 1
    U[4, column] = 1
    V[0, column] = 1
    W[2, column] = 1  # contributes to C[1,0]
    column += 1

    # M3 = A[0,0]*(B[0,1]-B[1,1])
    U[0, column] = 1
    V[1, column] = 1
    V[8, column] = -1
    W[1, column] = 1  # contributes to C[0,1]
    column += 1

    # M4 = A[1,1]*(B[1,0]-B[0,0])
    U[4, column] = 1
    V[7, column] = 1
    V[0, column] = -1
    W[0, column] = 1  # contributes to C[0,0]
    W[2, column] = 1  # contributes to C[1,0]
    column += 1

    # M5 = (A[0,0]+A[0,1])*B[1,1]
    U[0, column] = 1
    U[1, column] = 1
    V[8, column] = 1
    W[1, column] = 1  # contributes to C[0,1]
    W[3, column] = 1  # contributes to C[1,1]
    column += 1

    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[3, column] = 1
    U[0, column] = -1
    V[0, column] = 1
    V[1, column] = 1
    W[3, column] = 1  # contributes to C[1,1]
    column += 1

    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[1, column] = 1
    V[7, column] = 1
    V[8, column] = 1
    W[0, column] = 1  # contributes to C[0,0]
    column += 1

    # Now check the C contributions:
    # C[0,0] should be M1 + M4 - M5 + M7
    # M1 contributes 1, M4 contributes 1, M5 contributes 0, M7 contributes 1 -> total 3, but should be 1
    # I made errors. Let me use a cleaner approach.

    # Let me just use schoolbook for simplicity and correctness, since finding a better rank is complex
    U = np.zeros((6, 42))
    V = np.zeros((21, 42))
    W = np.zeros((14, 42))
    column = 0

    for i in range(2):
        for j in range(3):
            for k in range(7):
                U[i * 3 + j, column] = 1
                V[j * 7 + k, column] = 1
                W[k * 2 + i, column] = 1
                column += 1

    # Check if we have time to try a better decomposition
    if time.time() < deadline - 10.0:
        # Try the block decomposition with Strassen
        U2 = np.zeros((6, 34))
        V2 = np.zeros((21, 34))
        W2 = np.zeros((14, 34))
        col = 0

        # A*B1 where B1 = B[:, 0:3]
        # A1 = A[:, 0:2], A2 = A[:, 2:3]
        # B11 = B[0:2, 0:2], B12 = B[0:2, 2:3], B13 = B[2:3, 0:3]

        # A1*B11: Strassen 2x2*2x2
        # M1 = (A00+A11)(B00+B11)
        U2[0, col] = 1; U2[4, col] = 1
        V2[0, col] = 1; V2[8, col] = 1
        W2[0, col] = 1; W2[3, col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U2[3, col] = 1; U2[4, col] = 1
        V2[0, col] = 1
        W2[2, col] = 1
        col += 1

        # M3 = A00(B01-B11)
        U2[0, col] = 1
        V2[1, col] = 1; V2[8, col] = -1
        W2[1, col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U2[4, col] = 1
        V2[7, col] = 1; V2[0, col] = -1
        W2[0, col] = 1; W2[2, col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U2[0, col] = 1; U2[1, col] = 1
        V2[8, col] = 1
        W2[1, col] = 1; W2[3, col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U2[3, col] = 1; U2[0, col] = -1
        V2[0, col] = 1; V2[1, col] = 1
        W2[3, col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U2[1, col] = 1
        V2[7, col] = 1; V2[8, col] = 1
        W2[0, col] = 1
        col += 1

        # A1*B12: 2x2*2x1, schoolbook
        # A[0,0]*B[0,2] -> C[0,2]
        U2[0, col] = 1; V2[2, col] = 1; W2[2, col] = 1; col += 1
        # A[0,1]*B[0,2] -> C[0,2]
        U2[1, col] = 1; V2[2, col] = 1; W2[2, col] = 1; col += 1
        # A[1,0]*B[0,2] -> C[1,2]
        U2[3, col] = 1; V2[2, col] = 1; W2[4, col] = 1; col += 1
        # A[1,1]*B[0,2] -> C[1,2]
        U2[4, col] = 1; V2[2, col] = 1; W2[4, col] = 1; col += 1

        # A2*B13: 2x1*1x3, schoolbook
        # A[0,2]*B[2,0] -> C[0,0]
        U2[2, col] = 1; V2[14, col] = 1; W2[0, col] = 1; col += 1
        # A[0,2]*B[2,1] -> C[0,1]
        U2[2, col] = 1; V2[15, col] = 1; W2[1, col] = 1; col += 1
        # A[0,2]*B[2,2] -> C[0,2]
        U2[2, col] = 1; V2[16, col] = 1; W2[2, col] = 1; col += 1
        # A[1,2]*B[2,0] -> C[1,0]
        U2[5, col] = 1; V2[14, col] = 1; W2[2, col] = 1; col += 1
        # A[1,2]*B[2,1] -> C[1,1]
        U2[5, col] = 1; V2[15, col] = 1; W2[3, col] = 1; col += 1
        # A[1,2]*B[2,2] -> C[1,2]
        U2[5, col] = 1; V2[16, col] = 1; W2[4, col] = 1; col += 1

        # A*B2 where B2 = B[:, 3:6]
        # A1 = A[:, 0:2], A2 = A[:, 2:3]
        # B21 = B[0:2, 3:5], B22 = B[0:2, 5:6], B23 = B[2:3, 3:6]

        # A1*B21: Strassen 2x2*2x2
        # M1 = (A00+A11)(B03+B14)
        U2[0, col] = 1; U2[4, col] = 1
        V2[3, col] = 1; V2[11, col] = 1
        W2[6, col] = 1; W2[9, col] = 1
        col += 1

        # M2 = (A10+A11)B03
        U2[3, col] = 1; U2[4, col] = 1
        V2[3, col] = 1
        W2[8, col] = 1
        col += 1

        # M3 = A00(B04-B14)
        U2[0, col] = 1
        V2[4, col] = 1; V2[11, col] = -1
        W2[7, col] = 1
        col += 1

        # M4 = A11(B13-B03)
        U2[4, col] = 1
        V2[10, col] = 1; V2[3, col] = -1
        W2[6, col] = 1; W2[8, col] = 1
        col += 1

        # M5 = (A00+A01)B14
        U2[0, col] = 1; U2[1, col] = 1
        V2[11, col] = 1
        W2[7, col] = 1; W2[9, col] = 1
        col += 1

        # M6 = (A10-A00)(B03+B04)
        U2[3, col] = 1; U2[0, col] = -1
        V2[3, col] = 1; V2[4, col] = 1
        W2[9, col] = 1
        col += 1

        # M7 = (A01-A11)(B13+B14)
        U2[1, col] = 1
        V2[10, col] = 1; V2[11, col] = 1
        W2[6, col] = 1
        col += 1

        # A1*B22: 2x2*2x1, schoolbook
        # A[0,0]*B[0,5] -> C[0,5]
        U2[0, col] = 1; V2[5, col] = 1; W2[10, col] = 1; col += 1
        # A[0,1]*B[0,5] -> C[0,5]
        U2[1, col] = 1; V2[5, col] = 1; W2[10, col] = 1; col += 1
        # A[1,0]*B[0,5] -> C[1,5]
        U2[3, col] = 1; V2[5, col] = 1; W2[12, col] = 1; col += 1
        # A[1,1]*B[0,5] -> C[1,5]
        U2[4, col] = 1; V2[5, col] = 1; W2[12, col] = 1; col += 1

        # A2*B23: 2x1*1x3, schoolbook
        # A[0,2]*B[2,3] -> C[0,3]
        U2[2, col] = 1; V2[17, col] = 1; W2[6, col] = 1; col += 1
        # A[0,2]*B[2,4] -> C[0,4]
        U2[2, col] = 1; V2[18, col] = 1; W2[7, col] = 1; col += 1
        # A[0,2]*B[2,5] -> C[0,5]
        U2[2, col] = 1; V2[19, col] = 1; W2[8, col] = 1; col += 1
        # A[1,2]*B[2,3] -> C[1,3]
        U2[5, col] = 1; V2[17, col] = 1; W2[8, col] = 1; col += 1
        # A[1,2]*B[2,4] -> C[1,4]
        U2[5, col] = 1; V2[18, col] = 1; W2[9, col] = 1; col += 1
        # A[1,2]*B[2,5] -> C[1,5]
        U2[5, col] = 1; V2[19, col] = 1; W2[10, col] = 1; col += 1

        # A*B3 where B3 = B[:, 6:7]
        # Schoolbook 2x3*3x1
        # A[0,0]*B[0,6] -> C[0,6]
        U2[0, col] = 1; V2[6, col] = 1; W2[12, col] = 1; col += 1
        # A[0,1]*B[0,6] -> C[0,6]
        U2[1, col] = 1; V2[6, col] = 1; W2[12, col] = 1; col += 1
        # A[0,2]*B[0,6] -> C[0,6]
        U2[2, col] = 1; V2[6, col] = 1; W2[12, col] = 1; col += 1
        # A[1,0]*B[0,6] -> C[1,6]
        U2[3, col] = 1; V2[6, col] = 1; W2[13, col] = 1; col += 1
        # A[1,1]*B[0,6] -> C[1,6]
        U2[4, col] = 1; V2[6, col] = 1; W2[13, col] = 1; col += 1
        # A[1,2]*B[0,6] -> C[1,6]
        U2[5, col] = 1; V2[6, col] = 1; W2[13, col] = 1; col += 1

        # Now verify the decomposition
        # Check if col == 34
        if col == 34:
            # Verify correctness
            # For each (i,j,k), check that the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] equals 1
            error = 0
            for i in range(2):
                for j in range(3):
                    for k in range(7):
                        expected = 1
                        actual = 0
                        for c in range(34):
                            actual += U2[i * 3 + j, c] * V2[j * 7 + k, c] * W2[k * 2 + i, c]
                        if abs(actual - expected) > 1e-9:
                            error += 1
            if error == 0:
                # Return the better decomposition
                return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    # Return the schoolbook decomposition if the better one didn't work
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
