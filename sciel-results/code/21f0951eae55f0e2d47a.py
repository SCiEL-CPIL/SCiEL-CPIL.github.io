"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U (4x7), V (4x7), W (4x7) for 2x2x2 Strassen decomposition.
    U rows indexed by (i,j) for A[i,j]: row = i*2+j
    V rows indexed by (j,k) for B[j,k]: row = j*2+k
    W rows indexed by (k,i) for C[i,k]: row = k*2+i
    """
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U rows: A00=0, A01=1, A10=2, A11=3
    # V rows: B00=0, B01=1, B10=2, B11=3
    # W rows: C[0,0] -> row 0*2+0 = 0, C[0,1] -> row 1*2+0 = 2,
    #         C[1,0] -> row 0*2+1 = 1, C[1,1] -> row 1*2+1 = 3
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=np.int64)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ], dtype=np.int64)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0] row 0
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0] row 1
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1] row 2
        [1, -1, 1, 0, 0, 1, 0],   # C[1,1] row 3
    ], dtype=np.int64)
    return U, V, W


def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using block Strassen (rank 49).
    U: (16, 49), V: (16, 49), W: (16, 49)
    U row i*4+j = A[i,j]
    V row j*4+k = B[j,k]
    W row k*4+i = C[i,k]  (trace convention)
    """
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    u2, v2, w2 = _strassen_2x2()
    
    def a_row(i, j):
        return i * 4 + j
    
    def b_row(j, k):
        return j * 4 + k
    
    def w_row(k, i):
        return k * 4 + i
    
    # The 7 Strassen multiplications for 4x4x4:
    # Each is a 2x2x2 multiplication applied to blocks.
    # For each Mi, define:
    #   a_map: dict from 2x2 position (r,c) to list of (U_row, coeff)
    #   b_map: dict from 2x2 position (r,c) to list of (V_row, coeff)
    #   c_map: dict from 2x2 position (r,c) to list of (W_row, coeff)
    
    # M1: (A00+A11)(B00+B11) -> contributes to C00 and C11
    # A00 = A[0:2,0:2], A11 = A[2:4,2:4]
    # A_sub[0,0] = A[0,0]+A[2,2], A_sub[0,1] = A[0,1]+A[2,3]
    # A_sub[1,0] = A[1,0]+A[3,2], A_sub[1,1] = A[1,1]+A[3,3]
    # B00 = B[0:2,0:2], B11 = B[2:4,2:4]
    # B_sub[0,0] = B[0,0]+B[2,2], B_sub[0,1] = B[0,1]+B[2,3]
    # B_sub[1,0] = B[1,0]+B[3,2], B_sub[1,1] = B[1,1]+B[3,3]
    # C_sub[0,0] -> C[0,0], C_sub[0,1] -> C[0,1]
    # C_sub[1,0] -> C[1,0], C_sub[1,1] -> C[1,1]
    # But M1 also contributes to C11: C_sub[0,0] -> C[2,2], etc.
    # Wait, no. In the block Strassen, M1 contributes to C00 with +1 and C11 with +1.
    # But M1 is a 2x2 matrix. Its elements go to specific positions.
    # Actually, I think the correct interpretation is:
    # M1 contributes to C00 (the top-left 2x2 block) and C11 (the bottom-right 2x2 block).
    # So C_sub[0,0] contributes to both C[0,0] and C[2,2]?
    # No, that doesn't make sense. Let me reconsider.
    
    # I think the issue is that in the 4x4 case, the Strassen formulas are:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # where C00, C01, C10, C11 are 2x2 blocks, and M1...M7 are also 2x2 matrices.
    # So M1 contributes to both C00 and C11 as a 2x2 block.
    # This means the elements of M1 appear in both the top-left and bottom-right 2x2 blocks of C.
    
    # So for M1:
    # C_sub[0,0] contributes to C[0,0] (in C00) and C[2,2] (in C11)
    # C_sub[0,1] contributes to C[0,1] (in C00) and C[2,3] (in C11)
    # C_sub[1,0] contributes to C[1,0] (in C00) and C[3,2] (in C11)
    # C_sub[1,1] contributes to C[1,1] (in C00) and C[3,3] (in C11)
    
    # Let me define the 7 multiplications:
    
    def add_multiplication(a_map, b_map, c_map, col_offset):
        """Add a 2x2x2 Strassen multiplication to U, V, W."""
        for s in range(7):
            for pos in [(0,0), (0,1), (1,0), (1,1)]:
                for (row, coeff) in a_map[pos]:
                    U[row, col_offset + s] += coeff * u2[pos[0]*2+pos[1], s]
                for (row, coeff) in b_map[pos]:
                    V[row, col_offset + s] += coeff * v2[pos[0]*2+pos[1], s]
                for (row, coeff) in c_map[pos]:
                    W[row, col_offset + s] += coeff * w2[pos[0]*2+pos[1], s]
    
    # M1: (A00+A11)(B00+B11) -> C00 (+1) and C11 (+1)
    a_map = {
        (0,0): [(a_row(0,0), 1), (a_row(2,2), 1)],
        (0,1): [(a_row(0,1), 1), (a_row(2,3), 1)],
        (1,0): [(a_row(1,0), 1), (a_row(3,2), 1)],
        (1,1): [(a_row(1,1), 1), (a_row(3,3), 1)],
    }
    b_map = {
        (0,0): [(b_row(0,0), 1), (b_row(2,2), 1)],
        (0,1): [(b_row(0,1), 1), (b_row(2,3), 1)],
        (1,0): [(b_row(1,0), 1), (b_row(3,2), 1)],
        (1,1): [(b_row(1,1), 1), (b_row(3,3), 1)],
    }
    c_map = {
        (0,0): [(w_row(0,0), 1), (w_row(2,2), 1)],
        (0,1): [(w_row(1,0), 1), (w_row(3,2), 1)],
        (1,0): [(w_row(0,1), 1), (w_row(2,3), 1)],
        (1,1): [(w_row(1,1), 1), (w_row(3,3), 1)],
    }
    add_multiplication(a_map, b_map, c_map, 0)
    
    # M2: (A10+A11)B00 -> C10 (+1)
    # A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # A_sub[0,0] = A[2,0]+A[2,2], A_sub[0,1] = A[2,1]+A[2,3]
    # A_sub[1,0] = A[3,0]+A[3,2], A_sub[1,1] = A[3,1]+A[3,3]
    # B00 = B[0:2,0:2]
    # C_sub[0,0] -> C[2,0], C_sub[0,1] -> C[2,1]
    # C_sub[1,0] -> C[3,0], C_sub[1,1] -> C[3,1]
    a_map = {
        (0,0): [(a_row(2,0), 1), (a_row(2,2), 1)],
        (0,1): [(a_row(2,1), 1), (a_row(2,3), 1)],
        (1,0): [(a_row(3,0), 1), (a_row(3,2), 1)],
        (1,1): [(a_row(3,1), 1), (a_row(3,3), 1)],
    }
    b_map = {
        (0,0): [(b_row(0,0), 1)],
        (0,1): [(b_row(0,1), 1)],
        (1,0): [(b_row(1,0), 1)],
        (1,1): [(b_row(1,1), 1)],
    }
    c_map = {
        (0,0): [(w_row(0,2), 1)],
        (0,1): [(w_row(1,2), 1)],
        (1,0): [(w_row(0,3), 1)],
        (1,1): [(w_row(1,3), 1)],
    }
    add_multiplication(a_map, b_map, c_map, 7)
    
    # M3: A00(B01-B11) -> C01 (+1)
    # A00 = A[0:2,0:2]
    # B01 = B[0:2,2:4], B11 = B[2:4,2:4]
    # B_sub[0,0] = B[0,2]-B[2,2], B_sub[0,1] = B[0,3]-B[2,3]
    # B_sub[1,0] = B[1,2]-B[3,2], B_sub[1,1] = B[1,3]-B[3,3]
    # C_sub[0,0] -> C[0,2], C_sub[0,1] -> C[0,3]
    # C_sub[1,0] -> C[1,2], C_sub[1,1] -> C[1,3]
    a_map = {
        (0,0): [(a_row(0,0), 1)],
        (0,1): [(a_row(0,1), 1)],
        (1,0): [(a_row(1,0), 1)],
        (1,1): [(a_row(1,1), 1)],
    }
    b_map = {
        (0,0): [(b_row(0,2), 1), (b_row(2,2), -1)],
        (0,1): [(b_row(0,3), 1), (b_row(2,3), -1)],
        (1,0): [(b_row(1,2), 1), (b_row(3,2), -1)],
        (1,1): [(b_row(1,3), 1), (b_row(3,3), -1)],
    }
    c_map = {
        (0,0): [(w_row(2,0), 1)],
        (0,1): [(w_row(3,0), 1)],
        (1,0): [(w_row(2,1), 1)],
        (1,1): [(w_row(3,1), 1)],
    }
    add_multiplication(a_map, b_map, c_map, 14)
    
    # M4: A11(B10-B00) -> C00 (+1) and C11 (+1)
    # A11 = A[2:4,2:4]
    # B10 = B[2:4,0:2], B00 = B[0:2,0:2]
    # B_sub[0,0] = B[2,0]-B[0,0], B_sub[0,1] = B[2,1]-B[0,1]
    # B_sub[1,0] = B[3,0]-B[1,0], B_sub[1,1] = B[3,1]-B[1,1]
    # C_sub[0,0] -> C[0,0] and C[2,2]
    # C_sub[0,1] -> C[0,1] and C[2,3]
    # C_sub[1,0] -> C[1,0] and C[3,2]
    # C_sub[1,1] -> C[1,1] and C[3,3]
    a_map = {
        (0,0): [(a_row(2,2), 1)],
        (0,1): [(a_row(2,3), 1)],
        (1,0): [(a_row(3,2), 1)],
        (1,1): [(a_row(3,3), 1)],
    }
    b_map = {
        (0,0): [(b_row(2,0), 1), (b_row(0,0), -1)],
        (0,1): [(b_row(2,1), 1), (b_row(0,1), -1)],
        (1,0): [(b_row(3,0), 1), (b_row(1,0), -1)],
        (1,1): [(b_row(3,1), 1), (b_row(1,1), -1)],
    }
    c_map = {
        (0,0): [(w_row(0,0), 1), (w_row(2,2), 1)],
        (0,1): [(w_row(1,0), 1), (w_row(3,2), 1)],
        (1,0): [(w_row(0,1), 1), (w_row(2,3), 1)],
        (1,1): [(w_row(1,1), 1), (w_row(3,3), 1)],
    }
    add_multiplication(a_map, b_map, c_map, 21)
    
    # M5: (A00+A01)B11 -> C00 (-1) and C01 (+1)
    # A00 = A[0:2,0:2], A01 = A[0:2,2:4]
    # A_sub[0,0] = A[0,0]+A[0,2], A_sub[0,1] = A[0,1]+A[0,3]
    # A_sub[1,0] = A[1,0]+A[1,2], A_sub[1,1] = A[1,1]+A[1,3]
    # B11 = B[2:4,2:4]
    # C_sub[0,0] -> C[0,0] (-1) and C[0,2] (+1)
    # C_sub[0,1] -> C[0,1] (-1) and C[0,3] (+1)
    # C_sub[1,0] -> C[1,0] (-1) and C[1,2] (+1)
    # C_sub[1,1] -> C[1,1] (-1) and C[1,3] (+1)
    a_map = {
        (0,0): [(a_row(0,0), 1), (a_row(0,2), 1)],
        (0,1): [(a_row(0,1), 1), (a_row(0,3), 1)],
        (1,0): [(a_row(1,0), 1), (a_row(1,2), 1)],
        (1,1): [(a_row(1,1), 1), (a_row(1,3), 1)],
    }
    b_map = {
        (0,0): [(b_row(2,2), 1)],
        (0,1): [(b_row(2,3), 1)],
        (1,0): [(b_row(3,2), 1)],
        (1,1): [(b_row(3,3), 1)],
    }
    c_map = {
        (0,0): [(w_row(0,0), -1), (w_row(2,0), 1)],
        (0,1): [(w_row(1,0), -1), (w_row(3,0), 1)],
        (1,0): [(w_row(0,1), -1), (w_row(2,1), 1)],
        (1,1): [(w_row(1,1), -1), (w_row(3,1), 1)],
    }
    add_multiplication(a_map, b_map, c_map, 28)
    
    # M6: (A10-A00)(B00+B01) -> C11 (+1)
    # A10 = A[2:4,0:2], A00 = A[0:2,0:2]
    # A_sub[0,0] = A[2,0]-A[0,0], A_sub[0,1] = A[2,1]-A[0,1]
    # A_sub[1,0] = A[3,0]-A[1,0], A_sub[1,1] = A[3,1]-A[1,1]
    # B00 = B[0:2,0:2], B01 = B[0:2,2:4]
    # B_sub[0,0] = B[0,0]+B[0,2], B_sub[0,1] = B[0,1]+B[0,3]
    # B_sub[1,0] = B[1,0]+B[1,2], B_sub[1,1] = B[1,1]+B[1,3]
    # C_sub[0,0] -> C[2,2], C_sub[0,1] -> C[2,3]
    # C_sub[1,0] -> C[3,2], C_sub[1,1] -> C[3,3]
    a_map = {
        (0,0): [(a_row(2,0), 1), (a_row(0,0), -1)],
        (0,1): [(a_row(2,1), 1), (a_row(0,1), -1)],
        (1,0): [(a_row(3,0), 1), (a_row(1,0), -1)],
        (1,1): [(a_row(3,1), 1), (a_row(1,1), -1)],
    }
    b_map = {
        (0,0): [(b_row(0,0), 1), (b_row(0,2), 1)],
        (0,1): [(b_row(0,1), 1), (b_row(0,3), 1)],
        (1,0): [(b_row(1,0), 1), (b_row(1,2), 1)],
        (1,1): [(b_row(1,1), 1), (b_row(1,3), 1)],
    }
    c_map = {
        (0,0): [(w_row(2,2), 1)],
        (0,1): [(w_row(3,2), 1)],
        (1,0): [(w_row(2,3), 1)],
        (1,1): [(w_row(3,3), 1)],
    }
    add_multiplication(a_map, b_map, c_map, 35)
    
    # M7: (A01-A11)(B10+B11) -> C00 (+1)
    # A01 = A[0:2,2:4], A11 = A[2:4,2:4]
    # A_sub[0,0] = A[0,2]-A[2,2], A_sub[0,1] = A[0,3]-A[2,3]
    # A_sub[1,0] = A[1,2]-A[3,2], A_sub[1,1] = A[1,3]-A[3,3]
    # B10 = B[2:4,0:2], B11 = B[2:4,2:4]
    # B_sub[0,0] = B[2,0]+B[2,2], B_sub[0,1] = B[2,1]+B[2,3]
    # B_sub[1,0] = B[3,0]+B[3,2], B_sub[1,1] = B[3,1]+B[3,3]
    # C_sub[0,0] -> C[0,0], C_sub[0,1] -> C[0,1]
    # C_sub[1,0] -> C[1,0], C_sub[1,1] -> C[1,1]
    a_map = {
        (0,0): [(a_row(0,2), 1), (a_row(2,2), -1)],
        (0,1): [(a_row(0,3), 1), (a_row(2,3), -1)],
        (1,0): [(a_row(1,2), 1), (a_row(3,2), -1)],
        (1,1): [(a_row(1,3), 1), (a_row(3,3), -1)],
    }
    b_map = {
        (0,0): [(b_row(2,0), 1), (b_row(2,2), 1)],
        (0,1): [(b_row(2,1), 1), (b_row(2,3), 1)],
        (1,0): [(b_row(3,0), 1), (b_row(3,2), 1)],
        (1,1): [(b_row(3,1), 1), (b_row(3,3), 1)],
    }
    c_map = {
        (0,0): [(w_row(0,0), 1)],
        (0,1): [(w_row(1,0), 1)],
        (1,0): [(w_row(0,1), 1)],
        (1,1): [(w_row(1,1), 1)],
    }
    add_multiplication(a_map, b_map, c_map, 42)
    
    return U, V, W


def construct():
    """Construct a decomposition for 4x4x5 matrix multiplication.
    Returns {'U': (16, rank), 'V': (20, rank), 'W': (20, rank)}
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # Build 4x4x1 schoolbook decomposition (rank 16)
    U1 = np.zeros((16, 16), dtype=np.int64)
    V1 = np.zeros((20, 16), dtype=np.int64)
    W1 = np.zeros((20, 16), dtype=np.int64)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # For the 4x4x1 part, B has shape (4,1), so B[j,0]
                # V row j*p + k = j*5 + 0 = j*5
                # W row k*n + i = k*4 + i
                U1[i * 4 + j, col] = 1
                V1[j * 5 + 0, col] = 1  # B[j,0] for the 5th column
                W1[k * 4 + i, col] = 1
                col += 1
    
    # Combine: total rank = 49 + 16 = 65
    U = np.zeros((16, 65), dtype=np.int64)
    V = np.zeros((20, 65), dtype=np.int64)
    W = np.zeros((20, 65), dtype=np.int64)
    
    # First 49 columns: 4x4x4 part
    U[:, :49] = U4
    V[:16, :49] = V4  # V4 is (16, 49), V is (20, 65)
    W[:16, :49] = W4  # W4 is (16, 49), W is (20, 65)
    
    # Last 16 columns: 4x4x1 part (for B's 5th column)
    U[:, 49:] = U1
    V[:, 49:] = V1
    W[:, 49:] = W1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
