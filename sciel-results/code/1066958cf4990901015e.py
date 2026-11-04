"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try block decomposition using Strassen for 2x2x2 sub-blocks
    result = _block_decomposition(n, m, p)
    
    # Fallback to schoolbook if block decomposition fails
    if result is None:
        result = _schoolbook(n, m, p)
    
    return result

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with shape (4,7), (4,7), (4,7)
    where indices are: U: i*2+j (i,j in 0,1), V: j*2+k, W: k*2+i
    """
    # Strassen's 7 products for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]  # n*m = 2*2 = 4
    V = [[0]*7 for _ in range(4)]  # m*p = 2*2 = 4
    W = [[0]*7 for _ in range(4)]  # p*n = 2*2 = 4
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=(0,0), A11=(1,1) -> U[0*2+0]=U[0], U[1*2+1]=U[3]
    # V: B00=(0,0), B11=(1,1) -> V[0*2+0]=V[0], V[1*2+1]=V[3]
    # W: C00 gets +1, C11 gets +1 -> W[0*2+0]=W[0], W[1*2+1]=W[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # U: A10=(1,0), A11=(1,1) -> U[1*2+0]=U[2], U[1*2+1]=U[3]
    # V: B00=(0,0) -> V[0*2+0]=V[0]
    # W: C10 gets +1 -> W[0*2+1]=W[1]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = A00(B10-B11)
    # U: A00=(0,0) -> U[0*2+0]=U[0]
    # V: B10=(1,0), B11=(1,1) -> V[1*2+0]=V[2], V[1*2+1]=V[3]
    # W: C01 gets +1 -> W[1*2+0]=W[2]
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4 = A11(B01-B10)
    # U: A11=(1,1) -> U[1*2+1]=U[3]
    # V: B01=(0,1), B10=(1,0) -> V[0*2+1]=V[1], V[1*2+0]=V[2]
    # W: C00 gets +1, C10 gets +1 -> W[0*2+0]=W[0], W[0*2+1]=W[1]
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    # U: A00=(0,0), A01=(0,1) -> U[0*2+0]=U[0], U[0*2+1]=U[1]
    # V: B11=(1,1) -> V[1*2+1]=V[3]
    # W: C00 gets -1, C01 gets +1 -> W[0*2+0]=W[0], W[1*2+0]=W[2]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10=(1,0), A00=(0,0) -> U[1*2+0]=U[2], U[0*2+0]=U[0]
    # V: B00=(0,0), B01=(0,1) -> V[0*2+0]=V[0], V[0*2+1]=V[1]
    # W: C11 gets +1 -> W[1*2+1]=W[3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01=(0,1), A11=(1,1) -> U[0*2+1]=U[1], U[1*2+1]=U[3]
    # V: B10=(1,0), B11=(1,1) -> V[1*2+0]=V[2], V[1*2+1]=V[3]
    # W: C00 gets +1 -> W[0*2+0]=W[0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _block_decomposition(n, m, p):
    """Try to decompose n×m×p matrix multiplication using block structure.
    For 4x5x5, we can split into blocks and use Strassen where possible.
    """
    if n != 4 or m != 5 or p != 5:
        return None
    
    # For 4x5x5: split A (4x5) into [A_top (2x5); A_bottom (2x5)]
    # Split B (5x5) into [B_left (5x2), B_right (5x3)]
    # Split C (4x5) into [C_top_left (2x2), C_top_right (2x3); C_bottom_left (2x2), C_bottom_right (2x3)]
    
    # This gives:
    # C_top_left = A_top * B_left (2x2) -> use Strassen
    # C_top_right = A_top * B_right (2x3) -> schoolbook: 2*2*3=12
    # C_bottom_left = A_bottom * B_left (2x2) -> use Strassen
    # C_bottom_right = A_bottom * B_right (2x3) -> schoolbook: 2*2*3=12
    
    # Total rank: 7 + 12 + 7 + 12 = 38
    
    # Let's construct this
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to extract Strassen
    sU, sV, sW = _strassen_2x2x2()
    
    # Block 1: C_top_left = A_top * B_left (2x2x2)
    # A_top rows 0-1, cols 0-4 -> but we only use cols 0-1 for B_left
    # Actually, B_left is 5x2, so B_left[:, 0:2]
    # A_top is 2x5, so A_top[:, 0:5] but we multiply by B_left which is 5x2
    # Wait, I need to be more careful.
    
    # Let me reconsider. A is 4x5, B is 5x5, C is 4x5.
    # Split A into A1 (2x5) and A2 (2x5)
    # Split B into B1 (5x2) and B2 (5x3)
    # C = [A1*B1, A1*B2; A2*B1, A2*B2]
    
    # A1*B1: 2x5 * 5x2 = 2x2 -> Strassen (rank 7)
    # A1*B2: 2x5 * 5x3 = 2x3 -> schoolbook (rank 2*5*3=30)
    # A2*B1: 2x5 * 5x2 = 2x2 -> Strassen (rank 7)
    # A2*B2: 2x5 * 5x3 = 2x3 -> schoolbook (rank 2*5*3=30)
    
    # Total: 7 + 30 + 7 + 30 = 74
    
    # That's worse than schoolbook (100)? No, 74 < 100. Good.
    
    # But wait, can we do better? What if we split differently?
    # A: 4x5 -> [2x5; 2x5]
    # B: 5x5 -> [5x2 | 5x3]
    # This seems reasonable.
    
    # Let's build the decomposition
    U = [[0]*74 for _ in range(20)]  # n*m = 4*5 = 20
    V = [[0]*74 for _ in range(25)]  # m*p = 5*5 = 25
    W = [[0]*74 for _ in range(20)]  # p*n = 5*4 = 20
    
    col = 0
    
    # Block A1*B1: rows 0-1 of A, cols 0-1 of B -> C rows 0-1, cols 0-1
    # Strassen for 2x2x2
    for c in range(7):
        for u_row in range(4):
            # u_row corresponds to (i,j) in 2x2, i in 0-1, j in 0-1
            i = u_row // 2
            j = u_row % 2
            # In full A, this is row i, col j (since A1 is rows 0-1, and B1 is cols 0-1)
            # U index: i*m + j = i*5 + j
            U[i*5 + j][col] = sU[u_row][c]
        for v_row in range(4):
            # v_row corresponds to (j,k) in 2x2, j in 0-1, k in 0-1
            j = v_row // 2
            k = v_row % 2
            # In full B, this is row j, col k (since B1 is cols 0-1)
            # V index: j*p + k = j*5 + k
            V[j*5 + k][col] = sV[v_row][c]
        for w_row in range(4):
            # w_row corresponds to (k,i) in 2x2, k in 0-1, i in 0-1
            k = w_row // 2
            i = w_row % 2
            # In full C, this is col k, row i (since C top-left is rows 0-1, cols 0-1)
            # W index: k*n + i = k*4 + i
            W[k*4 + i][col] = sW[w_row][c]
        col += 1
    
    # Block A1*B2: rows 0-1 of A, cols 2-4 of B -> C rows 0-1, cols 2-4
    # Schoolbook: 2x5 * 5x3
    for i in range(2):  # rows of A1
        for j in range(5):  # cols of A1 = rows of B
            for k in range(3):  # cols of B2
                # U: row i of A, col j
                U[i*5 + j][col] = 1
                # V: row j of B, col (2+k) since B2 starts at col 2
                V[j*5 + (2+k)][col] = 1
                # W: col (2+k) of C, row i
                W[(2+k)*4 + i][col] = 1
                col += 1
    
    # Block A2*B1: rows 2-3 of A, cols 0-1 of B -> C rows 2-3, cols 0-1
    # Strassen for 2x2x2
    for c in range(7):
        for u_row in range(4):
            i = u_row // 2
            j = u_row % 2
            # In full A, this is row (2+i), col j
            U[(2+i)*5 + j][col] = sU[u_row][c]
        for v_row in range(4):
            j = v_row // 2
            k = v_row % 2
            V[j*5 + k][col] = sV[v_row][c]
        for w_row in range(4):
            k = w_row // 2
            i = w_row % 2
            # In full C, this is col k, row (2+i)
            W[k*4 + (2+i)][col] = sW[w_row][c]
        col += 1
    
    # Block A2*B2: rows 2-3 of A, cols 2-4 of B -> C rows 2-3, cols 2-4
    # Schoolbook: 2x5 * 5x3
    for i in range(2):  # rows of A2
        for j in range(5):  # cols of A2 = rows of B
            for k in range(3):  # cols of B2
                # U: row (2+i) of A, col j
                U[(2+i)*5 + j][col] = 1
                # V: row j of B, col (2+k)
                V[j*5 + (2+k)][col] = 1
                # W: col (2+k) of C, row (2+i)
                W[(2+k)*4 + (2+i)][col] = 1
                col += 1
    
    # Verify col == 74
    if col != 74:
        return None
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, 4, 5, 5):
        return None
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

# EVOLVE-BLOCK-END

# The construct function is already defined above in the EVOLVE-BLOCK.
# We just need to make sure the module-level construct is the one we want.
# Since the EVOLVE-BLOCK defines construct, it will be used.
