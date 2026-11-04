"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a low-rank exact tensor decomposition for 4x4 x 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Build hybrid decomposition: Strassen for 4x4*4x4 + schoolbook for 4x4*4x3
    result = _hybrid_decomposition(n, m, p)
    return result


def _hybrid_decomposition(n, m, p):
    """
    Hybrid decomposition:
    - Split B (4x7) into B_left (4x4) and B_right (4x3)
    - C_left = A * B_left (4x4) computed via Strassen-based rank-56 decomposition
    - C_right = A * B_right (4x3) computed via schoolbook rank-48 decomposition
    - Total rank = 56 + 48 = 104
    """
    # Strassen part: 4x4 * 4x4, rank 56
    U_str, V_str, W_str = _strassen_4x4_4x4()
    # Schoolbook part: 4x4 * 4x3, rank 48
    U_sch, V_sch, W_sch = _schoolbook_4x4_4x3()
    
    rank_total = 56 + 48
    U = [[0] * rank_total for _ in range(n * m)]  # 16 x 104
    V = [[0] * rank_total for _ in range(m * p)]  # 28 x 104
    W = [[0] * rank_total for _ in range(p * n)]  # 28 x 104
    
    # Place Strassen part (columns 0-55)
    for i in range(n * m):
        for c in range(56):
            U[i][c] = U_str[i][c]
    for i in range(m * 4):  # Only first 4 columns of B
        for c in range(56):
            V[i][c] = V_str[i][c]
    for i in range(4 * n):  # Only first 4 rows of C (k=0..3)
        for c in range(56):
            W[i][c] = W_str[i][c]
    
    # Place schoolbook part (columns 56-103)
    for i in range(n * m):
        for c in range(48):
            U[i][56 + c] = U_sch[i][c]
    for j in range(m):
        for k in range(4, 7):  # Columns 4,5,6 of B
            for c in range(48):
                V[j * p + k][56 + c] = V_sch[j * 3 + (k - 4)][c]
    for k in range(4, 7):
        for i in range(n):
            for c in range(48):
                W[k * n + i][56 + c] = W_sch[(k - 4) * n + i][c]
    
    return {"U": U, "V": V, "W": W}


def _strassen_4x4_4x4():
    """
    Strassen-based decomposition for 4x4 * 4x4, rank 56.
    
    Split into 2x2 blocks:
    A = [[A00, A01], [A10, A11]]  (each 2x2)
    B = [[B00, B01], [B10, B11]]  (each 2x2)
    C = [[C00, C01], [C10, C11]]  (each 2x2)
    
    Use Strassen at the 2x2 block level (7 block multiplications),
    each block multiplication done via Strassen (7 scalar mults).
    Total: 7 * 7 = 49... but we need 56.
    
    Alternative: treat each of the 4 output blocks independently.
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each Ci,j is a 2x2*2x2 + 2x2*2x2 = 2 Strassen products = 14 mults.
    4 output blocks * 14 = 56. This works!
    """
    rank = 56
    U = [[0] * rank for _ in range(16)]  # 4x4 A
    V = [[0] * rank for _ in range(16)]  # 4x4 B
    W = [[0] * rank for _ in range(16)]  # 4x4 C
    
    # A indices: A[i][j] -> U[i*4+j]
    # B indices: B[j][k] -> V[j*4+k]
    # C indices: C[k][i] -> W[k*4+i]  (trace convention: W[k*n+i])
    
    # For each output block (ci, cj) where ci,cj in {0,1}:
    # C[ci*2:(ci+1)*2, cj*2:(cj+1)*2] = sum over (ai, bj) of A[ai*2:(ai+1)*2, bj*2:(bj+1)*2] * B[bj*2:(bj+1)*2, cj*2:(cj+1)*2]
    
    # We need: for all k,i in {0,1,2,3}: C[k][i] = sum_j A[i][j] * B[j][k]
    # In our indexing: C[k][i] = W[k*4+i], A[i][j] = U[i*4+j], B[j][k] = V[j*4+k]
    
    # For each (ai, bj, cj) triple where ai,bj,cj in {0,1}:
    # The contribution to C[2*cj..2*cj+1, 2*ai..2*ai+1] from A[2*ai..2*ai+1, 2*bj..2*bj+1] * B[2*bj..2*bj+1, 2*cj..2*cj+1]
    # is a 2x2*2x2 multiplication, done via Strassen (rank 7).
    
    # There are 2*2*2 = 8 such (ai, bj, cj) combinations, each contributing rank 7.
    # Total: 8 * 7 = 56. 
    
    col = 0
    for ai in range(2):
        for bj in range(2):
            for cj in range(2):
                # Strassen for 2x2*2x2: A_block * B_block -> C_block
                # A_block = A[2*ai:2*ai+2, 2*bj:2*bj+2]
                # B_block = B[2*bj:2*bj+2, 2*cj:2*cj+2]
                # C_block = C[2*cj:2*cj+2, 2*ai:2*ai+2]
                
                # Strassen 7 multiplications:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B10-B11)
                # M4 = A11(B01-B00)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B01)
                # M7 = (A01-A11)(B10+B11)
                
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # A_block indices in U:
                # A00 = U[2*ai*4 + 2*bj], A01 = U[2*ai*4 + 2*bj+1]
                # A10 = U[(2*ai+1)*4 + 2*bj], A11 = U[(2*ai+1)*4 + 2*bj+1]
                
                # B_block indices in V:
                # B00 = V[(2*bj)*4 + 2*cj], B01 = V[(2*bj)*4 + 2*cj+1]
                # B10 = V[(2*bj+1)*4 + 2*cj], B11 = V[(2*bj+1)*4 + 2*cj+1]
                
                # C_block indices in W:
                # C00 = W[(2*cj)*4 + 2*ai], C01 = W[(2*cj)*4 + 2*ai+1]
                # C10 = W[(2*cj+1)*4 + 2*ai], C11 = W[(2*cj+1)*4 + 2*ai+1]
                
                a00 = 2*ai*4 + 2*bj
                a01 = 2*ai*4 + 2*bj + 1
                a10 = (2*ai+1)*4 + 2*bj
                a11 = (2*ai+1)*4 + 2*bj + 1
                
                b00 = 2*bj*4 + 2*cj
                b01 = 2*bj*4 + 2*cj + 1
                b10 = (2*bj+1)*4 + 2*cj
                b11 = (2*bj+1)*4 + 2*cj + 1
                
                c00 = 2*cj*4 + 2*ai
                c01 = 2*cj*4 + 2*ai + 1
                c10 = (2*cj+1)*4 + 2*ai
                c11 = (2*cj+1)*4 + 2*ai + 1
                
                # M1 = (A00+A11)(B00+B11)
                U[a00][col] += 1; U[a11][col] += 1
                V[b00][col] += 1; V[b11][col] += 1
                W[c00][col] += 1; W[c11][col] += 1
                col += 1
                
                # M2 = (A10+A11)B00
                U[a10][col] += 1; U[a11][col] += 1
                V[b00][col] += 1
                W[c10][col] += 1; W[c11][col] += 1
                col += 1
                
                # M3 = A00(B10-B11)
                U[a00][col] += 1
                V[b10][col] += 1; V[b11][col] += -1
                W[c01][col] += 1; W[c11][col] += 1
                col += 1
                
                # M4 = A11(B01-B00)
                U[a11][col] += 1
                V[b01][col] += 1; V[b00][col] += -1
                W[c00][col] += 1; W[c10][col] += 1
                col += 1
                
                # M5 = (A00+A01)B11
                U[a00][col] += 1; U[a01][col] += 1
                V[b11][col] += 1
                W[c00][col] += 1; W[c01][col] += 1
                col += 1
                
                # M6 = (A10-A00)(B00+B01)
                U[a10][col] += 1; U[a00][col] += -1
                V[b00][col] += 1; V[b01][col] += 1
                W[c11][col] += 1
                col += 1
                
                # M7 = (A01-A11)(B10+B11)
                U[a01][col] += 1; U[a11][col] += -1
                V[b10][col] += 1; V[b11][col] += 1
                W[c00][col] += 1
                col += 1
    
    return U, V, W


def _schoolbook_4x4_4x3():
    """
    Schoolbook decomposition for 4x4 * 4x3, rank 48.
    A is 4x4, B is 4x3, C is 4x3.
    C[k][i] = sum_j A[i][j] * B[j][k]
    
    U: 16 x 48 (A[i][j] -> U[i*4+j])
    V: 12 x 48 (B[j][k] -> V[j*3+k])
    W: 12 x 48 (C[k][i] -> W[k*4+i])
    """
    rank = 48
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    return U, V, W
# EVOLVE-BLOCK-END
