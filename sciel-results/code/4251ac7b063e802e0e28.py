"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank) with convention W[k*n+i, r]
    
    The tensor entry T[i, j, k] is 1 if (i,j,k) is a valid index.
    The decomposition says:
    T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank16(n=2, m=2, p=5):
    """
    Construct a rank-16 decomposition:
    - Strassen (rank 7) for B columns 0,1
    - Custom rank-9 decomposition for B columns 2,3,4
    
    B indices: B00=0, B01=1, B10=2, B11=3, B02=4, B03=5, B04=6, B12=7, B13=8, B14=9
    C indices: C00=0, C01=1, C02=2, C03=3, C04=4, C10=5, C11=6, C12=7, C13=8, C14=9
    """
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # --- Part 1: Strassen for cols 0, 1 (7 terms) ---
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=5, C11=6
    
    # Col 0: P1 = A00 * (B00 + B11)
    U[0][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[6][0] = 1
    
    # Col 1: P2 = (A10 + A11) * B00
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[5][1] = 1
    
    # Col 2: P3 = (A00 + A01) * B11
    U[0][2] = 1
    U[1][2] = 1
    V[3][2] = 1
    W[1][2] = 1
    W[6][2] = 1
    
    # Col 3: P4 = A11 * (B01 - B00)
    U[3][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    
    # Col 4: P5 = A00 * (B10 - B11)
    U[0][4] = 1
    V[2][4] = 1
    V[3][4] = -1
    W[0][4] = -1
    W[5][4] = 1
    
    # Col 5: P6 = (A10 - A00) * B00
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    W[6][5] = 1
    
    # Col 6: P7 = (A01 - A11) * B11
    U[1][6] = 1
    U[3][6] = -1
    V[3][6] = 1
    W[0][6] = -1
    W[1][6] = 1

    # --- Part 2: Custom rank-9 for cols 2,3,4 ---
    # B indices: B02=4, B03=5, B04=6, B12=7, B13=8, B14=9
    # C indices: C02=2, C03=3, C04=4, C12=7, C13=8, C14=9
    
    # Col 7: S1 = A00 * (B02 + B13)
    U[0][7] = 1
    V[4][7] = 1
    V[8][7] = 1
    W[2][7] = 1
    W[8][7] = 1
    
    # Col 8: S2 = (A10 + A11) * B02
    U[2][8] = 1
    U[3][8] = 1
    V[4][8] = 1
    W[7][8] = 1
    
    # Col 9: S3 = (A00 + A01) * B13
    U[0][9] = 1
    U[1][9] = 1
    V[8][9] = 1
    W[3][9] = 1
    W[8][9] = 1
    
    # Col 10: S4 = A11 * (B03 - B02)
    U[3][10] = 1
    V[5][10] = 1
    V[4][10] = -1
    W[2][10] = 1
    
    # Col 11: S5 = A00 * (B12 - B13)
    U[0][11] = 1
    V[7][11] = 1
    V[8][11] = -1
    W[2][11] = -1
    W[7][11] = 1
    
    # Col 12: S6 = (A10 - A00) * B02
    U[2][12] = 1
    U[0][12] = -1
    V[4][12] = 1
    W[8][12] = 1
    
    # Col 13: S7 = (A01 - A11) * B13
    U[1][13] = 1
    U[3][13] = -1
    V[8][13] = 1
    W[2][13] = -1
    W[3][13] = 1
    
    # Col 14: S8 = A00 * B04
    U[0][14] = 1
    V[6][14] = 1
    W[4][14] = 1
    
    # Col 15: S9 = A10 * B04
    U[2][15] = 1
    V[6][15] = 1
    W[9][15] = 1

    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using Strassen for 2x2x2 (cols 0-1),
    Strassen for 2x2x2 (cols 2-3), and Schoolbook for 2x2x1 (col 4).
    """
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # --- Part 1: Strassen for cols 0, 1 ---
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # Col 0: P1
    U[0][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # Col 1: P2
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Col 2: P3
    U[0][2] = 1
    U[1][2] = 1
    V[3][2] = 1
    W[1][2] = 1
    W[3][2] = 1
    
    # Col 3: P4
    U[3][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    
    # Col 4: P5
    U[0][4] = 1
    V[2][4] = 1
    V[3][4] = -1
    W[0][4] = -1
    W[2][4] = 1
    
    # Col 5: P6
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    W[3][5] = 1
    
    # Col 6: P7
    U[1][6] = 1
    U[3][6] = -1
    V[3][6] = 1
    W[0][6] = -1
    W[1][6] = 1

    # --- Part 2: Strassen for cols 2, 3 ---
    # B indices: B02=4, B03=5, B12=7, B13=8
    # C indices: C02=4, C03=5, C12=6, C13=7
    
    # Col 7: Q1
    U[0][7] = 1
    V[4][7] = 1
    V[8][7] = 1
    W[4][7] = 1
    W[7][7] = 1
    
    # Col 8: Q2
    U[2][8] = 1
    U[3][8] = 1
    V[4][8] = 1
    W[6][8] = 1
    
    # Col 9: Q3
    U[0][9] = 1
    U[1][9] = 1
    V[8][9] = 1
    W[5][9] = 1
    W[7][9] = 1
    
    # Col 10: Q4
    U[3][10] = 1
    V[5][10] = 1
    V[4][10] = -1
    W[4][10] = 1
    
    # Col 11: Q5
    U[0][11] = 1
    V[7][11] = 1
    V[8][11] = -1
    W[4][11] = -1
    W[6][11] = 1
    
    # Col 12: Q6
    U[2][12] = 1
    U[0][12] = -1
    V[4][12] = 1
    W[7][12] = 1
    
    # Col 13: Q7
    U[1][13] = 1
    U[3][13] = -1
    V[8][13] = 1
    W[4][13] = -1
    W[5][13] = 1

    # --- Part 3: Schoolbook for col 4 ---
    # B indices: B04=6, B14=9
    # C indices: C04=8, C14=9
    
    # Col 14: R1
    U[0][14] = 1
    V[6][14] = 1
    W[8][14] = 1
    
    # Col 15: R2
    U[1][15] = 1
    V[9][15] = 1
    W[8][15] = 1
    
    # Col 16: R3
    U[2][16] = 1
    V[6][16] = 1
    W[9][16] = 1
    
    # Col 17: R4
    U[3][17] = 1
    V[9][17] = 1
    W[9][17] = 1

    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_schoolbook(n=2, m=2, p=5):
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _search_rank15(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-15 decomposition by starting from rank-16 and trying to eliminate one term.
    Uses local search with integer/half-integer adjustments.
    """
    start_time = time.time()
    base = _construct_rank16(n, m, p)
    if base is None:
        return None
    
    U0 = [row[:] for row in base['U']]
    V0 = [row[:] for row in base['V']]
    W0 = [row[:] for row in base['W']]
    rank0 = 16
    
    # Try removing each column and see if we can fix it
    for remove_col in range(rank0):
        if time.time() - start_time > timeout - 10:
            break
        
        # Create a copy without the removed column
        U = [[U0[i][j] for j in range(rank0) if j != remove_col] for i in range(n*m)]
        V = [[V0[i][j] for j in range(rank0) if j != remove_col] for i in range(m*p)]
        W = [[W0[i][j] for j in range(rank0) if j != remove_col] for i in range(p*n)]
        
        # Check if it's already valid (unlikely but check)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Try small perturbations to fix the errors
        for attempt in range(300):
            if time.time() - start_time > timeout - 5:
                break
            
            # Pick a random entry to adjust
            mat_choice = random.randint(0, 2)  # 0=U, 1=V, 2=W
            if mat_choice == 0:
                row = random.randint(0, n*m-1)
                col = random.randint(0, rank0-2)
                old_val = U[row][col]
                delta = random.choice([-1, -0.5, 0.5, 1])
                U[row][col] = old_val + delta
            elif mat_choice == 1:
                row = random.randint(0, m*p-1)
                col = random.randint(0, rank0-2)
                old_val = V[row][col]
                delta = random.choice([-1, -0.5, 0.5, 1])
                V[row][col] = old_val + delta
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, rank0-2)
                old_val = W[row][col]
                delta = random.choice([-1, -0.5, 0.5, 1])
                W[row][col] = old_val + delta
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            
            # Revert
            if mat_choice == 0:
                U[row][col] = old_val
            elif mat_choice == 1:
                V[row][col] = old_val
            else:
                W[row][col] = old_val
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Try rank 15 search first
    if time.time() < deadline - 30:
        result = _search_rank15(n, m, p, timeout=min(25, deadline - time.time() - 30))
        if result is not None:
            return result
    
    # Try rank 16
    result = _construct_rank16(n, m, p)
    if result is not None:
        return result
    
    # Try rank 18
    result = _construct_rank18(n, m, p)
    if result is not None:
        return result
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
