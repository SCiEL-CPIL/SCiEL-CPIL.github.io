"""Matrix multiplication 2x5x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Target Rank 54 based on Strassen decomposition of 2x2 blocks
    # 2x2x2 blocks use rank 7 (Strassen)
    # 2x1x2 blocks use rank 4 (Naive)
    # Split j (middle dim 5) into 2, 2, 1
    # Split k (output cols 6) into 2, 2, 2
    
    # For j in {0,1} and k in {0,1}, {2,3}, {4,5}: 3 blocks of 2x2x2 -> 3 * 7 = 21
    # For j in {2,3} and k in {0,1}, {2,3}, {4,5}: 3 blocks of 2x2x2 -> 3 * 7 = 21
    # For j in {4} and k in {0,1}, {2,3}, {4,5}: 3 blocks of 2x1x2 -> 3 * 4 = 12
    # Total = 21 + 21 + 12 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        i_base: base index for i (rows of A, rows of C)
        j_base: base index for j (cols of A, rows of B)
        k_base: base index for k (cols of B, cols of C)
        """
        nonlocal col
        
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base + 1) * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base + 1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base + 1)][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] -= 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        """
        Adds a rank-4 naive decomposition for a 2x1 * 1x2 block.
        i_base: base index for i
        j_base: single index for j
        k_base: base index for k
        """
        nonlocal col
        
        # 1: A00 B00 -> C00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

    # Apply Strassen blocks for j pairs (0,1) and (2,3)
    # k pairs are (0,1), (2,3), (4,5)
    
    # j_base = 0
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # j_base = 2
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4
    # k pairs are (0,1), (2,3), (4,5)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    # Verify col == 54
    if col != 54:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
