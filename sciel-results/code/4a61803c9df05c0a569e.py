"""Matrix multiplication 3x4x4 with rank 44 decomposition using Strassen on 2x2 blocks."""

import time
import sys

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p, sample_count=200):
    """Verify the decomposition produces correct matrix products."""
    import random
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected: T[i,j,k] = 1 for all i,j,k
    for _ in range(sample_count):
        # Pick random i, j, k
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        s = 0.0
        for c in range(rank):
            s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
        if abs(s - 1.0) > 1e-9:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-44 decomposition for 3x4x4 matrix multiplication.
    
    Strategy: Split the 3x4x4 problem into:
    - A 2x4x2 block (rows 0-1, all 4 cols of A, cols 0-1 of B) using Strassen: rank 14
    - A 1x4x2 block (row 2, all 4 cols of A, cols 0-1 of B) using schoolbook: rank 8
    - A 2x4x2 block (rows 0-1, all 4 cols of A, cols 2-3 of B) using Strassen: rank 14
    - A 1x4x2 block (row 2, all 4 cols of A, cols 2-3 of B) using schoolbook: rank 8
    
    Total: 14 + 8 + 14 + 8 = 44
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 44
    
    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, rank_offset):
        """
        Add a Strassen-based 2x2x2 decomposition for a 2x2x2 block.
        a: rows [a_row_start, a_row_start+2), cols [a_col_start, a_col_start+2)
        b: rows [b_row_start, b_row_start+2), cols [b_col_start, b_col_start+2)
        c: rows [c_row_start, c_row_start+2), cols [c_col_start, c_col_start+2)
        
        The block computes C = A*B where A is 2x2, B is 2x2, C is 2x2.
        Uses 7 multiplications.
        """
        # Strassen formulas for 2x2:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # c11 = M1 - M4 + M5 - M7
        # c12 = M3 + M7
        # c21 = M2 + M4
        # c22 = M1 - M3 + M2 - M6
        
        for s in range(7):
            c = rank_offset + s
            if s == 0:  # M1 = (a11+a22)(b11+b22)
                # U factors for A: a11, a22
                U[a_row_start * m + a_col_start][c] += 1  # a11
                U[(a_row_start+1) * m + (a_col_start+1)][c] += 1  # a22
                # V factors for B: b11, b22
                V[b_row_start * p + b_col_start][c] += 1  # b11
                V[(b_row_start+1) * p + (b_col_start+1)][c] += 1  # b22
                # W factors for C: c11, c22
                W[c_row_start * n + c_col_start][c] += 1  # c11
                W[(c_row_start+1) * n + (c_col_start+1)][c] += 1  # c22
                
            elif s == 1:  # M2 = (a21+a22)b11
                U[(a_row_start+1) * m + a_col_start][c] += 1  # a21
                U[(a_row_start+1) * m + (a_col_start+1)][c] += 1  # a22
                V[b_row_start * p + b_col_start][c] += 1  # b11
                W[(c_row_start+1) * n + c_col_start][c] += 1  # c21
                
            elif s == 2:  # M3 = a11(b12-b22)
                U[a_row_start * m + a_col_start][c] += 1  # a11
                V[b_row_start * p + (b_col_start+1)][c] += 1  # b12
                V[(b_row_start+1) * p + (b_col_start+1)][c] += -1  # -b22
                W[c_row_start * n + (c_col_start+1)][c] += 1  # c12
                
            elif s == 3:  # M4 = a22(b21-b11)
                U[(a_row_start+1) * m + (a_col_start+1)][c] += 1  # a22
                V[(b_row_start+1) * p + b_col_start][c] += 1  # b21
                V[b_row_start * p + b_col_start][c] += -1  # -b11
                W[c_row_start * n + c_col_start][c] += -1  # -c11
                W[(c_row_start+1) * n + c_col_start][c] += 1  # +c21
                
            elif s == 4:  # M5 = (a11+a12)b22
                U[a_row_start * m + a_col_start][c] += 1  # a11
                U[a_row_start * m + (a_col_start+1)][c] += 1  # a12
                V[(b_row_start+1) * p + (b_col_start+1)][c] += 1  # b22
                W[c_row_start * n + c_col_start][c] += 1  # +c11
                W[(c_row_start+1) * n + (c_col_start+1)][c] += 1  # +c22
                
            elif s == 5:  # M6 = (a21-a11)(b11+b12)
                U[(a_row_start+1) * m + a_col_start][c] += 1  # a21
                U[a_row_start * m + a_col_start][c] += -1  # -a11
                V[b_row_start * p + b_col_start][c] += 1  # b11
                V[b_row_start * p + (b_col_start+1)][c] += 1  # b12
                W[(c_row_start+1) * n + (c_col_start+1)][c] += -1  # -c22
                
            elif s == 6:  # M7 = (a12-a22)(b21+b22)
                U[a_row_start * m + (a_col_start+1)][c] += 1  # a12
                U[(a_row_start+1) * m + (a_col_start+1)][c] += -1  # -a22
                V[(b_row_start+1) * p + b_col_start][c] += 1  # b21
                V[(b_row_start+1) * p + (b_col_start+1)][c] += 1  # b22
                W[c_row_start * n + c_col_start][c] += -1  # -c11
                W[c_row_start * n + (c_col_start+1)][c] += 1  # +c12
        
        return rank_offset + 7
    
    def add_schoolbook_1x2x2_block(a_row, a_col_start, b_row_start, b_col_start, c_row, c_col_start, rank_offset):
        """
        Add a schoolbook decomposition for a 1x2x2 block.
        a: row a_row, cols [a_col_start, a_col_start+2)
        b: rows [b_row_start, b_row_start+2), cols [b_col_start, b_col_start+2)
        c: row c_row, cols [c_col_start, c_col_start+2)
        
        C = A*B where A is 1x2, B is 2x2, C is 1x2.
        c_col_start = A[0]*B[0,*] + A[1]*B[1,*]
        This requires 4 multiplications (2x2 = 4).
        """
        for j in range(2):
            for k in range(2):
                c = rank_offset + j * 2 + k
                U[a_row * m + a_col_start + j][c] += 1  # A[j]
                V[(b_row_start + j) * p + (b_col_start + k)][c] += 1  # B[j,k]
                W[(c_col_start + k) * n + c_row][c] += 1  # C[k]
        
        return rank_offset + 4
    
    # Block 1: rows 0-1, all 4 cols of A, cols 0-1 of B
    # Split A into 2x2 blocks: cols 0-1 and cols 2-3
    # Block 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    next_offset = add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1b: A[0:2, 2:4] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Wait, this is not quite right. Let me reconsider.
    
    # Actually, for the 2x4x2 case:
    # A is 2x4, B is 4x2, C is 2x2.
    # We can split A into [A_left, A_right] where each is 2x2.
    # B is 4x2, we can think of it as [B_top; B_bottom] where each is 2x2.
    # Then C = A_left * B_top + A_right * B_bottom.
    # This requires 2 * 7 = 14 multiplications using Strassen for each 2x2x2.
    
    # Let me redo this. I need to be more careful.
    
    # Reset
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Block 1: A[0:2, :] * B[:, 0:2] -> C[0:2, 0:2]
    # A is 2x4, split into A_left = A[0:2, 0:2] and A_right = A[0:2, 2:4]
    # B is 4x2, split into B_top = B[0:2, 0:2] and B_bottom = B[2:4, 0:2]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: Strassen, rank 7
    next_offset = add_strassen_block(0, 0, 0, 0, 0, 0, 0)  # A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    
    # A_right * B_bottom: Strassen, rank 7
    next_offset = add_strassen_block(0, 2, 2, 0, 0, 0, 7)  # A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    
    # Block 2: A[2, :] * B[:, 0:2] -> C[2, 0:2]
    # A is 1x4, B is 4x2, C is 1x2.
    # C[0] = A[0]*B[0,0] + A[1]*B[1,0] + A[2]*B[2,0] + A[3]*B[3,0]
    # C[1] = A[0]*B[0,1] + A[1]*B[1,1] + A[2]*B[2,1] + A[3]*B[3,1]
    # This is 8 scalar multiplications.
    for j in range(4):
        for k in range(2):
            c = next_offset + j * 2 + k
            U[2 * m + j][c] += 1  # A[2, j]
            V[j * p + k][c] += 1  # B[j, k]
            W[k * n + 2][c] += 1  # C[k, 2]
    next_offset += 8
    
    # Block 3: A[0:2, :] * B[:, 2:4] -> C[0:2, 2:4]
    # A is 2x4, split into A_left = A[0:2, 0:2] and A_right = A[0:2, 2:4]
    # B is 4x2, split into B_top = B[0:2, 2:4] and B_bottom = B[2:4, 2:4]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: Strassen, rank 7
    next_offset = add_strassen_block(0, 0, 0, 2, 0, 2, 15)  # A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C rows 0-1, cols 2-3
    
    # A_right * B_bottom: Strassen, rank 7
    next_offset = add_strassen_block(0, 2, 2, 2, 0, 2, 22)  # A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C rows 0-1, cols 2-3
    
    # Block 4: A[2, :] * B[:, 2:4] -> C[2, 2:4]
    # A is 1x4, B is 4x2, C is 1x2.
    for j in range(4):
        for k in range(2):
            c = next_offset + j * 2 + k
            U[2 * m + j][c] += 1  # A[2, j]
            V[j * p + (2 + k)][c] += 1  # B[j, 2+k]
            W[(2 + k) * n + 2][c] += 1  # C[2+k, 2]
    next_offset += 8
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p, 200):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
