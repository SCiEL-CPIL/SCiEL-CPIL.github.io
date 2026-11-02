import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook rank-72 baseline
    def schoolbook():
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
    
    # Strassen-based decomposition: rank 66
    # Strategy:
    # Split p=6 into three blocks of 2 columns: B = [B0 B1 B2], each 4x2
    # For each block c in 0,1,2:
    #   C[:, 2c:2c+2] = A * Bc (3x2)
    #   Split A into A_top (2x4) and A_bot (1x4)
    #   C_top = A_top * Bc (2x2)
    #   C_bot = A_bot * Bc (1x2)
    #   
    #   For C_top (2x2x4): split inner dim m=4 into 2+2
    #     C_top = A_top_left * Bc_top + A_top_right * Bc_bot
    #     Each is 2x2x2 -> use Strassen (rank 7 each)
    #     Total: 14
    #   
    #   For C_bot (1x2x4): split inner dim m=4 into 2+2
    #     C_bot = A_bot_left * Bc_top + A_bot_right * Bc_bot
    #     Each is 1x2x2 -> schoolbook (rank 4 each)
    #     Total: 8
    #   
    #   Per block: 14 + 8 = 22
    #   Total: 3 * 22 = 66

    def strassen_2x2x2(base_col, U, V, W):
        """
        Add Strassen decomposition for 2x2x2 multiplication starting at base_col.
        Returns the next column index.
        
        A is 2x2, B is 2x2, C is 2x2.
        C[i,k] = sum_j A[i,j] B[j,k]
        
        Strassen's 7 multiplications:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A01-A11)(B01+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # We need to map local indices to global indices.
        # The caller provides the row/col offsets for A, B, C blocks.
        # For now, we'll handle this in the main loop.
        pass
    
    def build_strassen_decomposition():
        """Build the full rank-66 decomposition."""
        rank = 66
        U = [[0] * rank for _ in range(n * m)]  # (12, 66)
        V = [[0] * rank for _ in range(m * p)]  # (24, 66)
        W = [[0] * rank for _ in range(p * n)]  # (18, 66)
        
        col = 0
        
        for c in range(3):  # 3 column blocks of B, each 4x2
            # Bc is rows 0-3 of B, columns 2c and 2c+1
            # A_top is rows 0-1 of A (2x4), A_bot is row 2 of A (1x4)
            
            # C_top = A_top * Bc (2x2)
            # Split inner dim: A_top = [A_top_L A_top_R] (each 2x2)
            #                Bc = [Bc_T; Bc_B] (each 2x2)
            # C_top = A_top_L * Bc_T + A_top_R * Bc_B
            
            # Term 1: A_top_L * Bc_T (2x2x2) - Strassen
            # A_top_L: A rows 0-1, cols 0-1
            # Bc_T: B rows 0-1, cols 2c, 2c+1
            # C_top_TL: C rows 0-1, cols 2c, 2c+1
            
            # Strassen for A_top_L * Bc_T
            # Local A indices: (i, j) -> global U index: i*m + j = i*4 + j
            # Local B indices: (j, k) -> global V index: j*p + (2c+k) = j*6 + 2c + k
            # Local C indices: (i, k) -> global W index: (2c+k)*n + i = (2c+k)*3 + i
            
            # M1 = (A00+A11)(B00+B11)
            # U: A00=(0,0)->idx 0, A11=(1,1)->idx 5
            # V: B00=(0,2c)->idx 2c, B11=(1,2c+1)->idx 6+2c+1=7+2c
            # W: C00=(2c,0)->idx 2c*3+0=6c, C11=(2c+1,1)->idx (2c+1)*3+1=6c+4
            
            # For each Strassen multiplication, we need 7 terms
            # Let's define a helper to add a Strassen 2x2x2 block
            
            def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset, c_row_offset):
                """
                Add Strassen decomposition for multiplying A[a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2]
                by B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
                into C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
                
                Returns next column index.
                """
                nonlocal col
                
                # Local to global index mappings
                # A[i,j] -> U[(a_row_offset+i)*m + (a_col_offset+j)]
                # B[j,k] -> V[(b_row_offset+j)*p + (b_col_offset+k)]
                # C[i,k] -> W[(c_col_offset+k)*n + (c_row_offset+i)]
                
                # Strassen's 7 intermediate products:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B10-B11)
                # M4 = A11(B01-B00)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B10)
                # M7 = (A01-A11)(B01+B11)
                
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # Each Mt is a product of a linear combination of A entries and a linear combination of B entries.
                # We represent each Mt as one column in U, V, W.
                
                # M1: (A00+A11)(B00+B11)
                U[(a_row_offset+0)*m + (a_col_offset+0)][col] = 1
                U[(a_row_offset+1)*m + (a_col_offset+1)][col] = 1
                V[(b_row_offset+0)*p + (b_col_offset+0)][col] = 1
                V[(b_row_offset+1)*p + (b_col_offset+1)][col] = 1
                # C00 += M1: W[(c_col_offset+0)*n + (c_row_offset+0)]
                W[(c_col_offset+0)*n + (c_row_offset+0)][col] = 1
                # C11 += M1: W[(c_col_offset+1)*n + (c_row_offset+1)]
                W[(c_col_offset+1)*n + (c_row_offset+1)][col] = 1
                col += 1
                
                # M2: (A10+A11)B00
                U[(a_row_offset+1)*m + (a_col_offset+0)][col] = 1
                U[(a_row_offset+1)*m + (a_col_offset+1)][col] = 1
                V[(b_row_offset+0)*p + (b_col_offset+0)][col] = 1
                # C10 += M2: W[(c_col_offset+0)*n + (c_row_offset+1)]
                W[(c_col_offset+0)*n + (c_row_offset+1)][col] = 1
                # C11 -= M2: W[(c_col_offset+1)*n + (c_row_offset+1)]
                W[(c_col_offset+1)*n + (c_row_offset+1)][col] = -1
                col += 1
                
                # M3: A00(B10-B11)
                U[(a_row_offset+0)*m + (a_col_offset+0)][col] = 1
                V[(b_row_offset+1)*p + (b_col_offset+0)][col] = 1
                V[(b_row_offset+1)*p + (b_col_offset+1)][col] = -1
                # C01 += M3: W[(c_col_offset+1)*n + (c_row_offset+0)]
                W[(c_col_offset+1)*n + (c_row_offset+0)][col] = 1
                # C11 += M3: W[(c_col_offset+1)*n + (c_row_offset+1)]
                W[(c_col_offset+1)*n + (c_row_offset+1)][col] = 1
                col += 1
                
                # M4: A11(B01-B00)
                U[(a_row_offset+1)*m + (a_col_offset+1)][col] = 1
                V[(b_row_offset+0)*p + (b_col_offset+1)][col] = 1
                V[(b_row_offset+0)*p + (b_col_offset+0)][col] = -1
                # C00 += M4: W[(c_col_offset+0)*n + (c_row_offset+0)]
                W[(c_col_offset+0)*n + (c_row_offset+0)][col] = 1
                # C10 += M4: W[(c_col_offset+0)*n + (c_row_offset+1)]
                W[(c_col_offset+0)*n + (c_row_offset+1)][col] = 1
                col += 1
                
                # M5: (A00+A01)B11
                U[(a_row_offset+0)*m + (a_col_offset+0)][col] = 1
                U[(a_row_offset+0)*m + (a_col_offset+1)][col] = 1
                V[(b_row_offset+1)*p + (b_col_offset+1)][col] = 1
                # C00 -= M5: W[(c_col_offset+0)*n + (c_row_offset+0)]
                W[(c_col_offset+0)*n + (c_row_offset+0)][col] = -1
                # C01 += M5: W[(c_col_offset+1)*n + (c_row_offset+0)]
                W[(c_col_offset+1)*n + (c_row_offset+0)][col] = 1
                col += 1
                
                # M6: (A10-A00)(B00+B10)
                U[(a_row_offset+1)*m + (a_col_offset+0)][col] = 1
                U[(a_row_offset+0)*m + (a_col_offset+0)][col] = -1
                V[(b_row_offset+0)*p + (b_col_offset+0)][col] = 1
                V[(b_row_offset+1)*p + (b_col_offset+0)][col] = 1
                # C11 += M6: W[(c_col_offset+1)*n + (c_row_offset+1)]
                W[(c_col_offset+1)*n + (c_row_offset+1)][col] = 1
                col += 1
                
                # M7: (A01-A11)(B01+B11)
                U[(a_row_offset+0)*m + (a_col_offset+1)][col] = 1
                U[(a_row_offset+1)*m + (a_col_offset+1)][col] = -1
                V[(b_row_offset+0)*p + (b_col_offset+1)][col] = 1
                V[(b_row_offset+1)*p + (b_col_offset+1)][col] = 1
                # C00 += M7: W[(c_col_offset+0)*n + (c_row_offset+0)]
                W[(c_col_offset+0)*n + (c_row_offset+0)][col] = 1
                col += 1
                
                return col
            
            # Add Strassen for A_top_L * Bc_T
            # A_top_L: rows 0-1, cols 0-1 of A
            # Bc_T: rows 0-1, cols 2c to 2c+1 of B
            # C output: rows 0-1, cols 2c to 2c+1 of C
            col = add_strassen(a_row_offset=0, a_col_offset=0, b_row_offset=0, b_col_offset=2*c, c_col_offset=2*c, c_row_offset=0)
            
            # Add Strassen for A_top_R * Bc_B
            # A_top_R: rows 0-1, cols 2-3 of A
            # Bc_B: rows 2-3, cols 2c to 2c+1 of B
            # C output: rows 0-1, cols 2c to 2c+1 of C
            col = add_strassen(a_row_offset=0, a_col_offset=2, b_row_offset=2, b_col_offset=2*c, c_col_offset=2*c, c_row_offset=0)
            
            # C_bot = A_bot * Bc (1x2)
            # A_bot: row 2 of A (1x4)
            # Split inner dim: A_bot = [A_bot_L A_bot_R] (each 1x2)
            #                Bc = [Bc_T; Bc_B] (each 2x2)
            # C_bot = A_bot_L * Bc_T + A_bot_R * Bc_B
            
            # Term 1: A_bot_L * Bc_T (1x2x2) - schoolbook, rank 4
            # A_bot_L: A row 2, cols 0-1
            # Bc_T: B rows 0-1, cols 2c, 2c+1
            # C output: C row 2, cols 2c, 2c+1
            
            for j in range(2):  # j=0,1 (local B row)
                for k in range(2):  # k=0,1 (local B col)
                    U[2*m + j][col] = 1  # A[2, j]
                    V[(0+j)*p + (2*c+k)][col] = 1  # B[j, 2c+k]
                    W[(2*c+k)*n + 2][col] = 1  # C[2c+k, 2]
                    col += 1
            
            # Term 2: A_bot_R * Bc_B (1x2x2) - schoolbook, rank 4
            # A_bot_R: A row 2, cols 2-3
            # Bc_B: B rows 2-3, cols 2c, 2c+1
            # C output: C row 2, cols 2c, 2c+1
            
            for j in range(2):  # j=0,1 (local B row within bottom half)
                for k in range(2):  # k=0,1 (local B col)
                    U[2*m + (2+j)][col] = 1  # A[2, 2+j]
                    V[(2+j)*p + (2*c+k)][col] = 1  # B[2+j, 2c+k]
                    W[(2*c+k)*n + 2][col] = 1  # C[2c+k, 2]
                    col += 1
        
        # Verify col == rank
        if col != rank:
            # Fallback to schoolbook
            return schoolbook()
        
        return {"U": U, "V": V, "W": W}
    
    # Try the Strassen-based decomposition
    try:
        result = build_strassen_decomposition()
        
        # Verify the decomposition
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return result
        else:
            # Fallback to schoolbook
            return schoolbook()
    except Exception:
        return schoolbook()
# EVOLVE-BLOCK-END
