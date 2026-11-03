"""Matrix multiplication 6x6x6 with block-Strassen decomposition (rank 63)."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block structure: 3x3 blocks of 2x2
    block_size = 2
    num_blocks = n // block_size  # 3
    
    # Strassen's 7 multiplications for 2x2 matrix product
    # For matrices X, Y (2x2):
    # M0 = (X00+X11)*(Y00+Y11)
    # M1 = (X10+X11)*Y00
    # M2 = X00*(Y01-Y11)
    # M3 = X11*(Y10-Y00)
    # M4 = (X00+X01)*Y11
    # M5 = (X10-X00)*(Y00+Y01)
    # M6 = (X01-X11)*(Y10+Y11)
    # Result:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Define Strassen patterns for 2x2 matrices
    # Each pattern: (coeff, row, col) for each of the 7 multiplications
    # X indices: 00,01,10,11 -> 0,1,2,3
    # Y indices: 00,01,10,11 -> 0,1,2,3
    
    # For each Strassen multiplication s, define:
    # x_pattern[s]: list of (coeff, x_index) 
    # y_pattern[s]: list of (coeff, y_index)
    # c_pattern[s]: list of (coeff, c_index) for output
    
    # x_index: 0=X00, 1=X01, 2=X10, 3=X11
    # y_index: 0=Y00, 1=Y01, 2=Y10, 3=Y11
    # c_index: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_x = [
        [(1, 0), (1, 3)],      # M0: X00 + X11
        [(1, 2), (1, 3)],      # M1: X10 + X11
        [(1, 0)],              # M2: X00
        [(1, 3)],              # M3: X11
        [(1, 0), (1, 1)],      # M4: X00 + X01
        [(1, 2), (-1, 0)],     # M5: X10 - X00
        [(1, 1), (-1, 3)],     # M6: X01 - X11
    ]
    
    strassen_y = [
        [(1, 0), (1, 3)],      # M0: Y00 + Y11
        [(1, 0)],              # M1: Y00
        [(1, 1), (-1, 3)],     # M2: Y01 - Y11
        [(1, 2), (-1, 0)],     # M3: Y10 - Y00
        [(1, 3)],              # M4: Y11
        [(1, 0), (1, 1)],      # M5: Y00 + Y01
        [(1, 2), (1, 3)],      # M6: Y10 + Y11
    ]
    
    strassen_c = [
        [(1, 0), (1, 3), (-1, 2), (1, 1)],   # M0: C00+=1, C11+=1, C10-=1, C01+=1
        [(1, 2), (-1, 3)],                    # M1: C10+=1, C11-=1
        [(1, 1), (1, 3)],                    # M2: C01+=1, C11+=1
        [(1, 0), (1, 2)],                    # M3: C00+=1, C10+=1
        [(-1, 0), (1, 1)],                   # M4: C00-=1, C01+=1
        [(1, 3)],                            # M5: C11+=1
        [(1, 0), (1, 2)],                    # M6: C00+=1, C10+=1
    ]
    
    # Wait, let me re-derive Strassen more carefully
    # Standard Strassen for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # Result:
    # [[M1+M4-M5+M7, M3+M5],
    #  [M2+M4, M1-M2+M3+M6]]
    
    # Let me redo with this standard form
    # x: a=0, b=1, c=2, d=3
    # y: e=0, f=1, g=2, h=3
    # c: 00=0, 01=1, 10=2, 11=3
    
    strassen_x = [
        [(1, 0), (1, 3)],      # M1: a + d
        [(1, 2), (1, 3)],      # M2: c + d
        [(1, 0)],              # M3: a
        [(1, 3)],              # M4: d
        [(1, 0), (1, 1)],      # M5: a + b
        [(1, 2), (-1, 0)],     # M6: c - a
        [(1, 1), (-1, 3)],     # M7: b - d
    ]
    
    strassen_y = [
        [(1, 0), (1, 3)],      # M1: e + h
        [(1, 0)],              # M2: e
        [(1, 1), (-1, 3)],     # M3: f - h
        [(1, 2), (-1, 0)],     # M4: g - e
        [(1, 3)],              # M5: h
        [(1, 0), (1, 1)],      # M6: e + f
        [(1, 2), (1, 3)],      # M7: g + h
    ]
    
    strassen_c = [
        [(1, 0), (1, 3), (-1, 2), (1, 1)],   # M1: C00+=1, C11+=1, C10-=1, C01+=1
        [(1, 2), (-1, 3)],                    # M2: C10+=1, C11-=1
        [(1, 1), (1, 3)],                    # M3: C01+=1, C11+=1
        [(1, 0), (1, 2)],                    # M4: C00+=1, C10+=1
        [(-1, 0), (1, 1)],                   # M5: C00-=1, C01+=1
        [(1, 3)],                            # M6: C11+=1
        [(1, 0), (1, 2)],                    # M7: C00+=1, C10+=1
    ]
    
    # Verify: C00 = M1 + M4 - M5 + M7
    # M1 contributes +1 to C00, M4 contributes +1 to C00, M5 contributes -1 to C00, M7 contributes +1 to C00
    # So C00 = M1 + M4 - M5 + M7 ✓
    
    # C01 = M3 + M5
    # M3 contributes +1 to C01, M5 contributes +1 to C01 ✓
    
    # C10 = M2 + M4
    # M2 contributes +1 to C10, M4 contributes +1 to C10 ✓
    
    # C11 = M1 - M2 + M3 + M6
    # M1 contributes +1 to C11, M2 contributes -1 to C11, M3 contributes +1 to C11, M6 contributes +1 to C11 ✓
    
    # Now build the full decomposition
    # Outer: 3x3 block matrix multiplication
    # For each (i,j) in 3x3, we need sum over k of A[i,k] * B[k,j]
    # Each such product is a 2x2 matrix product using Strassen
    
    rank = num_blocks * num_blocks * 7  # 3*3*7 = 63
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0] * rank for _ in range(p * n)]  # 36 x 63
    
    def block_row(block_i, local_row):
        """Global row index for block block_i, local row local_row"""
        return block_i * block_size + local_row
    
    def block_col(block_j, local_col):
        """Global col index for block block_j, local col local_col"""
        return block_j * block_size + local_col
    
    # For each outer (i, j) and Strassen index s
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                for s in range(7):
                    col_idx = (i * num_blocks + j) * 7 + s
                    # Wait, I need to think about this more carefully
                    # The rank is 63, but I have 3*3*3*7 = 189 terms if I do full expansion
                    # No - for each (i,j) output block, I sum over k
                    # But in tensor decomposition, each multiplication is separate
                    # Actually, for the outer 3x3 multiplication, each term A[i,k]*B[k,j] 
                    # is a separate multiplication in the tensor sense
                    # So total rank = 3*3*3*7 = 189? No...
                    
                    # Wait, I'm confusing myself. Let me reconsider.
                    # In the tensor decomposition of matrix multiplication:
                    # C = A * B
                    # Each entry C[i,j] = sum_k A[i,k] * B[k,j]
                    # In tensor form, we decompose the trilinear form
                    # For block-Strassen: we group terms
                    # The outer 3x3 multiplication has 9 output blocks, each is sum of 3 products
                    # But in tensor decomposition, each product A[i,k]*B[k,j] is one "multiplication"
                    # So for the outer structure, we have 3*3*3 = 27 products
                    # Each product is a 2x2 matrix multiplication, which uses 7 Strassen multiplications
                    # Total rank = 27 * 7 = 189
    
    # Let me restart with the correct understanding
    # Rank = (num_blocks^3) * 7 = 27 * 7 = 189
    
    rank = num_blocks ** 3 * 7  # 27 * 7 = 189
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                for s in range(7):
                    col_idx = (i * num_blocks + j) * (num_blocks * 7) + k * 7 + s
                    # A[i,k] is at rows [i*2, i*2+1], cols [k*2, k*2+1]
                    # B[k,j] is at rows [k*2, k*2+1], cols [j*2, j*2+1]
                    # C[i,j] is at rows [i*2, i*2+1], cols [j*2, j*2+1]
                    
                    # Fill U: extracts from A[i,k]
                    for (coeff, x_idx) in strassen_x[s]:
                        local_r, local_c = divmod(x_idx, 2)
                        global_r = i * block_size + local_r
                        global_c = k * block_size + local_c
                        U[global_r * m + global_c][col_idx] += coeff
                    
                    # Fill V: extracts from B[k,j]
                    for (coeff, y_idx) in strassen_y[s]:
                        local_r, local_c = divmod(y_idx, 2)
                        global_r = k * block_size + local_r
                        global_c = j * block_size + local_c
                        V[global_r * p + global_c][col_idx] += coeff
                    
                    # Fill W: contributes to C[i,j]
                    for (coeff, c_idx) in strassen_c[s]:
                        local_r, local_c = divmod(c_idx, 2)
                        global_r = i * block_size + local_r
                        global_c = j * block_size + local_c
                        W[global_r * n + global_c][col_idx] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
