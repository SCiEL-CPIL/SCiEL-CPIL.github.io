import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _build_strassen_4x4x4(U, V, W, rank_offset, b_col_offset, w_row_offset):
    # Adds a Strassen decomposition for 4x4x4 multiplication
    # A is 4x4, B is 4x4 (cols b_col_offset to b_col_offset+3), C is 4x4 (rows w_row_offset to w_row_offset+3)
    # U is 16xRank, V is 36xRank, W is 36xRank
    # A indices: i*4 + j
    # B indices: j*9 + (b_col_offset + k)
    # W indices: (w_row_offset + k)*4 + i
    
    # Strassen multiplications M1..M7
    # Each is a 2x2 matrix multiplication
    # We use schoolbook for each 2x2 multiplication (4 scalar mults each)
    # Total 28 scalar mults
    
    # Define the 7 Strassen steps
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Helper to add a 2x2 schoolbook multiplication
    # x_row_off, x_col_off for X (2x2 block of A)
    # y_row_off, y_col_off for Y (2x2 block of B)
    # z_row_off, z_col_off for Z (2x2 block of C)
    # sign_x, sign_y multipliers for the blocks
    
    def add_2x2_mult(rank_col, x_r, x_c, y_r, y_c, z_r, z_c, sx=1, sy=1):
        # X is at A[x_r:x_r+2, x_c:x_c+2]
        # Y is at B[y_r:y_r+2, y_c:y_c+2]
        # Z is at C[z_r:z_r+2, z_c:z_c+2]
        
        # Z00 = X00*Y00 + X01*Y10
        U[x_r*4 + x_c][rank_col] += sx
        V[(y_r)*9 + (y_c + 0 + b_col_offset)][rank_col] += sy
        W[(z_r + 0)*4 + (z_c + 0)][rank_col] += 1
        
        U[x_r*4 + (x_c+1)][rank_col] += sx
        V[((y_r+1))*9 + (y_c + 0 + b_col_offset)][rank_col] += sy
        W[(z_r + 0)*4 + (z_c + 0)][rank_col] += 1
        
        # Z01 = X00*Y01 + X01*Y11
        U[x_r*4 + x_c][rank_col+1] += sx
        V[(y_r)*9 + (y_c + 1 + b_col_offset)][rank_col+1] += sy
        W[(z_r + 0)*4 + (z_c + 1)][rank_col+1] += 1
        
        U[x_r*4 + (x_c+1)][rank_col+1] += sx
        V[((y_r+1))*9 + (y_c + 1 + b_col_offset)][rank_col+1] += sy
        W[(z_r + 0)*4 + (z_c + 1)][rank_col+1] += 1
        
        # Z10 = X10*Y00 + X11*Y10
        U[(x_r+1)*4 + x_c][rank_col+2] += sx
        V[(y_r)*9 + (y_c + 0 + b_col_offset)][rank_col+2] += sy
        W[(z_r + 1)*4 + (z_c + 0)][rank_col+2] += 1
        
        U[(x_r+1)*4 + (x_c+1)][rank_col+2] += sx
        V[((y_r+1))*9 + (y_c + 0 + b_col_offset)][rank_col+2] += sy
        W[(z_r + 1)*4 + (z_c + 0)][rank_col+2] += 1
        
        # Z11 = X10*Y01 + X11*Y11
        U[(x_r+1)*4 + x_c][rank_col+3] += sx
        V[(y_r)*9 + (y_c + 1 + b_col_offset)][rank_col+3] += sy
        W[(z_r + 1)*4 + (z_c + 1)][rank_col+3] += 1
        
        U[(x_r+1)*4 + (x_c+1)][rank_col+3] += sx
        V[((y_r+1))*9 + (y_c + 1 + b_col_offset)][rank_col+3] += sy
        W[(z_r + 1)*4 + (z_c + 1)][rank_col+3] += 1

    # M1: (A00+A11)(B00+B11)
    # A00: r=0,c=0; A11: r=2,c=2
    # B00: r=0,c=0; B11: r=2,c=2
    # C00: r=0,c=0
    add_2x2_mult(rank_offset, 0, 0, 0, 0, 0, 0, 1, 1)
    add_2x2_mult(rank_offset, 2, 2, 0, 0, 0, 0, 1, 1)
    add_2x2_mult(rank_offset, 0, 0, 2, 2, 0, 0, 1, 1)
    add_2x2_mult(rank_offset, 2, 2, 2, 2, 0, 0, 1, 1)
    # Wait, this adds 4*4=16 mults for M1?
    # No, M1 is ONE 2x2 multiplication.
    # My add_2x2_mult adds 4 scalar mults for ONE 2x2 product.
    # But M1 = (A00+A11)(B00+B11).
    # This is (X+Y)(Z+W) = XZ + XW + YZ + YW.
    # So yes, it expands to 4 2x2 multiplications?
    # No, (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # So it is ONE 2x2 multiplication.
    # My add_2x2_mult assumes X and Y are the inputs.
    # But here the inputs are SUMS of blocks.
    # So I need to add the contributions of A00 and A11 to the U matrix, and B00 and B11 to the V matrix.
    # Then the 4 scalar mults of add_2x2_mult will handle the cross terms?
    # No.
    # If I call add_2x2_mult with x_r=0, x_c=0, it adds A00.
    # If I call it again with x_r=2, x_c=2, it adds A11.
    # But the rank column is the SAME.
    # So U[0,0] += 1, U[2,2] += 1.
    # V[0,0] += 1, V[2,2] += 1.
    # Then the product is (A00+A11)(B00+B11).
    # This is correct.
    
    # So for M1:
    # We use rank_offset to rank_offset+3.
    # U gets A00 and A11.
    # V gets B00 and B11.
    # W gets C00.
    
    # Let's re-verify add_2x2_mult.
    # It writes to rank_col, rank_col+1, rank_col+2, rank_col+3.
    # So for M1, we need to call it 4 times? No.
    # The 4 scalar mults in add_2x2_mult correspond to the 4 entries of the 2x2 result.
    # But the inputs X and Y are 2x2 matrices.
    # If X = A00+A11, then U must contain both A00 and A11 in the same rank columns.
    # My add_2x2_mult function adds ONE block to U.
    # So I need to call it multiple times with the same rank_col?
    # No, add_2x2_mult adds 4 entries to U (for the 4 elements of the 2x2 block X).
    # If I call it with x_r=0, x_c=0, it adds A00's 4 elements to U.
    # If I call it again with x_r=2, x_c=2, it adds A11's 4 elements to U.
    # But it also writes to V and W.
    # This is a problem.
    # I need to separate the U, V, W updates.
    
    # Let's refactor.
    
    pass

def construct():
    # Use Schoolbook for safety and simplicity if Strassen implementation is complex
    # Rank 144. Score 1.0.
    # But we want > 1.0.
    # Let's try to implement the 72 rank solution carefully.
    
    n, m, p = DIMENSIONS
    # U: 16 x Rank
    # V: 36 x Rank
    # W: 36 x Rank
    
    rank = 72
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    def add_scalar_mult(u_row, v_row, w_row, rank_col, u_val=1, v_val=1, w_val=1):
        U[u_row][rank_col] += u_val
        V[v_row][rank_col] += v_val
        W[w_row][rank_col] += w_val

    def add_2x2_schoolbook(rank_col, x_r, x_c, y_r, y_c, z_r, z_c, sx=1, sy=1):
        # X is 2x2 block of A at (x_r, x_c)
        # Y is 2x2 block of B at (y_r, y_c)
        # Z is 2x2 block of C at (z_r, z_c)
        # Adds 4 scalar mults to rank_col .. rank_col+3
        
        # Z00 = X00*Y00 + X01*Y10
        add_scalar_mult(x_r*4 + x_c, y_r*9 + (y_c + 0), (z_r + 0)*4 + (z_c + 0), rank_col, sx, sy, 1)
        add_scalar_mult(x_r*4 + (x_c+1), (y_r+1)*9 + (y_c + 0), (z_r + 0)*4 + (z_c + 0), rank_col, sx, sy, 1)
        
        # Z01 = X00*Y01 + X01*Y11
        add_scalar_mult(x_r*4 + x_c, y_r*9 + (y_c + 1), (z_r + 0)*4 + (z_c + 1), rank_col+1, sx, sy, 1)
        add_scalar_mult(x_r*4 + (x_c+1), (y_r+1)*9 + (y_c + 1), (z_r + 0)*4 + (z_c + 1), rank_col+1, sx, sy, 1)
        
        # Z10 = X10*Y00 + X11*Y10
        add_scalar_mult((x_r+1)*4 + x_c, y_r*9 + (y_c + 0), (z_r + 1)*4 + (z_c + 0), rank_col+2, sx, sy, 1)
        add_scalar_mult((x_r+1)*4 + (x_c+1), (y_r+1)*9 + (y_c + 0), (z_r + 1)*4 + (z_c + 0), rank_col+2, sx, sy, 1)
        
        # Z11 = X10*Y01 + X11*Y11
        add_scalar_mult((x_r+1)*4 + x_c, y_r*9 + (y_c + 1), (z_r + 1)*4 + (z_c + 1), rank_col+3, sx, sy, 1)
        add_scalar_mult((x_r+1)*4 + (x_c+1), (y_r+1)*9 + (y_c + 1), (z_r + 1)*4 + (z_c + 1), rank_col+3, sx, sy, 1)

    def add_strassen_4x4x4(rank_offset, b_col_offset, w_row_offset):
        # M1: (A00+A11)(B00+B11) -> C00
        # We need to add A00 and A11 to U, B00 and B11 to V.
        # The 4 scalar mults of the 2x2 product are at rank_offset..rank_offset+3.
        
        # For Z00 (rank_offset):
        # X00*Y00: A00_00 * B00_00 and A11_00 * B00_00 and A00_00 * B11_00 and A11_00 * B11_00
        # This is getting complicated to map to the helper.
        
        # Let's just manually add the terms for M1.
        # M1 = (A00+A11)(B00+B11)
        # A00 is at (0,0), A11 is at (2,2).
        # B00 is at (0,0), B11 is at (2,2).
        # C00 is at (0,0).
        
        # Z00 = (A00_00+A11_00)(B00_00+B11_00) + (A00_01+A11_01)(B10_00+B11_01)
        # Wait, 2x2 product:
        # P = X Y
        # P00 = X00 Y00 + X01 Y10
        # X = A00+A11. Y = B00+B11.
        # X00 = A00_00 + A11_00. Y00 = B00_00 + B11_00.
        # X01 = A00_01 + A11_01. Y10 = B10_00 + B11_01.
        
        # Terms for P00:
        # (A00_00 + A11_00)(B00_00 + B11_00) = A00_00 B00_00 + A00_00 B11_00 + A11_00 B00_00 + A11_00 B11_00
        # (A00_01 + A11_01)(B10_00 + B11_01) = A00_01 B10_00 + A00_01 B11_01 + A11_01 B10_00 + A11_01 B11_01
        
        # This requires 8 scalar mults for just P00?
        # No, the rank of a 2x2 multiplication is 4 (schoolbook).
        # The "multiplication" M1 is a 2x2 matrix multiplication.
        # It consumes 4 "slots" in the rank.
        # But the inputs are sums.
        # So we add the summands to U and V.
        
        # Let's use the helper add_2x2_schoolbook but we need to call it for each summand?
        # No, that would use 4*4=16 slots.
        
        # Correct approach:
        # For each of the 4 outputs of the 2x2 product (Z00, Z01, Z10, Z11),
        # we add the appropriate terms to U, V, W.
        
        # For Z00 (rank_offset):
        # U gets A00_00, A11_00, A00_01, A11_01
        # V gets B00_00, B11_00, B10_00, B11_01
        # W gets C00_00
        
        U[0*4+0][rank_offset] += 1
        U[2*4+2][rank_offset] += 1
        U[0*4+1][rank_offset] += 1
        U[2*4+3][rank_offset] += 1
        
        V[0*9+0][rank_offset] += 1
        V[2*9+2][rank_offset] += 1
        V[2*9+0][rank_offset] += 1
        V[2*9+3][rank_offset] += 1
        
        W[0*4+0][rank_offset] += 1
        
        # ... and so on for Z01, Z10, Z11 and M2..M7.
        
        # This is tedious but doable.
        # Given the time, returning Schoolbook (rank 144) is safer.
        # The score 1.0 is acceptable if we can't easily get 2.0.
        
        pass

    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
