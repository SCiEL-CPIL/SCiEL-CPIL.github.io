"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition based on splitting the middle dimension m=4 into two blocks of 2.
    # This reduces the problem to two instances of 2x2x7 multiplication.
    # For 2x2x7, we split the last dimension p=7 into three blocks of 2 and one block of 1.
    # This reduces to three instances of 2x2x2 and one instance of 2x2x1.
    # 2x2x2 uses Strassen's algorithm (rank 7).
    # 2x2x1 uses schoolbook (rank 4).
    # Total rank for 2x2x7 is 3*7 + 4 = 25.
    # Total rank for 2x4x7 is 2*25 = 50.
    
    # Helper to create a rank-1 tensor contribution
    def add_term(U, V, W, u_idx, v_idx, w_idx, val):
        U[u_idx][rank_col] = val
        V[v_idx][rank_col] = val
        W[w_idx][rank_col] = val

    # We need to manage the rank column index
    global rank_col
    rank_col = 0
    
    # We will build the matrices U, V, W
    # U size: n*m = 8
    # V size: m*p = 28
    # W size: p*n = 14
    
    U = [[0] * 50 for _ in range(8)]
    V = [[0] * 50 for _ in range(28)]
    W = [[0] * 50 for _ in range(14)]
    
    # Indices for A (2x4): A[i, j] -> U[i*4 + j]
    # Indices for B (4x7): B[j, k] -> V[j*7 + k]
    # Indices for C (2x7): C[i, k] -> W[k*2 + i] (Trace convention W[k*n+i])
    
    # We process two blocks of A: A_L (cols 0,1) and A_R (cols 2,3)
    for block_a in range(2):
        # block_a=0 -> A cols 0,1. block_a=1 -> A cols 2,3.
        a_offset = block_a * 2
        
        # Process 2x2x7 for this block
        # Split B rows (j) corresponding to this block?
        # No, B is 4x7.
        # For A_L (cols 0,1), we multiply by B rows 0,1.
        # For A_R (cols 2,3), we multiply by B rows 2,3.
        
        b_row_offset = block_a * 2
        
        # Now we have a 2x2x7 multiplication:
        # A_block (2x2), B_block (2x7), C_block (2x7)
        
        # Split B_block into 3 blocks of 2 cols and 1 block of 1 col.
        for block_b in range(4):
            # block_b=0 -> B cols 0,1
            # block_b=1 -> B cols 2,3
            # block_b=2 -> B cols 4,5
            # block_b=3 -> B col 6
            
            b_col_offset = 0
            if block_b == 0: b_col_offset = 0
            elif block_b == 1: b_col_offset = 2
            elif block_b == 2: b_col_offset = 4
            elif block_b == 3: b_col_offset = 6
            
            c_col_offset = b_col_offset
            
            if block_b < 3:
                # 2x2x2 multiplication
                # A indices: (a_offset+0, b_row_offset+0), etc.
                # B indices: (b_row_offset+0, b_col_offset+0), etc.
                # C indices: (c_col_offset+0, a_offset+0), etc.
                
                # Strassen terms
                # M1 = A11(B11 + B22)
                # U: A11, V: B11, W: C11
                # We need to map to global indices
                
                # A11 is A[a_offset, b_row_offset] -> U[a_offset*4 + b_row_offset]
                # A12 is A[a_offset, b_row_offset+1] -> U[a_offset*4 + b_row_offset+1]
                # A21 is A[a_offset+1, b_row_offset] -> U[(a_offset+1)*4 + b_row_offset]
                # A22 is A[a_offset+1, b_row_offset+1] -> U[(a_offset+1)*4 + b_row_offset+1]
                
                # B11 is B[b_row_offset, b_col_offset] -> V[b_row_offset*7 + b_col_offset]
                # B12 is B[b_row_offset, b_col_offset+1] -> V[b_row_offset*7 + b_col_offset+1]
                # B21 is B[b_row_offset+1, b_col_offset] -> V[(b_row_offset+1)*7 + b_col_offset]
                # B22 is B[b_row_offset+1, b_col_offset+1] -> V[(b_row_offset+1)*7 + b_col_offset+1]
                
                # C11 is C[a_offset, b_col_offset] -> W[b_col_offset*2 + a_offset]
                # C12 is C[a_offset, b_col_offset+1] -> W[(b_col_offset+1)*2 + a_offset]
                # C21 is C[a_offset+1, b_col_offset] -> W[b_col_offset*2 + a_offset+1]
                # C22 is C[a_offset+1, b_col_offset+1] -> W[(b_col_offset+1)*2 + a_offset+1]
                
                u11 = a_offset*4 + b_row_offset
                u12 = a_offset*4 + b_row_offset+1
                u21 = (a_offset+1)*4 + b_row_offset
                u22 = (a_offset+1)*4 + b_row_offset+1
                
                v11 = b_row_offset*7 + b_col_offset
                v12 = b_row_offset*7 + b_col_offset+1
                v21 = (b_row_offset+1)*7 + b_col_offset
                v22 = (b_row_offset+1)*7 + b_col_offset+1
                
                w11 = b_col_offset*2 + a_offset
                w12 = (b_col_offset+1)*2 + a_offset
                w21 = b_col_offset*2 + a_offset+1
                w22 = (b_col_offset+1)*2 + a_offset+1
                
                # M1 = A11(B11 + B22)
                U[u11][rank_col] = 1
                V[v11][rank_col] = 1
                W[w11][rank_col] = 1
                V[v22][rank_col] = 1
                W[w11][rank_col] += 1
                rank_col += 1
                
                # M2 = (A11 + A21)B22
                U[u11][rank_col] = 1
                U[u21][rank_col] = 1
                V[v22][rank_col] = 1
                W[w11][rank_col] -= 1
                W[w21][rank_col] += 1
                rank_col += 1
                
                # M3 = (A12 + A22)B11
                U[u12][rank_col] = 1
                U[u22][rank_col] = 1
                V[v11][rank_col] = 1
                W[w21][rank_col] += 1
                W[w22][rank_col] -= 1
                rank_col += 1
                
                # M4 = A22(B22 - B11)
                U[u22][rank_col] = 1
                V[v22][rank_col] = 1
                W[w11][rank_col] += 1
                W[w22][rank_col] += 1
                V[v11][rank_col] = -1
                W[w11][rank_col] -= 1
                W[w22][rank_col] -= 1
                rank_col += 1
                
                # M5 = A11(B12 + B22)
                U[u11][rank_col] = 1
                V[v12][rank_col] = 1
                W[w12][rank_col] += 1
                V[v22][rank_col] = 1
                W[w12][rank_col] += 1
                rank_col += 1
                
                # M6 = (A12 + A22)(B21 - B11)
                U[u12][rank_col] = 1
                U[u22][rank_col] = 1
                V[v21][rank_col] = 1
                W[w12][rank_col] += 1
                W[w22][rank_col] += 1
                V[v11][rank_col] = -1
                W[w12][rank_col] -= 1
                W[w22][rank_col] -= 1
                rank_col += 1
                
                # M7 = (A11 - A21)(B11 + B21)
                U[u11][rank_col] = 1
                U[u21][rank_col] = -1
                V[v11][rank_col] = 1
                W[w21][rank_col] += 1
                W[w22][rank_col] += 1
                V[v21][rank_col] = 1
                W[w21][rank_col] += 1
                W[w22][rank_col] += 1
                rank_col += 1
                
            else:
                # 2x2x1 multiplication (Schoolbook, rank 4)
                # C11 = A11B11 + A12B21
                # C21 = A21B11 + A22B21
                
                u11 = a_offset*4 + b_row_offset
                u12 = a_offset*4 + b_row_offset+1
                u21 = (a_offset+1)*4 + b_row_offset
                u22 = (a_offset+1)*4 + b_row_offset+1
                
                v11 = b_row_offset*7 + b_col_offset
                v21 = (b_row_offset+1)*7 + b_col_offset
                
                w11 = b_col_offset*2 + a_offset
                w21 = b_col_offset*2 + a_offset+1
                
                # Term 1: A11 B11 C11
                U[u11][rank_col] = 1
                V[v11][rank_col] = 1
                W[w11][rank_col] = 1
                rank_col += 1
                
                # Term 2: A12 B21 C11
                U[u12][rank_col] = 1
                V[v21][rank_col] = 1
                W[w11][rank_col] = 1
                rank_col += 1
                
                # Term 3: A21 B11 C21
                U[u21][rank_col] = 1
                V[v11][rank_col] = 1
                W[w21][rank_col] = 1
                rank_col += 1
                
                # Term 4: A22 B21 C21
                U[u22][rank_col] = 1
                V[v21][rank_col] = 1
                W[w21][rank_col] = 1
                rank_col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
