"""Program for rank-minimal exact tensor decomposition of 6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    # Decompose each 6x6 matrix into nine 2x2 blocks.
    # A = sum_{i,k} A[i,k] E_{i,k} where A[i,k] is a 2x2 block (rows 2i..2i+1, cols 2k..2k+1)
    # B = sum_{k,j} B[k,j] E_{k,j} where B[k,j] is a 2x2 block (rows 2k..2k+1, cols 2j..2j+1)
    # C = A*B = sum_{i,j} (sum_k A[i,k]*B[k,j]) E_{i,j}
    #
    # For each (i,j,k), we need to compute A[i,k]*B[k,j], which is a 2x2 product.
    # Using Strassen's algorithm for 2x2 multiplication, we need 7 scalar multiplications per block product.
    # Total rank = 3*3*3*7 = 189.
    
    # Strassen's 7 products for 2x2 matrices:
    # Given A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (36, 189)
    # V has shape (m*p, rank) = (36, 189)
    # W has shape (p*n, rank) = (36, 189)
    #
    # Indexing convention:
    # U[row, col] where row = i_block*2 + i_row, col = overall column index
    # V[row, col] where row = k_block*2 + k_row, col = overall column index
    # W[row, col] where row = j_block*2 + j_row (output index), col = overall column index
    #
    # For the output C[i,j], the entry at position (output_row, output_col) corresponds to:
    # output_row = j_block*2 + j_row (since C has dimensions n x p, and W is indexed by [k*n+i, r] in schoolbook,
    # but here we need to be careful with the trace convention)
    
    # Let me re-read the trace convention: W[k*n+i, r]
    # In schoolbook: C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor T[i,k,j] = delta (for the multiplication tensor)
    # U[i*m+j, c] = coefficient for A[i,j]
    # V[j*p+k, c] = coefficient for B[j,k]
    # W[k*n+i, c] = coefficient for C[i,k]
    #
    # Wait, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    # So W is indexed as W[k*n + i, r] where the row index encodes (output_row=i, output_col=k)?
    # Or (output_col=k, output_row=i)?
    #
    # In the schoolbook code:
    # U[i*m+j, col] = 1  -> A[i,j]
    # V[j*p+k, col] = 1  -> B[j,k]
    # W[k*n+i, col] = 1  -> C[i,k]
    #
    # So the output matrix C has entries C[i,k] where i is row, k is column.
    # W row index = k*n + i, which means the first index in the row is k (column of C), second is i (row of C).
    # So W[k*n+i] corresponds to C[i,k].
    
    # Now for our block decomposition:
    # A has blocks A[i,k] at rows 2i..2i+1, cols 2k..2k+1
    # B has blocks B[k,j] at rows 2k..2k+1, cols 2j..2j+1
    # C has blocks C[i,j] at rows 2i..2i+1, cols 2j..2j+1
    #
    # For a given (i_block, j_block, k_block), we compute A[i_block,k_block] * B[k_block,j_block]
    # This contributes to C[i_block,j_block].
    #
    # In the U matrix, the row corresponding to A[i_block,k_block][a,b] is:
    #   row_U = (i_block*2 + a)*m + (k_block*2 + b) = (i_block*2 + a)*6 + k_block*2 + b
    #
    # In the V matrix, the row corresponding to B[k_block,j_block][c,d] is:
    #   row_V = (k_block*2 + c)*p + (j_block*2 + d) = (k_block*2 + c)*6 + j_block*2 + d
    #
    # In the W matrix, the row corresponding to C[i_block,j_block][e,f] is:
    #   row_W = (j_block*2 + f)*n + (i_block*2 + e) = (j_block*2 + f)*6 + i_block*2 + e
    #   (since W[k*n+i] corresponds to C[i,k], so k=j_block*2+f, i=i_block*2+e)
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                # Strassen products for A[i_block,k_block] * B[k_block,j_block]
                # A block entries: A11 = A[i_block,k_block][0,0], A12 = A[i_block,k_block][0,1]
                #                  A21 = A[i_block,k_block][1,0], A22 = A[i_block,k_block][1,1]
                # B block entries: B11 = B[k_block,j_block][0,0], B12 = B[k_block,j_block][0,1]
                #                  B21 = B[k_block,j_block][1,0], B22 = B[k_block,j_block][1,1]
                
                # Define helper to get U row index for A[i_block,k_block][a,b]
                def u_row(a, b):
                    return (i_block * 2 + a) * m + (k_block * 2 + b)
                
                # Define helper to get V row index for B[k_block,j_block][c,d]
                def v_row(c, d):
                    return (k_block * 2 + c) * p + (j_block * 2 + d)
                
                # Define helper to get W row index for C[i_block,j_block][e,f]
                def w_row(e, f):
                    return (j_block * 2 + f) * n + (i_block * 2 + e)
                
                # M1 = (A11 + A22) * (B11 + B22)
                # U factors: A11, A22
                # V factors: B11, B22
                # W factors: C11, C22 (from C11 = M1 + M4 - M5 + M7, C22 = M1 - M2 + M3 + M6)
                # So M1 contributes +1 to C11 and +1 to C22
                U[u_row(0, 0)][col] = 1  # A11
                U[u_row(1, 1)][col] = 1  # A22
                V[v_row(0, 0)][col] = 1  # B11
                V[v_row(1, 1)][col] = 1  # B22
                W[w_row(0, 0)][col] = 1  # C11
                W[w_row(1, 1)][col] = 1  # C22
                col += 1
                
                # M2 = (A21 + A22) * B11
                U[u_row(1, 0)][col] = 1  # A21
                U[u_row(1, 1)][col] = 1  # A22
                V[v_row(0, 0)][col] = 1  # B11
                W[w_row(1, 0)][col] = 1  # C21
                W[w_row(1, 1)][col] = -1  # C22
                col += 1
                
                # M3 = A11 * (B12 - B22)
                U[u_row(0, 0)][col] = 1  # A11
                V[v_row(0, 1)][col] = 1  # B12
                V[v_row(1, 1)][col] = -1  # B22
                W[w_row(0, 1)][col] = 1  # C12
                W[w_row(1, 1)][col] = 1  # C22
                col += 1
                
                # M4 = A22 * (B21 - B11)
                U[u_row(1, 1)][col] = 1  # A22
                V[v_row(1, 0)][col] = 1  # B21
                V[v_row(0, 0)][col] = -1  # B11
                W[w_row(0, 0)][col] = 1  # C11
                W[w_row(1, 0)][col] = 1  # C21
                col += 1
                
                # M5 = (A11 + A12) * B22
                U[u_row(0, 0)][col] = 1  # A11
                U[u_row(0, 1)][col] = 1  # A12
                V[v_row(1, 1)][col] = 1  # B22
                W[w_row(0, 0)][col] = -1  # C11
                W[w_row(0, 1)][col] = 1  # C12
                col += 1
                
                # M6 = (A21 - A11) * (B11 + B12)
                U[u_row(1, 0)][col] = 1  # A21
                U[u_row(0, 0)][col] = -1  # A11
                V[v_row(0, 0)][col] = 1  # B11
                V[v_row(0, 1)][col] = 1  # B12
                W[w_row(1, 1)][col] = 1  # C22
                col += 1
                
                # M7 = (A12 - A22) * (B21 + B22)
                U[u_row(0, 1)][col] = 1  # A12
                U[u_row(1, 1)][col] = -1  # A22
                V[v_row(1, 0)][col] = 1  # B21
                V[v_row(1, 1)][col] = 1  # B22
                W[w_row(0, 0)][col] = 1  # C11
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
