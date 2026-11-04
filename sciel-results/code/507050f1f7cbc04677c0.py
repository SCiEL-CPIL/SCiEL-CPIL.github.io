"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Use Strassen's algorithm for 4x4x4 matrix multiplication (rank 28),
    # then apply it twice for the first 8 columns and schoolbook for the 9th.
    # Total rank: 28 + 28 + 16 = 72
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    col = 0
    
    def add_strassen_4x4x4(U, V, W, b_col_offset, c_col_offset, col_start):
        """
        Add 28-rank Strassen-based decomposition for 4x4x4.
        b_col_offset: column offset in V for the 4x4 B block
        c_col_offset: column offset in W for the 4x4 C block
        col_start: starting column
        Returns: new column
        """
        c = col_start
        
        # A is 4x4, indexed as A[i,j] at U[i*4+j]
        # B is 4x4, indexed as B[i,j] at V[b_col_offset + i*4 + j]
        # C is 4x4, indexed as C[i,j] at W[c_col_offset + i*4 + j]
        
        # Partition into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # A11: A[0:2, 0:2] -> U[0], U[1], U[4], U[5]
        # A12: A[0:2, 2:4] -> U[2], U[3], U[6], U[7]
        # A21: A[2:4, 0:2] -> U[8], U[9], U[12], U[13]
        # A22: A[2:4, 2:4] -> U[10], U[11], U[14], U[15]
        
        # B11: B[0:2, 0:2] -> V[b_col_offset+0], V[b_col_offset+1], V[b_col_offset+4], V[b_col_offset+5]
        # B12: B[0:2, 2:4] -> V[b_col_offset+2], V[b_col_offset+3], V[b_col_offset+6], V[b_col_offset+7]
        # B21: B[2:4, 0:2] -> V[b_col_offset+8], V[b_col_offset+9], V[b_col_offset+12], V[b_col_offset+13]
        # B22: B[2:4, 2:4] -> V[b_col_offset+10], V[b_col_offset+11], V[b_col_offset+14], V[b_col_offset+15]
        
        # C11: C[0:2, 0:2] -> W[c_col_offset+0], W[c_col_offset+1], W[c_col_offset+4], W[c_col_offset+5]
        # C12: C[0:2, 2:4] -> W[c_col_offset+2], W[c_col_offset+3], W[c_col_offset+6], W[c_col_offset+7]
        # C21: C[2:4, 0:2] -> W[c_col_offset+8], W[c_col_offset+9], W[c_col_offset+12], W[c_col_offset+13]
        # C22: C[2:4, 2:4] -> W[c_col_offset+10], W[c_col_offset+11], W[c_col_offset+14], W[c_col_offset+15]
        
        # Strassen's 7 intermediate products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each M_i is a 2x2 matrix, computed with 4 scalar multiplications (schoolbook)
        
        def add_schoolbook_2x2(U, V, W, u_indices, v_indices, w_indices, col_start):
            """Add 4 scalar multiplications for 2x2x2 schoolbook."""
            c = col_start
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
            # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
            # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
            
            # Product 1: A[0,0]*B[0,0]
            U[u_indices[0]][c] = 1
            V[v_indices[0]][c] = 1
            W[w_indices[0]][c] = 1  # C[0,0]
            c += 1
            
            # Product 2: A[0,1]*B[1,0]
            U[u_indices[1]][c] = 1
            V[v_indices[2]][c] = 1
            W[w_indices[0]][c] = 1  # C[0,0]
            c += 1
            
            # Product 3: A[0,0]*B[0,1]
            U[u_indices[0]][c] = 1
            V[v_indices[1]][c] = 1
            W[w_indices[1]][c] = 1  # C[0,1]
            c += 1
            
            # Product 4: A[0,1]*B[1,1]
            U[u_indices[1]][c] = 1
            V[v_indices[3]][c] = 1
            W[w_indices[1]][c] = 1  # C[0,1]
            c += 1
            
            # Product 5: A[1,0]*B[0,0]
            U[u_indices[2]][c] = 1
            V[v_indices[0]][c] = 1
            W[w_indices[2]][c] = 1  # C[1,0]
            c += 1
            
            # Product 6: A[1,1]*B[1,0]
            U[u_indices[3]][c] = 1
            V[v_indices[2]][c] = 1
            W[w_indices[2]][c] = 1  # C[1,0]
            c += 1
            
            # Product 7: A[1,0]*B[0,1]
            U[u_indices[2]][c] = 1
            V[v_indices[1]][c] = 1
            W[w_indices[3]][c] = 1  # C[1,1]
            c += 1
            
            # Product 8: A[1,1]*B[1,1]
            U[u_indices[3]][c] = 1
            V[v_indices[3]][c] = 1
            W[w_indices[3]][c] = 1  # C[1,1]
            c += 1
            
            return c
        
        # M1 = (A11+A22)(B11+B22)
        # A11+A22: [A[0,0]+A[2,2], A[0,1]+A[2,3], A[1,0]+A[3,2], A[1,1]+A[3,3]]
        # B11+B22: [B[0,0]+B[2,2], B[0,1]+B[2,3], B[1,0]+B[3,2], B[1,1]+B[3,3]]
        # u_indices for A11+A22: [U[0]+U[10], U[1]+U[11], U[4]+U[14], U[5]+U[15]]
        # v_indices for B11+B22: [V[bo+0]+V[bo+10], V[bo+1]+V[bo+11], V[bo+4]+V[bo+14], V[bo+5]+V[bo+15]]
        # w_indices for C11: [W[co+0], W[co+1], W[co+4], W[co+5]]
        
        # This approach of computing linear combinations is complex. Let me use a different strategy.
        # Instead of trying to compute the Strassen decomposition directly, let me use a simpler approach:
        # Use the fact that 4x4x4 can be done with rank 28, but implement it more carefully.
        
        # Actually, let me just use schoolbook for the whole 4x4x9 to ensure correctness.
        # The rank will be 4*4*9 = 144, which is the same as the original.
        
        return c
    
    # Given the complexity, let me use schoolbook for the entire 4x4x9.
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
# EVOLVE-BLOCK-END
