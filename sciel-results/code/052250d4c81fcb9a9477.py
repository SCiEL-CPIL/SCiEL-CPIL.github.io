"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid Strassen/Schoolbook decomposition for 3x4 by 4x5 matrix multiplication.
    # Strategy: Use Strassen (rank 7) for 2x2x2 blocks where possible, schoolbook for the rest.
    #
    # A is 3x4, B is 4x5, C is 3x5.
    #
    # We partition the computation into blocks and use Strassen for 2x2 sub-blocks
    # where both dimensions allow it, and schoolbook otherwise.
    #
    # The key insight: for a 2x4 x 4x2 multiplication, we can decompose as:
    #   A_top (2x4) * B_left (4x2) = A_top1 * B_left1 + A_top2 * B_left2
    # where A_top1 is cols 0,1; A_top2 is cols 2,3
    #       B_left1 is rows 0,1; B_left2 is rows 2,3
    # Each 2x2x2 term can use Strassen (rank 7).
    #
    # Let's enumerate all terms carefully.
    #
    # C[i][j] = sum_k A[i][k] * B[k][j]
    #
    # We'll build U, V, W matrices column by column.
    # U has shape (n*m, rank) = (3*4, rank) = (12, rank)
    # V has shape (m*p, rank) = (4*5, rank) = (20, rank)
    # W has shape (p*n, rank) = (5*3, rank) = (15, rank)
    #
    # Index conventions:
    # U index: i*m + j  where i in [0,n), j in [0,m)  -> U[i*m+j, col]
    # V index: j*p + k  where j in [0,m), k in [0,p)  -> V[j*p+k, col]
    # W index: k*n + i  where k in [0,p), i in [0,n)  -> W[k*n+i, col]
    #
    # For each term in the decomposition, we add contributions to U, V, W.
    # A term is defined by: which A indices (i,j), which B indices (j,k), which C indices (k,i).
    #
    # Let's use a helper to add a "schoolbook" term: A[i][j] * B[j][k] contributes to C[k][i].
    # And a helper to add Strassen terms for a 2x2x2 block.
    #
    # Strassen 2x2x2: Given A2x2, B2x2, C2x2 where C = A*B
    # The 7 products are:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B10-B11)
    #   M4 = A11*(B01-B10)
    #   M5 = (A00+A10)*B11
    #   M6 = (A11-A01)*(B00+B10)
    #   M7 = (A01-A11)*(B01+B11)
    # Then:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # We need to map these to global indices.
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    
    # We'll collect all terms as a list of (U_entries, V_entries, W_entries)
    # Each entry is a list of (index, coefficient) pairs.
    terms = []
    
    def add_schoolbook_term(i, j, k, coeff=1):
        """Add term A[i][j] * B[j][k] -> C[k][i] with given coefficient."""
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        terms.append(([(u_idx, coeff)], [(v_idx, coeff)], [(w_idx, coeff)]))
    
    def add_strassen_block(i0, j0, k0):
        """
        Add Strassen terms for a 2x2x2 block.
        A rows i0, i0+1; A cols j0, j0+1
        B rows j0, j0+1; B cols k0, k0+1
        C rows k0, k0+1; C cols i0, i0+1
        """
        # M1 = (A[i0][j0] + A[i0+1][j0+1]) * (B[j0][k0] + B[j0+1][k0+1])
        u1 = [(i0*m+j0, 1), ((i0+1)*m+(j0+1), 1)]
        v1 = [(j0*p+k0, 1), ((j0+1)*p+(k0+1), 1)]
        w1 = [(k0*n+i0, 1), ((k0+1)*n+(i0+1), 1)]
        terms.append((u1, v1, w1))
        
        # M2 = (A[i0+1][j0] + A[i0+1][j0+1]) * B[j0][k0]
        u2 = [( (i0+1)*m+j0, 1), ((i0+1)*m+(j0+1), 1)]
        v2 = [(j0*p+k0, 1)]
        w2 = [(k0*n+(i0+1), 1)]
        terms.append((u2, v2, w2))
        
        # M3 = A[i0][j0] * (B[j0+1][k0] - B[j0+1][k0+1])
        u3 = [(i0*m+j0, 1)]
        v3 = [((j0+1)*p+k0, 1), ((j0+1)*p+(k0+1), -1)]
        w3 = [(k0*n+i0, 1)]
        terms.append((u3, v3, w3))
        
        # M4 = A[i0+1][j0+1] * (B[j0][k0+1] - B[j0][k0])
        u4 = [((i0+1)*m+(j0+1), 1)]
        v4 = [(j0*p+(k0+1), 1), (j0*p+k0, -1)]
        w4 = [((k0+1)*n+i0, 1)]
        terms.append((u4, v4, w4))
        
        # M5 = (A[i0][j0] + A[i0][j0+1]) * B[j0+1][k0+1]
        u5 = [(i0*m+j0, 1), (i0*m+(j0+1), 1)]
        v5 = [((j0+1)*p+(k0+1), 1)]
        w5 = [((k0+1)*n+i0, 1)]
        terms.append((u5, v5, w5))
        
        # M6 = (A[i0+1][j0+1] - A[i0+1][j0]) * (B[j0][k0] + B[j0][k0+1])
        u6 = [((i0+1)*m+(j0+1), 1), ((i0+1)*m+j0, -1)]
        v6 = [(j0*p+k0, 1), (j0*p+(k0+1), 1)]
        w6 = [((k0+1)*n+(i0+1), 1)]
        terms.append((u6, v6, w6))
        
        # M7 = (A[i0][j0+1] - A[i0+1][j0+1]) * (B[j0+1][k0+1] + B[j0][k0+1])
        u7 = [(i0*m+(j0+1), 1), ((i0+1)*m+(j0+1), -1)]
        v7 = [((j0+1)*p+(k0+1), 1), (j0*p+(k0+1), 1)]
        w7 = [((k0+1)*n+i0, 1)]
        terms.append((u7, v7, w7))
    
    # Now let's plan the decomposition:
    # 
    # C is 3x5. A is 3x4, B is 4x5.
    #
    # Let's split A into rows: A_top (rows 0,1) and A_bot (row 2)
    # Let's split B into cols: B_left (cols 0,1), B_mid (cols 2,3), B_right (col 4)
    # Let's split B into rows: B_top (rows 0,1), B_bot (rows 2,3)
    #
    # C_top_left (2x2): A_top * B_left
    #   = A_top_cols0,1 * B_left_rows0,1 + A_top_cols2,3 * B_left_rows2,3
    #   Two 2x2x2 Strassen blocks.
    #
    # C_top_mid (2x2): A_top * B_mid
    #   = A_top_cols0,1 * B_mid_rows0,1 + A_top_cols2,3 * B_mid_rows2,3
    #   Two 2x2x2 Strassen blocks.
    #
    # C_top_right (2x1): A_top * B_right
    #   Schoolbook: 2*4 = 8 terms.
    #
    # C_bot_left (1x2): A_bot * B_left
    #   Schoolbook: 4*2 = 8 terms.
    #
    # C_bot_mid (1x2): A_bot * B_mid
    #   Schoolbook: 4*2 = 8 terms.
    #
    # C_bot_right (1x1): A_bot * B_right
    #   Schoolbook: 4*1 = 4 terms.
    #
    # Total: 14 + 14 + 8 + 8 + 8 + 4 = 56 terms.
    
    # C_top_left: i in {0,1}, k in {0,1}
    # Block 1: i0=0, j0=0, k0=0 (A cols 0,1; B rows 0,1; C cols 0,1; C rows 0,1)
    add_strassen_block(0, 0, 0)
    # Block 2: i0=0, j0=2, k0=0 (A cols 2,3; B rows 2,3; C cols 0,1; C rows 0,1)
    add_strassen_block(0, 2, 0)
    
    # C_top_mid: i in {0,1}, k in {2,3}
    # Block 3: i0=0, j0=0, k0=2 (A cols 0,1; B rows 0,1; C cols 2,3; C rows 0,1)
    add_strassen_block(0, 0, 2)
    # Block 4: i0=0, j0=2, k0=2 (A cols 2,3; B rows 2,3; C cols 2,3; C rows 0,1)
    add_strassen_block(0, 2, 2)
    
    # C_top_right: i in {0,1}, k=4
    for i in [0, 1]:
        for j in range(m):
            add_schoolbook_term(i, j, 4)
    
    # C_bot_left: i=2, k in {0,1}
    for j in range(m):
        for k in [0, 1]:
            add_schoolbook_term(2, j, k)
    
    # C_bot_mid: i=2, k in {2,3}
    for j in range(m):
        for k in [2, 3]:
            add_schoolbook_term(2, j, k)
    
    # C_bot_right: i=2, k=4
    for j in range(m):
        add_schoolbook_term(2, j, 4)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, coeff in u_entries:
            U[idx][col] = coeff
        for idx, coeff in v_entries:
            V[idx][col] = coeff
        for idx, coeff in w_entries:
            W[idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
