"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build the U, V, W matrices for the rank 44 decomposition.
    # Rank 44 = 4 * 7 (Strassen for 2x2x2 blocks) + 16 (Naive for 1x4x4)
    
    # We use lists of lists for U, V, W
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for a 2x2x2 block
    # i_offset, j_offset, k_offset
    # A indices: i in {i_offset, i_offset+1}, j in {j_offset, j_offset+1}
    # B indices: j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
    # C indices: i in {i_offset, i_offset+1}, k in {k_offset, k_offset+1}
    # W index: k*n + i
    
    def add_strassen_block(i_off, j_off, k_off, start_col):
        # Strassen factors for 2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to map these to U, V, W contributions
        # U[i*m+j, col], V[j*p+k, col], W[k*n+i, col]
        
        # Local indices for A: (0,0)->(i_off, j_off), (0,1)->(i_off, j_off+1), (1,0)->(i_off+1, j_off), (1,1)->(i_off+1, j_off+1)
        # Local indices for B: (0,0)->(j_off, k_off), (0,1)->(j_off, k_off+1), (1,0)->(j_off+1, k_off), (1,1)->(j_off+1, k_off+1)
        # Local indices for W: (0,0)->(k_off, i_off), (0,1)->(k_off+1, i_off), (1,0)->(k_off, i_off+1), (1,1)->(k_off+1, i_off+1)
        
        # M1
        c = start_col
        # U: A00+A11
        U[i_off*m + j_off][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = 1
        # V: B00+B11
        V[j_off*p + k_off][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        # W: C00+C01+C10+C11
        W[k_off*n + i_off][c] = 1
        W[(k_off+1)*n + i_off][c] = 1
        W[k_off*n + (i_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        col = c + 1
        
        # M2
        c = start_col + 1
        # U: A10+A11
        U[(i_off+1)*m + j_off][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = 1
        # V: B00
        V[j_off*p + k_off][c] = 1
        # W: -C10 + C11
        W[k_off*n + (i_off+1)][c] = -1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        col = c + 1
        
        # M3
        c = start_col + 2
        # U: A00
        U[i_off*m + j_off][c] = 1
        # V: B01-B11
        V[j_off*p + (k_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = -1
        # W: C00 - C01
        W[k_off*n + i_off][c] = 1
        W[(k_off+1)*n + i_off][c] = -1
        col = c + 1
        
        # M4
        c = start_col + 3
        # U: A11
        U[(i_off+1)*m + (j_off+1)][c] = 1
        # V: B10-B00
        V[(j_off+1)*p + k_off][c] = 1
        V[j_off*p + k_off][c] = -1
        # W: -C00 + C10
        W[k_off*n + i_off][c] = -1
        W[k_off*n + (i_off+1)][c] = 1
        col = c + 1
        
        # M5
        c = start_col + 4
        # U: A00+A01
        U[i_off*m + j_off][c] = 1
        U[i_off*m + (j_off+1)][c] = 1
        # V: B11
        V[(j_off+1)*p + (k_off+1)][c] = 1
        # W: C00 + C10
        W[k_off*n + i_off][c] = 1
        W[k_off*n + (i_off+1)][c] = 1
        col = c + 1
        
        # M6
        c = start_col + 5
        # U: A10-A00
        U[(i_off+1)*m + j_off][c] = 1
        U[i_off*m + j_off][c] = -1
        # V: B00+B01
        V[j_off*p + k_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1
        # W: C01 - C11
        W[(k_off+1)*n + i_off][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = -1
        col = c + 1
        
        # M7
        c = start_col + 6
        # U: A01-A11
        U[i_off*m + (j_off+1)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = -1
        # V: B10+B11
        V[(j_off+1)*p + k_off][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        # W: -C01 + C11
        W[(k_off+1)*n + i_off][c] = -1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        col = c + 1
        
        return col

    # Part 1: i in {0, 1} (top 2 rows)
    # Split j into {0,1} and {2,3}
    # Split k into {0,1} and {2,3}
    
    # Block 1: i_off=0, j_off=0, k_off=0
    col = add_strassen_block(0, 0, 0, col)
    
    # Block 2: i_off=0, j_off=2, k_off=0
    col = add_strassen_block(0, 2, 0, col)
    
    # Block 3: i_off=0, j_off=0, k_off=2
    col = add_strassen_block(0, 0, 2, col)
    
    # Block 4: i_off=0, j_off=2, k_off=2
    col = add_strassen_block(0, 2, 2, col)
    
    # Part 2: i = 2 (bottom row)
    # Naive multiplication
    # C[2, k] = sum_j A[2, j] B[j, k]
    # For each j, k, we have a term.
    # U[2*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + 2, col] = 1
    
    for j in range(m):
        for k in range(p):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    # col should be 44
    if col != rank:
        # Fallback to schoolbook if something went wrong
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
                    if column == rank:
                        break
                if column == rank:
                    break
            if column == rank:
                break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
