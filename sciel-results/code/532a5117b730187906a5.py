"""Matrix multiplication 2x4x7 with Strassen-based decomposition."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-50 tensor decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy: 
    A (2x4) = [A_left A_right] where each is 2x2
    B (4x7) = [B_top; B_bottom] where each is 2x7
    C = A_left * B_top + A_right * B_bottom
    
    Each 2x2 * 2x7 is split into 2x2*2x3 + 2x2*2x4
    2x2*2x3 = 2x2*2x2 + 2x2*2x1 (rank 7+4=11)
    2x2*2x4 = 2x2*2x2 + 2x2*2x2 (rank 7+7=14)
    Total per 2x2*2x7: 25
    Total: 2 * 25 = 50
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 2, 4, 7
    total_rank = 50
    
    U = [[0] * total_rank for _ in range(8)]   # 2*4 = 8
    V = [[0] * total_rank for _ in range(28)]  # 4*7 = 28
    W = [[0] * total_rank for _ in range(14)]  # 7*2 = 14
    
    col = 0
    
    def add_222_block(i_off, j_off, k_off):
        """Add a 2x2x2 Strassen block (rank 7)."""
        nonlocal col
        if time.time() > deadline:
            return None
        cols = []
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        c = col
        U[i_off*4 + j_off + 0][c] = 1
        U[(i_off+1)*4 + j_off + 1][c] = 1
        V[(j_off+0)*7 + k_off + 0][c] = 1
        V[(j_off+1)*7 + k_off + 1][c] = 1
        cols.append(c)
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        c = col
        U[(i_off+1)*4 + j_off + 0][c] = 1
        U[(i_off+1)*4 + j_off + 1][c] = 1
        V[(j_off+0)*7 + k_off + 0][c] = 1
        cols.append(c)
        col += 1
        
        # M3 = A[0,0] * (B[1,0]-B[0,0])
        c = col
        U[i_off*4 + j_off + 0][c] = 1
        V[(j_off+1)*7 + k_off + 0][c] = 1
        V[(j_off+0)*7 + k_off + 0][c] = -1
        cols.append(c)
        col += 1
        
        # M4 = A[1,1] * (B[0,1]-B[1,1])
        c = col
        U[(i_off+1)*4 + j_off + 1][c] = 1
        V[(j_off+0)*7 + k_off + 1][c] = 1
        V[(j_off+1)*7 + k_off + 1][c] = -1
        cols.append(c)
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        c = col
        U[i_off*4 + j_off + 0][c] = 1
        U[i_off*4 + j_off + 1][c] = 1
        V[(j_off+1)*7 + k_off + 1][c] = 1
        cols.append(c)
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        c = col
        U[(i_off+1)*4 + j_off + 0][c] = 1
        U[i_off*4 + j_off + 0][c] = -1
        V[(j_off+0)*7 + k_off + 0][c] = 1
        V[(j_off+0)*7 + k_off + 1][c] = 1
        cols.append(c)
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        c = col
        U[i_off*4 + j_off + 1][c] = 1
        U[(i_off+1)*4 + j_off + 1][c] = -1
        V[(j_off+1)*7 + k_off + 0][c] = 1
        V[(j_off+1)*7 + k_off + 1][c] = 1
        cols.append(c)
        col += 1
        
        # Set W values
        W[(k_off+0)*2 + i_off + 0][cols[0]] = 1   # M1 -> C[0,0]
        W[(k_off+0)*2 + i_off + 0][cols[3]] = 1   # M4 -> C[0,0]
        W[(k_off+0)*2 + i_off + 0][cols[4]] = -1  # -M5 -> C[0,0]
        W[(k_off+0)*2 + i_off + 0][cols[6]] = 1   # M7 -> C[0,0]
        
        W[(k_off+1)*2 + i_off + 0][cols[2]] = 1   # M3 -> C[0,1]
        W[(k_off+1)*2 + i_off + 0][cols[4]] = 1   # M5 -> C[0,1]
        
        W[(k_off+0)*2 + i_off + 1][cols[1]] = 1   # M2 -> C[1,0]
        W[(k_off+0)*2 + i_off + 1][cols[3]] = 1   # M4 -> C[1,0]
        
        W[(k_off+1)*2 + i_off + 1][cols[0]] = 1   # M1 -> C[1,1]
        W[(k_off+1)*2 + i_off + 1][cols[1]] = -1  # -M2 -> C[1,1]
        W[(k_off+1)*2 + i_off + 1][cols[2]] = 1   # M3 -> C[1,1]
        W[(k_off+1)*2 + i_off + 1][cols[5]] = 1   # M6 -> C[1,1]
        
        return cols
    
    def add_221_block(i_off, j_off, k_off):
        """Add a 2x2x1 block (rank 4)."""
        nonlocal col
        if time.time() > deadline:
            return None
        cols = []
        
        # P1 = A[0,0] * B[0,0]
        c = col
        U[i_off*4 + j_off + 0][c] = 1
        V[(j_off+0)*7 + k_off][c] = 1
        cols.append(c)
        col += 1
        
        # P2 = A[0,1] * B[1,0]
        c = col
        U[i_off*4 + j_off + 1][c] = 1
        V[(j_off+1)*7 + k_off][c] = 1
        cols.append(c)
        col += 1
        
        # P3 = A[1,0] * B[0,0]
        c = col
        U[(i_off+1)*4 + j_off + 0][c] = 1
        V[(j_off+0)*7 + k_off][c] = 1
        cols.append(c)
        col += 1
        
        # P4 = A[1,1] * B[1,0]
        c = col
        U[(i_off+1)*4 + j_off + 1][c] = 1
        V[(j_off+1)*7 + k_off][c] = 1
        cols.append(c)
        col += 1
        
        # C[0] = P1 + P2
        W[k_off*2 + i_off + 0][cols[0]] = 1
        W[k_off*2 + i_off + 0][cols[1]] = 1
        
        # C[1] = P3 + P4
        W[k_off*2 + i_off + 1][cols[2]] = 1
        W[k_off*2 + i_off + 1][cols[3]] = 1
        
        return cols
    
    # Build the decomposition
    # A_left = A[0:2, 0:2], A_right = A[0:2, 2:4]
    # B_top = B[0:2, 0:7], B_bottom = B[2:4, 0:7]
    
    # A_left * B_top[0:2, 0:3] (2x2 * 2x3, rank 11)
    add_222_block(0, 0, 0)  # 2x2*2x2, cols 0-1
    add_221_block(0, 0, 2)  # 2x2*2x1, col 2
    
    # A_left * B_top[0:2, 3:7] (2x2 * 2x4, rank 14)
    add_222_block(0, 0, 3)  # 2x2*2x2, cols 3-4
    add_222_block(0, 0, 5)  # 2x2*2x2, cols 5-6
    
    # A_right * B_bottom[2:4, 0:3] (2x2 * 2x3, rank 11)
    add_222_block(0, 2, 0)  # 2x2*2x2, cols 0-1
    add_221_block(0, 2, 2)  # 2x2*2x1, col 2
    
    # A_right * B_bottom[2:4, 3:7] (2x2 * 2x4, rank 14)
    add_222_block(0, 2, 3)  # 2x2*2x2, cols 3-4
    add_222_block(0, 2, 5)  # 2x2*2x2, cols 5-6
    
    if col != 50:
        # Fallback to schoolbook if something went wrong
        U_fb = [[0] * 56 for _ in range(8)]
        V_fb = [[0] * 56 for _ in range(28)]
        W_fb = [[0] * 56 for _ in range(14)]
        c = 0
        for i in range(2):
            for j in range(4):
                for k in range(7):
                    U_fb[i*4+j][c] = 1
                    V[j*7+k][c] = 1
                    W[k*2+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

