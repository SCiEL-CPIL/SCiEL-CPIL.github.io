"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 9]
DEADLINE = 145.0  # Safety margin before 150s kill

def _build_strassen_block():
    """
    Returns (U_block, V_block, W_block) for a 2x2 * 2x2 -> 2x2 Strassen block.
    U shape: (4, 7)  - A indices: 00, 01, 10, 11
    V shape: (4, 7)  - B indices: 00, 01, 10, 11
    W shape: (4, 7)  - C indices: 00, 01, 10, 11 (k*n+i, n=2)
    
    Strassen multiplications:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A10)*B11
    M6 = (A11-A10)*(B00+B10)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00 += M1
    W[3][0] = 1  # C11 += M1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10 += M2
    W[3][1] = -1 # C11 -= M2
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01 += M3
    W[3][2] = 1  # C11 += M3
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00 += M4
    W[2][3] = 1  # C10 += M4
    
    # M5 = (A00+A10)*B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00 -= M5
    W[1][4] = 1  # C01 += M5
    
    # M6 = (A11-A10)*(B00+B10)
    U[3][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1  # C11 += M6
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1 # C00 -= M7
    
    return U, V, W

def _build_single_col_block():
    """
    Returns (U_block, V_block, W_block) for a 2x2 * 2x1 -> 2x1 schoolbook block.
    U shape: (4, 4)
    V shape: (2, 4)
    W shape: (2, 4)
    
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # A00*B0 -> C0
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # A01*B1 -> C0
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # A10*B0 -> C1
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # A11*B1 -> C1
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # 9 output columns: group as 4 pairs + 1 single
    # Pairs: (0,1), (2,3), (4,5), (6,7)
    # Single: 8
    
    # Each Strassen block contributes rank 7
    # Each single block contributes rank 4
    # Total rank = 4*7 + 1*4 = 28 + 4 = 32
    
    total_rank = 4 * 7 + 4
    U = [[0] * total_rank for _ in range(n * m)]  # (4, 32)
    V = [[0] * total_rank for _ in range(m * p)]  # (18, 32)
    W = [[0] * total_rank for _ in range(p * n)]  # (18, 32)
    
    # Precompute blocks
    strassen_U, strassen_V, strassen_W = _build_strassen_block()
    single_U, single_V, single_W = _build_single_col_block()
    
    col_offset = 0
    
    # 4 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        # Column pair: cols 2*block_idx and 2*block_idx+1
        # V indices for these two columns:
        # col c: B indices are j*p + c, so for j in 0..1: c, c+p
        # For block starting at col 2*block_idx:
        # V rows: 2*block_idx, 2*block_idx+1, 2*block_idx+p, 2*block_idx+1+p
        
        # W indices: k*n + i, for k in 0..8
        # The two output columns are 2*block_idx and 2*block_idx+1
        # So W rows: (2*block_idx)*n + i and (2*block_idx+1)*n + i
        # i in 0..1
        
        # U is shared across all blocks (same A)
        
        for r in range(7):
            for a_idx in range(4):
                U[a_idx][col_offset + r] = strassen_U[a_idx][r]
            
            # V: map local V indices (0,1,2,3) to global
            # local 0 -> B[0, 2*block_idx] -> global row 2*block_idx
            # local 1 -> B[0, 2*block_idx+1] -> global row 2*block_idx+1
            # local 2 -> B[1, 2*block_idx] -> global row p + 2*block_idx
            # local 3 -> B[1, 2*block_idx+1] -> global row p + 2*block_idx+1
            v_map = [2*block_idx, 2*block_idx+1, p+2*block_idx, p+2*block_idx+1]
            for v_local in range(4):
                V[v_map[v_local]][col_offset + r] = strassen_V[v_local][r]
            
            # W: map local W indices (0,1,2,3) to global
            # local 0 -> C[0, 2*block_idx] -> global row (2*block_idx)*n + 0
            # local 1 -> C[0, 2*block_idx+1] -> global row (2*block_idx+1)*n + 0
            # local 2 -> C[1, 2*block_idx] -> global row (2*block_idx)*n + 1
            # local 3 -> C[1, 2*block_idx+1] -> global row (2*block_idx+1)*n + 1
            w_map = [(2*block_idx)*n, (2*block_idx+1)*n, (2*block_idx)*n + 1, (2*block_idx+1)*n + 1]
            for w_local in range(4):
                W[w_map[w_local]][col_offset + r] = strassen_W[w_local][r]
        
        col_offset += 7
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    # 1 single block for column 8
    for r in range(4):
        for a_idx in range(4):
            U[a_idx][col_offset + r] = single_U[a_idx][r]
        
        # V: local 0 -> B[0, 8] -> global row 8
        #     local 1 -> B[1, 8] -> global row p+8 = 17
        v_map = [8, p+8]
        for v_local in range(2):
            V[v_map[v_local]][col_offset + r] = single_V[v_local][r]
        
        # W: local 0 -> C[0, 8] -> global row 8*n = 16
        #     local 1 -> C[1, 8] -> global row 8*n+1 = 17
        w_map = [8*n, 8*n+1]
        for w_local in range(2):
            W[w_map[w_local]][col_offset + r] = single_W[w_local][r]
    
    col_offset += 4
    
    return {"U": U, "V": V, "W": W}
