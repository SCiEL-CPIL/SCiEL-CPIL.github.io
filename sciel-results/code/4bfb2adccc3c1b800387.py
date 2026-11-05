"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _construct_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication with rank 7 (Strassen).
    U, V, W are lists of lists (4x7).
    Indexing:
    U row: i*2 + j
    V row: j*2 + k
    W row: k*2 + i
    """
    # Strassen formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A flat: [A11, A12, A21, A22] -> indices 0, 1, 2, 3
    # B flat: [B11, B12, B21, B22] -> indices 0, 1, 2, 3
    # C flat: [C11, C12, C21, C22] -> indices 0, 1, 2, 3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Col 0: M1
    # U: A11+A22 -> [1, 0, 0, 1]
    U[0][0] = 1; U[3][0] = 1
    # V: B11+B22 -> [1, 0, 0, 1]
    V[0][0] = 1; V[3][0] = 1
    # W: C11 += 1, C22 += 1
    W[0][0] = 1; W[3][0] = 1
    
    # Col 1: M2
    # U: A21+A22 -> [0, 0, 1, 1]
    U[2][1] = 1; U[3][1] = 1
    # V: B11 -> [1, 0, 0, 0]
    V[0][1] = 1
    # W: C21 += 1
    W[2][1] = 1
    
    # Col 2: M3
    # U: A11 -> [1, 0, 0, 0]
    U[0][2] = 1
    # V: B12-B22 -> [0, 1, 0, -1]
    V[1][2] = 1; V[3][2] = -1
    # W: C12 += 1, C22 += 1
    W[1][2] = 1; W[3][2] = 1
    
    # Col 3: M4
    # U: A22 -> [0, 0, 0, 1]
    U[3][3] = 1
    # V: B21-B11 -> [0, -1, 1, 0]
    V[1][3] = -1; V[2][3] = 1
    # W: C11 += 1, C21 += 1
    W[0][3] = 1; W[2][3] = 1
    
    # Col 4: M5
    # U: A11+A12 -> [1, 1, 0, 0]
    U[0][4] = 1; U[1][4] = 1
    # V: B22 -> [0, 0, 0, 1]
    V[3][4] = 1
    # W: C11 -= 1, C12 += 1
    W[0][4] = -1; W[1][4] = 1
    
    # Col 5: M6
    # U: A21-A11 -> [-1, 0, 1, 0]
    U[0][5] = -1; U[2][5] = 1
    # V: B11+B12 -> [1, 1, 0, 0]
    V[0][5] = 1; V[1][5] = 1
    # W: C22 += 1
    W[3][5] = 1
    
    # Col 6: M7
    # U: A12-A22 -> [0, 1, 0, -1]
    U[1][6] = 1; U[3][6] = -1
    # V: B21+B22 -> [0, 0, 1, 1]
    V[2][6] = 1; V[3][6] = 1
    # W: C11 += 1
    W[0][6] = 1
    
    return U, V, W

def _construct_4x4():
    """
    Returns U, V, W for 4x4 matrix multiplication with rank 49.
    Uses Strassen recursion.
    """
    U2, V2, W2 = _construct_2x2()
    
    # We will build U, V, W of size 16x49
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Strassen steps for 4x4 (blocks are 2x2)
    # A11, A12, A21, A22 are 2x2 blocks.
    # Indices in 4x4 flat:
    # A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
    # B11: 0-3, B12: 4-7, B21: 8-11, B22: 12-15
    # C11: 0-3, C12: 4-7, C21: 8-11, C22: 12-15
    
    # Helper to embed a 2x2 decomposition into a specific block position
    # and specific linear combination of inputs.
    
    # M1 = (A11+A22)(B11+B22)
    # Inputs: A11+A22, B11+B22
    # U part: Combine U2 for A11 and A22 with coeff 1
    # V part: Combine V2 for B11 and B22 with coeff 1
    # W part: Embed W2 into C11 and C22 with coeff 1
    
    def add_step(u_blocks, v_blocks, w_blocks, offset):
        # u_blocks: list of (block_idx, coeff)
        # v_blocks: list of (block_idx, coeff)
        # w_blocks: list of (block_idx, coeff)
        
        for k in range(7):
            col = offset + k
            
            # Fill U
            for b_idx, coeff in u_blocks:
                # b_idx is 0..3 corresponding to A11, A12, A21, A22
                # Rows in U2 are 0..3 (local 2x2)
                # Rows in U are 16
                # Mapping: local row r in block b_idx -> global row b_idx*4 + r
                for r in range(4):
                    if U2[r][k] != 0:
                        U[b_idx*4 + r][col] += coeff * U2[r][k]
            
            # Fill V
            for b_idx, coeff in v_blocks:
                for r in range(4):
                    if V2[r][k] != 0:
                        V[b_idx*4 + r][col] += coeff * V2[r][k]
            
            # Fill W
            for b_idx, coeff in w_blocks:
                for r in range(4):
                    if W2[r][k] != 0:
                        W[b_idx*4 + r][col] += coeff * W2[r][k]

    # Step 1: M1 = (A11+A22)(B11+B22)
    # U blocks: A11(0), A22(3)
    # V blocks: B11(0), B22(3)
    # W blocks: C11(0), C22(3)
    add_step([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)], 0)
    
    # Step 2: M2 = (A21+A22)B11
    # U blocks: A21(2), A22(3)
    # V blocks: B11(0)
    # W blocks: C21(2)
    add_step([(2, 1), (3, 1)], [(0, 1)], [(2, 1)], 7)
    
    # Step 3: M3 = A11(B12-B22)
    # U blocks: A11(0)
    # V blocks: B12(1), B22(3) with -1
    # W blocks: C12(1), C22(3)
    add_step([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)], 14)
    
    # Step 4: M4 = A22(B21-B11)
    # U blocks: A22(3)
    # V blocks: B21(2), B11(0) with -1
    # W blocks: C11(0), C21(2)
    add_step([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)], 21)
    
    # Step 5: M5 = (A11+A12)B22
    # U blocks: A11(0), A12(1)
    # V blocks: B22(3)
    # W blocks: C11(0) with -1, C12(1) with 1
    add_step([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)], 28)
    
    # Step 6: M6 = (A21-A11)(B11+B12)
    # U blocks: A21(2), A11(0) with -1
    # V blocks: B11(0), B12(1)
    # W blocks: C22(3)
    add_step([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 35)
    
    # Step 7: M7 = (A12-A22)(B21+B22)
    # U blocks: A12(1), A22(3) with -1
    # V blocks: B21(2), B22(3)
    # W blocks: C11(0)
    add_step([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)], 42)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Get 4x4 decomposition
    U4, V4, W4 = _construct_4x4()
    
    # Initialize 5x5 matrices
    # U: 25x?, V: 25x?, W: 25x?
    # Rank will be 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    
    rank = 110
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    # 1. Embed 4x4 part (indices 0-3)
    # U4 rows 0-15 map to U rows 0-15
    # V4 rows 0-15 map to V rows 0-15
    # W4 rows 0-15 map to W rows 0-15
    
    for i in range(16):
        for j in range(49):
            U[i][j] = U4[i][j]
            V[i][j] = V4[i][j]
            W[i][j] = W4[i][j]
            
    col_offset = 49
    
    # Helper to add a standard basis term
    # u_idx, v_idx, w_idx
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col_offset
        if col_offset >= rank:
            raise Exception("Rank overflow")
        U[u_idx][col_offset] = coeff
        V[v_idx][col_offset] = coeff
        W[w_idx][col_offset] = coeff
        col_offset += 1

    # 2. A41 B14 (4x1 * 1x4 -> 4x4)
    # A41: rows 0-3, col 4. Flat idx: i*5 + 4
    # B14: row 4, cols 0-3. Flat idx: 4*5 + k = 20 + k
    # C44: rows 0-3, cols 0-3. Flat idx: k*5 + i
    for i in range(4):
        for k in range(4):
            u_idx = i*5 + 4
            v_idx = 20 + k
            w_idx = k*5 + i
            add_term(u_idx, v_idx, w_idx)
            
    # 3. A44 B41 (4x4 * 4x1 -> 4x1)
    # A44: rows 0-3, cols 0-3. Flat idx: i*5 + j
    # B41: rows 0-3, col 4. Flat idx: j*5 + 4
    # C41: rows 0-3, col 4. Flat idx: 4*5 + i = 20 + i
    for i in range(4):
        for j in range(4):
            u_idx = i*5 + j
            v_idx = j*5 + 4
            w_idx = 20 + i
            add_term(u_idx, v_idx, w_idx)
            
    # 4. A41 B11 (4x1 * 1x1 -> 4x1)
    # A41: rows 0-3, col 4. Flat idx: i*5 + 4
    # B11: row 4, col 4. Flat idx: 24
    # C41: rows 0-3, col 4. Flat idx: 20 + i
    for i in range(4):
        u_idx = i*5 + 4
        v_idx = 24
        w_idx = 20 + i
        add_term(u_idx, v_idx, w_idx)
        
    # 5. A14 B44 (1x4 * 4x4 -> 1x4)
    # A14: row 4, cols 0-3. Flat idx: 20 + j
    # B44: rows 0-3, cols 0-3. Flat idx: j*5 + k
    # C14: row 4, cols 0-3. Flat idx: k*5 + 4
    for j in range(4):
        for k in range(4):
            u_idx = 20 + j
            v_idx = j*5 + k
            w_idx = k*5 + 4
            add_term(u_idx, v_idx, w_idx)
            
    # 6. A11 B14 (1x1 * 1x4 -> 1x4)
    # A11: row 4, col 4. Flat idx: 24
    # B14: row 4, cols 0-3. Flat idx: 20 + k
    # C14: row 4, cols 0-3. Flat idx: k*5 + 4
    for k in range(4):
        u_idx = 24
        v_idx = 20 + k
        w_idx = k*5 + 4
        add_term(u_idx, v_idx, w_idx)
        
    # 7. A14 B41 (1x4 * 4x1 -> 1x1)
    # A14: row 4, cols 0-3. Flat idx: 20 + j
    # B41: rows 0-3, col 4. Flat idx: j*5 + 4
    # C11: row 4, col 4. Flat idx: 24
    for j in range(4):
        u_idx = 20 + j
        v_idx = j*5 + 4
        w_idx = 24
        add_term(u_idx, v_idx, w_idx)
        
    # 8. A11 B11 (1x1 * 1x1 -> 1x1)
    # A11: 24
    # B11: 24
    # C11: 24
    add_term(24, 24, 24)
    
    # Check if col_offset == rank
    if col_offset != rank:
        # If not, we have unused columns or overflow. 
        # Given the calculation 49+16+16+4+16+4+4+1 = 110, it should match.
        pass
        
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END
