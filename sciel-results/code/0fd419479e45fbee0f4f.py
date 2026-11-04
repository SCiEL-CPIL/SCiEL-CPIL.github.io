import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_decomp():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    U shape: (4, 7) for A (2x2)
    V shape: (4, 7) for B (2x2)
    W shape: (4, 7) for C (2x2)
    
    Indices for 2x2 matrix (row-major):
    0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def _schoolbook_2x1x2_decomp():
    """
    Returns U, V, W for 2x1x2 matrix multiplication (schoolbook, rank 4).
    U shape: (2, 4) for A (2x1)
    V shape: (2, 4) for B (1x2)
    W shape: (4, 4) for C (2x2)
    
    Indices for A (2x1): 0: (0,0), 1: (1,0)
    Indices for B (1x2): 0: (0,0), 1: (0,1)
    Indices for C (2x2): 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(4)]
    
    # c00 = a00 * b00
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # c01 = a00 * b01
    U[0][1] = 1
    V[1][1] = 1
    W[1][1] = 1
    
    # c10 = a10 * b00
    U[1][2] = 1
    V[0][2] = 1
    W[2][2] = 1
    
    # c11 = a10 * b01
    U[1][3] = 1
    V[1][3] = 1
    W[3][3] = 1
    
    return U, V, W

def _construct_2x3x2_block(U_str, V_str, W_str, U_sb, V_sb, W_sb):
    """
    Construct U, V, W for 2x3x2 multiplication using Strassen + Schoolbook.
    A is 2x3: [A_str (2x2), A_sb (2x1)]
    B is 3x2: [B_str (2x2); B_sb (1x2)]
    C is 2x2: C_str + C_sb
    
    U shape: (6, 11)
    V shape: (6, 11)
    W shape: (4, 11)
    
    A indices:
    0: (0,0), 1: (0,1), 2: (0,2)
    3: (1,0), 4: (1,1), 5: (1,2)
    
    B indices:
    0: (0,0), 1: (0,1)
    2: (1,0), 3: (1,1)
    4: (2,0), 5: (2,1)
    
    C indices:
    0: (0,0), 1: (0,1)
    2: (1,0), 3: (1,1)
    """
    U = [[0]*11 for _ in range(6)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(4)]
    
    # Strassen part (cols 0-6)
    # A_str uses A indices 0,1,3,4
    # B_str uses B indices 0,1,2,3
    # C_str uses C indices 0,1,2,3
    
    # Map Strassen U (4x7) to U (6x11)
    # Strassen U row 0 -> A row 0, col 0 (idx 0)
    # Strassen U row 1 -> A row 0, col 1 (idx 1)
    # Strassen U row 2 -> A row 1, col 0 (idx 3)
    # Strassen U row 3 -> A row 1, col 1 (idx 4)
    
    a_map = [0, 1, 3, 4]
    b_map = [0, 1, 2, 3]
    c_map = [0, 1, 2, 3]
    
    for r in range(7):
        for i in range(4):
            U[a_map[i]][r] += U_str[i][r]
            V[b_map[i]][r] += V_str[i][r]
            W[c_map[i]][r] += W_str[i][r]
            
    # Schoolbook part (cols 7-10)
    # A_sb uses A indices 2, 5 (col 2 of A)
    # B_sb uses B indices 4, 5 (row 2 of B)
    # C_sb uses C indices 0,1,2,3
    
    # Map Schoolbook U (2x4) to U (6x11)
    # Schoolbook U row 0 -> A row 0, col 2 (idx 2)
    # Schoolbook U row 1 -> A row 1, col 2 (idx 5)
    
    # Map Schoolbook V (2x4) to V (6x11)
    # Schoolbook V row 0 -> B row 2, col 0 (idx 4)
    # Schoolbook V row 1 -> B row 2, col 1 (idx 5)
    
    a_sb_map = [2, 5]
    b_sb_map = [4, 5]
    c_sb_map = [0, 1, 2, 3]
    
    for r in range(4):
        for i in range(2):
            U[a_sb_map[i]][7+r] += U_sb[i][r]
            V[b_sb_map[i]][7+r] += V_sb[i][r]
            
        for i in range(4):
            W[c_sb_map[i]][7+r] += W_sb[i][r]
            
    return U, V, W

def _construct_2x3x6_decomp():
    """
    Construct full 2x3x6 decomposition.
    Rank 33.
    """
    U_str, V_str, W_str = _strassen_2x2_decomp()
    U_sb, V_sb, W_sb = _schoolbook_2x1x2_decomp()
    
    U_block, V_block, W_block = _construct_2x3x2_block(U_str, V_str, W_str, U_sb, V_sb, W_sb)
    
    # Full U: (6, 33)
    # Full V: (18, 33)
    # Full W: (12, 33)
    
    U = [[0]*33 for _ in range(6)]
    V = [[0]*33 for _ in range(18)]
    W = [[0]*33 for _ in range(12)]
    
    # B is 3x6. Split into 3 blocks of 3x2.
    # Block 0: cols 0,1 of B. Rows 0,1,2.
    # Block 1: cols 2,3 of B. Rows 0,1,2.
    # Block 2: cols 4,5 of B. Rows 0,1,2.
    
    # C is 2x6.
    # C block 0: cols 0,1.
    # C block 1: cols 2,3.
    # C block 2: cols 4,5.
    
    for block in range(3):
        offset = block * 11
        
        # U is the same for all blocks (A is shared)
        for i in range(6):
            for r in range(11):
                U[i][offset+r] += U_block[i][r]
                
        # V is different for each block.
        # V_block is (6, 11).
        # V_full is (18, 33).
        # Block b uses rows 0-5 of V_block, placed in rows 6*b to 6*b+5 of V_full?
        # Wait. B is 3x6.
        # Row j of B corresponds to V index j*6 + k.
        # In V_block, index j corresponds to row j of B_block (which is 3x2).
        # So for block b, row j of B_block corresponds to row j of B_full.
        # But B_full has 3 rows.
        # V_full index is j*6 + k.
        # For block b, the columns of B are 2*b and 2*b+1.
        # So for row j (0,1,2) and col k (0,1) in the block,
        # the index in V_full is j*6 + (2*b + k).
        
        for j in range(3):
            for k in range(2):
                v_idx = j * 6 + (2 * b + k) # b is block index
                # In V_block, the index is j*2 + k
                v_block_idx = j * 2 + k
                for r in range(11):
                    V[v_idx][offset+r] += V_block[v_block_idx][r]
                    
        # W is different for each block.
        # W_block is (4, 11).
        # W_full is (12, 33).
        # C is 2x6.
        # W_full index is k*2 + i.
        # For block b, the columns of C are 2*b and 2*b+1.
        # So for row i (0,1) and col k (0,1) in the block,
        # the index in W_full is (2*b + k)*2 + i.
        
        for i in range(2):
            for k in range(2):
                w_idx = (2 * b + k) * 2 + i
                # In W_block, the index is k*2 + i
                w_block_idx = k * 2 + i
                for r in range(11):
                    W[w_idx][offset+r] += W_block[w_block_idx][r]
                    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try Strassen-based construction (rank 33)
    res = _construct_2x3x6_decomp()
    
    if _verify_decomposition(res['U'], res['V'], res['W'], n, m, p, tol=1e-9):
        return res
        
    # Fallback to Schoolbook
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify_decomposition(res['U'], res['V'], res['W'], 2, 3, 6))
