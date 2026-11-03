"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    if time.time() > _START_TIME + _DEADLINE - _MARGIN:
        # Fallback: return schoolbook rank 96
        rank = 96
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
    # Rank 88 decomposition:
    # 32 from schoolbook for 1x4x8
    # 56 from Strassen for 2x4x8 (4 blocks of 2x2x2 Strassen x 2 halves)
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)] # 12 x 88
    V = [[0] * rank for _ in range(m * p)] # 32 x 88
    W = [[0] * rank for _ in range(p * n)] # 24 x 88
    
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    # A_1 is row 0 of A.
    for j in range(m): # 4
        for k in range(p): # 8
            # U index: i=0, j
            U[0 * m + j][col] = 1
            # V index: j, k
            V[j * p + k][col] = 1
            # W index: k, i=0
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    # Split B into 4 blocks of width 2 (columns 0-1, 2-3, 4-5, 6-7)
    # For each block j_block:
    #   Product C_block (2x2) = A_2 * B_block
    #   A_2 is 2x4. Split into A_2a (cols 0-1) and A_2b (cols 2-3).
    #   B_block is 4x2. Split into B_ba (rows 0-1) and B_bb (rows 2-3).
    #   C_block = A_2a * B_ba + A_2b * B_bb
    #   Each term is 2x2x2, use Strassen (rank 7).
    
    # Strassen decomposition helper data
    # Local indices 0,1 for 2x2 matrices
    # U_local, V_local, W_local coefficients for 7 terms
    
    # Term 1: (X00+X11)(Y00+Y11)
    # U: X00:1, X11:1
    # V: Y00:1, Y11:1
    # W: Z00:1, Z11:1
    strassen_terms = [
        { # M1
            'U': {0: 1, 3: 1}, # X00, X11 (indices 0,1,2,3 for 00,01,10,11)
            'V': {0: 1, 3: 1}, # Y00, Y11
            'W': {0: 1, 3: 1}  # Z00, Z11
        },
        { # M2
            'U': {2: 1, 3: 1}, # X10, X11
            'V': {0: 1},       # Y00
            'W': {2: 1, 3: -1} # Z10, Z11
        },
        { # M3
            'U': {0: 1},       # X00
            'V': {1: 1, 3: -1},# Y01, Y11
            'W': {1: 1, 3: 1}  # Z01, Z11
        },
        { # M4
            'U': {3: 1},       # X11
            'V': {2: 1, 0: -1},# Y10, Y00
            'W': {0: 1, 2: 1}  # Z00, Z10
        },
        { # M5
            'U': {0: 1, 1: 1}, # X00, X01
            'V': {3: 1},       # Y11
            'W': {0: -1, 1: 1} # Z00, Z01
        },
        { # M6
            'U': {2: 1, 0: -1},# X10, X00
            'V': {0: 1, 1: 1}, # Y00, Y01
            'W': {3: 1}        # Z11
        },
        { # M7
            'U': {1: 1, 3: -1},# X01, X11
            'V': {2: 1, 3: 1}, # Y10, Y11
            'W': {0: 1}        # Z00
        }
    ]
    
    for j_block in range(4):
        # Handle P1 = A_2a * B_ba
        # A_2a: global i in {1,2}, j in {0,1}
        # B_ba: global j in {0,1}, k in {2*j_block, 2*j_block+1}
        # Output: global i in {1,2}, k in {2*j_block, 2*j_block+1}
        
        for term in strassen_terms:
            # Fill U
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2) # row, col in 2x2
                global_i = 1 + lu
                global_j = lv
                U[global_i * m + global_j][col] += val
            
            # Fill V
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                global_j = lu
                global_k = 2 * j_block + lv
                V[global_j * p + global_k][col] += val
            
            # Fill W
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                global_i = 1 + lu
                global_k = 2 * j_block + lv
                W[global_k * n + global_i][col] += val
                
            col += 1
            
        # Handle P2 = A_2b * B_bb
        # A_2b: global i in {1,2}, j in {2,3}
        # B_bb: global j in {2,3}, k in {2*j_block, 2*j_block+1}
        # Output: global i in {1,2}, k in {2*j_block, 2*j_block+1}
        
        for term in strassen_terms:
            # Fill U
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                global_i = 1 + lu
                global_j = 2 + lv
                U[global_i * m + global_j][col] += val
            
            # Fill V
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                global_j = 2 + lu
                global_k = 2 * j_block + lv
                V[global_j * p + global_k][col] += val
            
            # Fill W
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                global_i = 1 + lu
                global_k = 2 * j_block + lv
                W[global_k * n + global_i][col] += val
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
