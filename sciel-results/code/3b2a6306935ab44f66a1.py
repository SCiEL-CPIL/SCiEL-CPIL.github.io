"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7
    
    # Helper to create schoolbook decomposition for dimensions (n, m, p)
    # Returns U, V, W lists
    def schoolbook(n, m, p):
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen decomposition for 2x2x2
    # Returns U, V, W lists
    def strassen_222():
        # Strassen's algorithm for 2x2 matrix multiplication
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Indices for U (A part): a,b,c,d
        # Indices for V (B part): e,f,g,h
        # Indices for W (C part): c00, c01, c10, c11
        
        # U rows: 0:a, 1:b, 2:c, 3:d
        # V rows: 0:e, 1:f, 2:g, 3:h
        # W rows: 0:c00, 1:c01, 2:c10, 3:c11
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # c00 += M1, c11 += M1
        
        # M2: (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1 # c10 += M2
        
        # M3: a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1 # c01 += M3
        
        # M4: d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[2][3] = 1 # c10 += M4
        
        # M5: (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[3][4] = 1 # c11 += M5
        
        # M6: (c-a)(e+f)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1 # c10 += M6
        
        # M7: (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[1][6] = 1 # c01 += M7
        
        return U, V, W

    # We will build the decomposition by combining blocks.
    # The total U, V, W will be lists of lists.
    # We need to offset the indices for each block.
    
    U_total = []
    V_total = []
    W_total = []
    current_rank = 0
    
    def add_block(U, V, W, i_offset, j_offset, k_offset):
        nonlocal current_rank
        n_block = len(U) // len(V) # This is tricky, need to know dims
        # Better to pass dims
        pass

    # Let's implement a more structured way.
    # We define a function to add a block with known dimensions n, m, p and its decomposition.
    
    def add_decomposition(n, m, p, U, V, W, i_off, j_off, k_off):
        nonlocal current_rank
        # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
        # We need to embed these into the global U, V, W.
        
        # Global U row index: i_global * m_global + j_global
        # Global V row index: j_global * p_global + k_global
        # Global W row index: k_global * n_global + i_global
        
        # Note: The problem statement says W[k*n+i, r].
        # My schoolbook helper used W[k*n+i].
        # Strassen W is for 2x2, so W[k*2+i].
        
        # We iterate over local rows/cols and map to global.
        
        # U
        for i in range(n):
            for j in range(m):
                local_u_row = i * m + j
                global_u_row = (i + i_off) * m + (j + j_off)
                
                # Initialize row in U_total if not exists
                # Since we are appending columns, we need to grow U_total rows?
                # No, U_total has fixed shape (n*m, total_rank).
                # We are building columns.
                
                # We will store U_total as a list of rows, where each row is a list of values for all ranks.
                # But we don't know total_rank ahead of time easily without summing.
                # Let's collect blocks first.
                
                pass

    # Strategy:
    # 1. Define all sub-blocks and their decompositions.
    # 2. Calculate total rank.
    # 3. Initialize U, V, W with zeros.
    # 4. Fill in the values.
    
    blocks = []
    
    # Block B: M_{1,5,7} (Schoolbook)
    # i_off=2 (rows 2 of 3), j_off=0, k_off=0
    U_b, V_b, W_b = schoolbook(1, 5, 7)
    blocks.append((1, 5, 7, U_b, V_b, W_b, 2, 0, 0))
    
    # Block A: M_{2,5,7}
    # i_off=0, j_off=0, k_off=0
    # Split A into A1 (M_{2,2,7}) and A2 (M_{2,3,7})
    
    # A1: M_{2,2,7}
    # i_off=0, j_off=0, k_off=0 (relative to A) -> global i=0, j=0, k=0
    # Split A1 into A1a (M_{2,2,2}) and A1b (M_{2,2,5})
    
    # A1a: M_{2,2,2} (Strassen)
    # i_off=0, j_off=0, k_off=0 (relative to A1) -> global i=0, j=0, k=0
    U_a1a, V_a1a, W_a1a = strassen_222()
    blocks.append((2, 2, 2, U_a1a, V_a1a, W_a1a, 0, 0, 0))
    
    # A1b: M_{2,2,5} (Schoolbook)
    # i_off=0, j_off=0, k_off=2 (relative to A1) -> global i=0, j=0, k=2
    U_a1b, V_a1b, W_a1b = schoolbook(2, 2, 5)
    blocks.append((2, 2, 5, U_a1b, V_a1b, W_a1b, 0, 0, 2))
    
    # A2: M_{2,3,7}
    # i_off=0, j_off=2, k_off=0 (relative to A) -> global i=0, j=2, k=0
    # Split A2 into A2a (M_{2,2,7}) and A2b (M_{2,1,7})
    
    # A2a: M_{2,2,7}
    # i_off=0, j_off=2, k_off=0 (relative to A2) -> global i=0, j=2, k=0
    # Split A2a into A2aa (M_{2,2,2}) and A2ab (M_{2,2,5})
    
    # A2aa: M_{2,2,2} (Strassen)
    # i_off=0, j_off=2, k_off=0 (relative to A2a) -> global i=0, j=2, k=0
    U_a2aa, V_a2aa, W_a2aa = strassen_222()
    blocks.append((2, 2, 2, U_a2aa, V_a2aa, W_a2aa, 0, 2, 0))
    
    # A2ab: M_{2,2,5} (Schoolbook)
    # i_off=0, j_off=2, k_off=2 (relative to A2a) -> global i=0, j=2, k=2
    U_a2ab, V_a2ab, W_a2ab = schoolbook(2, 2, 5)
    blocks.append((2, 2, 5, U_a2ab, V_a2ab, W_a2ab, 0, 2, 2))
    
    # A2b: M_{2,1,7} (Schoolbook)
    # i_off=0, j_off=4, k_off=0 (relative to A2) -> global i=0, j=4, k=0
    U_a2b, V_a2b, W_a2b = schoolbook(2, 1, 7)
    blocks.append((2, 1, 7, U_a2b, V_a2b, W_a2b, 0, 4, 0))

    # Calculate total rank
    total_rank = sum(len(U) for _, _, _, U, _, _, _, _, _ in blocks)
    
    # Initialize global matrices
    U_total = [[0] * total_rank for _ in range(3 * 5)]
    V_total = [[0] * total_rank for _ in range(5 * 7)]
    W_total = [[0] * total_rank for _ in range(7 * 3)]
    
    # Fill in the matrices
    rank_offset = 0
    for n_b, m_b, p_b, U_b, V_b, W_b, i_off, j_off, k_off in blocks:
        block_rank = len(U_b)
        
        for r in range(block_rank):
            global_r = rank_offset + r
            
            # Fill U
            for i in range(n_b):
                for j in range(m_b):
                    local_u_row = i * m_b + j
                    val = U_b[local_u_row][r]
                    if val != 0:
                        global_u_row = (i + i_off) * 5 + (j + j_off)
                        U_total[global_u_row][global_r] += val
                        
            # Fill V
            for j in range(m_b):
                for k in range(p_b):
                    local_v_row = j * p_b + k
                    val = V_b[local_v_row][r]
                    if val != 0:
                        global_v_row = (j + j_off) * 7 + (k + k_off)
                        V_total[global_v_row][global_r] += val
                        
            # Fill W
            for k in range(p_b):
                for i in range(n_b):
                    local_w_row = k * n_b + i
                    val = W_b[local_w_row][r]
                    if val != 0:
                        global_w_row = (k + k_off) * 3 + (i + i_off)
                        W_total[global_w_row][global_r] += val
                        
        rank_offset += block_rank

    return {'U': U_total, 'V': V_total, 'W': W_total}
# EVOLVE-BLOCK-END
