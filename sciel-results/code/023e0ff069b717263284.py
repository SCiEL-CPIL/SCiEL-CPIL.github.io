import time
import sys

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Constructs a rank-88 tensor decomposition for 3x4x8 matrix multiplication.
    
    Strategy:
    1. Split the 3x4 matrix A into A1 (1x4) and A2 (2x4).
    2. Compute C1 = A1 * B (1x4 * 4x8 -> 1x8) using schoolbook (rank 32).
    3. Compute C2 = A2 * B (2x4 * 4x8 -> 2x8) using Strassen blocks.
       Split A2's 4 columns into two 2-col blocks.
       Split B's 8 rows? No, B is 4x8.
       Split B's 4 rows into two 2-row blocks corresponding to A2's column splits.
       Wait, A2 is 2x4. Columns j=0,1 and j=2,3.
       B is 4x8. Rows k=0,1 and k=2,3.
       C2[i, k] = sum_j A2[i, j] B[j, k].
       Split sum into sum_{j=0,1} + sum_{j=2,3}.
       Term 1: A2[:, 0:2] * B[0:2, :]. (2x2 * 2x8).
       Term 2: A2[:, 2:4] * B[2:4, :]. (2x2 * 2x8).
       
       Each (2x2 * 2x8) is decomposed into 4 (2x2 * 2x2) Strassen blocks.
       (2x2 * 2x2) has rank 7.
       So (2x2 * 2x8) has rank 4 * 7 = 28.
       Two such blocks for A2: 2 * 28 = 56.
       
       Total rank: 32 + 56 = 88.
    """
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    start_time = time.time()
    deadline = 120.0
    
    rank = 88
    # Initialize factor matrices
    # U: (n*m, rank) = (12, 88)
    # V: (m*p, rank) = (32, 88)
    # W: (p*n, rank) = (24, 88)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (1x4) * B (4x8)
    # A1 is row 0 of A. Indices i=0, j=0..3.
    # B is 4x8. Indices j=0..3, k=0..7.
    # C1 is row 0 of C. Indices i=0, k=0..7.
    # Schoolbook: 4 * 8 = 32 terms.
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}

    # Part 2: A2 (2x4) * B (4x8)
    # A2 rows i=1,2. Cols j=0..3.
    # B rows j=0..3. Cols k=0..7.
    # C2 rows i=1,2. Cols k=0..7.
    
    # Strassen 2x2x2 factors
    # Local indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_U = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1]  # M7: A11-A01
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B10-B00
        [-1, 0, 1, 0], # M4: B11-B01
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1: C00+=1, C11+=1
        [0, 0, 1, -1], # M2: C10+=1, C11-=1
        [0, 1, 0, 1], # M3: C01+=1, C11+=1
        [1, 0, 1, 0], # M4: C00+=1, C10+=1
        [-1, 1, 0, 0], # M5: C00-=1, C01+=1
        [0, 0, 0, 1], # M6: C11+=1
        [1, 0, 0, 0]  # M7: C00+=1
    ]
    
    # Iterate over column blocks of A2 (j_start)
    # j_start = 0 (cols 0,1) and j_start = 2 (cols 2,3)
    for j_block in range(2):
        j_start = j_block * 2
        # Iterate over row blocks of B (k_block)
        # Wait, B is 4x8. The "j" dimension of B matches the "j" dimension of A.
        # So for A2 cols 0,1, we use B rows 0,1.
        # For A2 cols 2,3, we use B rows 2,3.
        
        # The 8 columns of B (k=0..7) are split into 4 blocks of 2?
        # No, the Strassen block is 2x2 * 2x2.
        # Here we have 2x2 * 2x8.
        # We decompose 2x8 into four 2x2 blocks along the k dimension.
        for k_block in range(4):
            k_start = k_block * 2
            
            # 7 Strassen terms
            for t in range(7):
                # Map local indices to global
                # A2 local i': 0->global i=1, 1->global i=2
                # A2 local j': 0->global j=j_start+0, 1->global j=j_start+1
                # B local j': 0->global j=j_start+0, 1->global j=j_start+1
                # B local k': 0->global k=k_start+0, 1->global k=k_start+1
                # C2 local i': 0->global i=1, 1->global i=2
                # C2 local k': 0->global k=k_start+0, 1->global k=k_start+1
                
                for l in range(4):
                    val = strassen_U[t][l]
                    if val != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += val
                        
                for l in range(4):
                    val = strassen_V[t][l]
                    if val != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += val
                        
                for l in range(4):
                    val = strassen_W[t][l]
                    if val != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i][col] += val
                        
                col += 1
                
                if time.time() > deadline - 2.0:
                    break
            
            if time.time() > deadline - 2.0:
                break
        
        if time.time() > deadline - 2.0:
            break

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
