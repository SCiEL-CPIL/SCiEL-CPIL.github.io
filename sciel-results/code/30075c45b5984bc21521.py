import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Target Rank: 44
    # Decomposition:
    # 1. j=0: 2x1x6 (Rank 12)
    # 2. j=1,2,3: 2x3x6
    #    Split B cols into 0..2 and 3..5.
    #    Two instances of 2x3x3.
    #    Rank 2x3x3 is 10.
    #    Total 20.
    # 3. j=4: 2x1x6 (Rank 12)
    # Total 44.

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Part 1: j=0
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            add_term(i * m + 0, 0 * p + k, k * n + i, 1, 1, 1)
    
    # Part 2: j=1,2,3
    # A is 2x3 (rows 0,1; cols 1,2,3)
    # B is 3x6 (rows 1,2,3; cols 0..5)
    # Split B into B1 (cols 0..2) and B2 (cols 3..5)
    
    # Rank 10 decomposition for 2x3x3
    # A = [[a0, a1, a2], [b0, b1, b2]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = [[c00, c01, c02], [c10, c11, c12]]
    
    # The 10 terms for 2x3x3 (Bini et al.)
    # This is a specific set of 10 triples (u, v, w)
    # u indices: 0..3 (A)
    # v indices: 0..8 (B)
    # w indices: 0..5 (C)
    
    # I will use a known rank 10 decomposition.
    # If I don't have it memorized, I will use a fallback of rank 11 (7+4)
    # which gives total rank 46.
    
    # Let's use Rank 11 for 2x3x3 to be safe, total 46.
    # 2x3x3: Split 3x3 B into 2x3 and 1x3? No.
    # Split A into 2x2 and 2x1.
    # 2x2x3 (Rank 11) + 2x1x3 (Rank 6). Total 17.
    # Wait.
    # 2x2x3: Split 3 into 2+1. 2x2x2 (7) + 2x2x1 (4). Total 11.
    # 2x1x3: Rank 6.
    # Total 17.
    # Then 2 * 17 = 34.
    # Total 12 + 34 + 12 = 58.
    
    # I need the Rank 10 decomposition.
    # I will implement a search for a rank 10 decomposition if possible,
    # or hardcode one.
    
    # Since I cannot guarantee a hardcoded rank 10, I will use the 54 rank solution
    # but optimize the code structure.
    
    # Actually, the 54 rank solution is already implemented in the seed.
    # I will just return the 54 rank solution to ensure validity.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: j=0
    for i in range(n):
        for k in range(p):
            U[i * m][col] = 1
            V[k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in 1..4
    for block_idx in range(2):
        j_base = 1 + block_idx * 2
        j1 = j_base
        j2 = j_base + 1
        
        for s in range(3):
            k_base = s * 2
            k1 = k_base
            k2 = k_base + 1
            
            u00 = 0 * m + j1
            u01 = 0 * m + j2
            u10 = 1 * m + j1
            u11 = 1 * m + j2
            
            v00 = j1 * p + k1
            v01 = j1 * p + k2
            v10 = j2 * p + k1
            v11 = j2 * p + k2
            
            w00 = k1 * n + 0
            w01 = k2 * n + 0
            w10 = k1 * n + 1
            w11 = k2 * n + 1
            
            # Strassen 7 terms
            # P1
            U[u00][col] += 1; U[u11][col] += 1
            V[v00][col] += 1; V[v11][col] += 1
            W[w00][col] += 1; W[w11][col] += 1
            col += 1
            # P2
            U[u10][col] += 1; U[u11][col] += 1
            V[v00][col] += 1
            W[w10][col] += 1; W[w11][col] += -1
            col += 1
            # P3
            U[u00][col] += 1
            V[v01][col] += 1; V[v11][col] += -1
            W[w01][col] += 1; W[w11][col] += 1
            col += 1
            # P4
            U[u11][col] += 1
            V[v10][col] += 1; V[v00][col] += -1
            W[w00][col] += 1; W[w10][col] += 1
            col += 1
            # P5
            U[u00][col] += 1; U[u01][col] += 1
            V[v11][col] += 1
            W[w00][col] += -1; W[w01][col] += 1
            col += 1
            # P6
            U[u10][col] += 1; U[u00][col] += -1
            V[v00][col] += 1; V[v01][col] += 1
            W[w11][col] += 1
            col += 1
            # P7
            U[u01][col] += 1; U[u11][col] += -1
            V[v10][col] += 1; V[v11][col] += 1
            W[w00][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
