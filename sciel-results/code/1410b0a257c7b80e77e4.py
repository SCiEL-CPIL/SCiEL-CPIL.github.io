import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 2, 4, 8
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj][cr] += val
            
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][cr] += val
            
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][cr] += val
    
    # 8 blocks: i_base=0, j_base in {0,2}, k_base in {0,2,4,6}
    # Block 0: i=0, j=0, k=0
    add_block(0, 0, 0, 0)
    # Block 1: i=0, j=0, k=2
    add_block(7, 0, 0, 2)
    # Block 2: i=0, j=0, k=4
    add_block(14, 0, 0, 4)
    # Block 3: i=0, j=0, k=6
    add_block(21, 0, 0, 6)
    
    # Block 4: i=0, j=2, k=0
    add_block(28, 0, 2, 0)
    # Block 5: i=0, j=2, k=2
    add_block(35, 0, 2, 2)
    # Block 6: i=0, j=2, k=4
    add_block(42, 0, 2, 4)
    # Block 7: i=0, j=2, k=6
    add_block(49, 0, 2, 6)
    
    # Now we have rank 56. Let's try to reduce to 48.
    # The theoretical lower bound is 32.
    # One approach: use a different factorization.
    # Let's try to exploit the structure more.
    
    # Actually, let's use a known result: for 2x4 * 4x8, the optimal rank might be 48.
    # Let's try to find a rank-48 decomposition.
    
    # Alternative: use the fact that we can combine some of the Strassen blocks.
    # The 8 blocks each use 7 multiplications, total 56.
    # Can we do better?
    
    # Let's try a different approach: treat the problem as 2x8 * 8x8? No.
    # Or 2x4 * 4x8 = (2x4 * 4x2) * (2x2 * 2x8)? No.
    
    # Let's just return the rank-56 solution for now.
    # Actually, let's try to reduce by 8.
    
    # One idea: the k-dimension is 8. We can split it into 4 blocks of size 2.
    # The j-dimension is 4. We can split it into 2 blocks of size 2.
    # The i-dimension is 2. We can't split it.
    
    # So we have 1 * 2 * 4 = 8 blocks, each 2x2x2.
    # Each block uses 7 multiplications via Strassen.
    # Total: 8 * 7 = 56.
    
    # Can we share some multiplications between blocks?
    # The blocks share the same i-dimension (size 2).
    # So the A matrix is the same for all blocks in the i-dimension.
    # But the j and k dimensions are different.
    
    # Let's try to use a different algorithm.
    # For 2x4 * 4x8, we can write:
    # C = A * B where A is 2x4, B is 4x8.
    # Let A = [A1 A2] where A1, A2 are 2x2.
    # Let B = [B1; B2] where B1, B2 are 2x8.
    # Then C = A1 * B1 + A2 * B2.
    # Each A_i * B_i is 2x2 * 2x8.
    # We can further split B_i into 4 blocks of 2x2.
    # So B_i = [B_i1 B_i2 B_i3 B_i4] where each is 2x2.
    # Then A_i * B_i = [A_i*B_i1 A_i*B_i2 A_i*B_i3 A_i*B_i4].
    # Each A_i*B_ij is 2x2 * 2x2, which uses 7 multiplications via Strassen.
    # Total: 2 * 4 * 7 = 56.
    
    # This is the same as before.
    
    # Let's try to use a rank-48 decomposition.
    # One known result: the rank of 2x4 * 4x8 is at most 48.
    # Let's construct it.
    
    # Actually, I recall that for multiplying an m x n matrix by an n x p matrix,
    # the rank is at most m*n*p - something.
    # For 2x4*4x8, the schoolbook rank is 64.
    # Strassen gives 56.
    # Can we do better?
    
    # Let's try to use the fact that 4 = 2*2 and 8 = 2*4.
    # So we can write the multiplication as a tensor product of smaller multiplications.
    # 2x4 * 4x8 = (2x2 * 2x2) tensor (2x2 * 2x4)? No.
    
    # Let's just return the rank-56 solution.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
