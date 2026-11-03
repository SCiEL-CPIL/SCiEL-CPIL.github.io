import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank is 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # U is (i,j) flattened, V is (j,k) flattened, W is (k,i) flattened
    # Each entry in rank r contributes to tensor[i,j,k] as U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(1,0): 1, (1,1): 1},   # r=1
        {(0,0): 1},              # r=2
        {(1,1): 1},              # r=3
        {(0,0): 1, (0,1): 1},   # r=4
        {(1,0): 1, (0,0): -1},  # r=5
        {(0,1): 1, (1,1): -1},  # r=6
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,0): 1},              # r=1
        {(0,1): 1, (1,1): -1},  # r=2
        {(1,0): 1, (0,0): -1},  # r=3
        {(1,1): 1},              # r=4
        {(0,0): 1, (0,1): 1},   # r=5
        {(1,0): 1, (1,1): 1},   # r=6
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,1): 1, (1,1): -1},  # r=1
        {(1,0): 1, (1,1): 1},   # r=2
        {(0,0): 1, (0,1): 1},   # r=3
        {(0,0): -1, (1,0): 1},  # r=4
        {(1,1): 1},              # r=5
        {(0,0): 1},              # r=6
    ]
    
    # 8 sub-blocks of 2x2x2 for 2x4x8
    # Block (bI, bJ, bK) where bI in {0,1}, bJ in {0,1,2,3}, bK in {0,1}
    # This gives us 2*4*2 = ... wait, we need to partition properly.
    
    # Actually, let's think about this differently.
    # A is 2x4, B is 4x8. 
    # We can partition A into two 2x2 blocks: A = [A0 | A1] where A0 is cols 0,1 and A1 is cols 2,3
    # We can partition B into four 4x4 ... no, B is 4x8. 
    # B rows correspond to the 4 rows, which we partitioned A into columns 0-1 and 2-3.
    # So B = [B_top; B_bottom] where B_top is rows 0,1 and B_bottom is rows 2,3. Each is 2x8.
    # So C = A0 * B_top + A1 * B_bottom
    
    # Now A0 is 2x2, B_top is 2x8.
    # B_top can be partitioned into four 2x2 blocks: B_top = [Bt0 | Bt1 | Bt2 | Bt3] (cols 0-1, 2-3, 4-5, 6-7)
    # So A0*B_top = A0*Bt0 + A0*Bt1 + A0*Bt2 + A0*Bt3
    # Each A0*BtX is 2x2 * 2x2, which takes rank 7 using Strassen.
    # So A0*B_top takes 4*7 = 28 rank.
    
    # Similarly, A1 is 2x2, B_bottom is 2x8 = [Bb0 | Bb1 | Bb2 | Bb3]
    # A1*B_bottom = A1*Bb0 + A1*Bb1 + A1*Bb2 + A1*Bb3
    # Takes another 4*7 = 28 rank.
    
    # Total rank: 56. This matches the known bound.
    
    def add_block(base_rank, i_base, j_base, k_base):
        """Add a Strassen decomposition for A[i_base:i_base+2, j_base:j_base+2] * 
           B[j_base:j_base+2, k_base:k_base+2] starting at rank index base_rank."""
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
    
    # A0 * Bt0: i_base=0, j_base=0, k_base=0
    add_block(0, 0, 0, 0)
    
    # A0 * Bt1: i_base=0, j_base=0, k_base=2
    add_block(7, 0, 0, 2)
    
    # A0 * Bt2: i_base=0, j_base=0, k_base=4
    add_block(14, 0, 0, 4)
    
    # A0 * Bt3: i_base=0, j_base=0, k_base=6
    add_block(21, 0, 0, 6)
    
    # A1 * Bb0: i_base=0, j_base=2, k_base=0
    add_block(28, 0, 2, 0)
    
    # A1 * Bb1: i_base=0, j_base=2, k_base=2
    add_block(35, 0, 2, 2)
    
    # A1 * Bb2: i_base=0, j_base=2, k_base=4
    add_block(42, 0, 2, 4)
    
    # A1 * Bb3: i_base=0, j_base=2, k_base=6
    add_block(49, 0, 2, 6)
    
    # Check time
    if time.time() > deadline:
        # If we ran out of time, return what we have (which should be complete)
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
