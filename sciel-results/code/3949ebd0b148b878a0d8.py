import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # Use block Strassen: split i into {0,1} and {2,3}, k into {0,1,2,3,4,5}
    # For each i_block (2 blocks) and for the j-dimension (4 values),
    # we have a 2x4x6 sub-tensor.
    # Actually, let's split all three dimensions into 2x2 blocks where possible.
    # i: 2 blocks of size 2
    # j: 2 blocks of size 2
    # k: 3 blocks of size 2 (since p=6)
    
    # For each (i_block, j_block, k_block), we have a 2x2x2 sub-tensor.
    # There are 2*2*3 = 12 such sub-tensors.
    # Each 2x2x2 sub-tensor can be decomposed with rank 7 (Strassen) vs rank 8 (schoolbook).
    # Total rank = 12 * 7 = 84 (vs schoolbook 96).
    
    # But we can do better by using Strassen across the j-blocks as well.
    # Let's use a 2-level Strassen:
    # Level 1: Split j into 2 blocks of size 2.
    # For each j_block, we have a 4x2x6 sub-tensor.
    # Level 2: For each 4x2x6 sub-tensor, split i into 2 blocks of size 2, k into 3 blocks of size 2.
    # This gives 2*3 = 6 sub-sub-tensors of size 2x2x2 per j_block.
    # Total 2x2x2 sub-tensors: 2 * 6 = 12.
    # Each with rank 7: total rank = 12 * 7 = 84.
    
    # Let's implement this.
    
    # Base 2x2x2 Strassen terms
    base_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    base_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    base_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    U = [[0] * 84 for _ in range(n * m)]  # (16, 84)
    V = [[0] * 84 for _ in range(m * p)]  # (24, 84)
    W = [[0] * 84 for _ in range(p * n)]  # (24, 84)
    
    col = 0
    # For each j_block (0 or 1), i_block (0 or 1), k_block (0, 1, or 2)
    for jb in range(2):
        for ib in range(2):
            for kb in range(3):
                # j_block jb: j values are 2*jb, 2*jb+1
                # i_block ib: i values are 2*ib, 2*ib+1
                # k_block kb: k values are 2*kb, 2*kb+1
                
                # For this 2x2x2 sub-tensor, use Strassen decomposition (7 terms)
                for r in range(7):
                    # base_u[r] is a 4-vector for the 2x2 A-block
                    # base_v[r] is a 4-vector for the 2x2 B-block
                    # base_w[r] is a 4-vector for the 2x2 C-block
                    
                    # Map to global indices:
                    # A indices: i in {2*ib, 2*ib+1}, j in {2*jb, 2*jb+1}
                    # B indices: j in {2*jb, 2*jb+1}, k in {2*kb, 2*kb+1}
                    # C indices: k in {2*kb, 2*kb+1}, i in {2*ib, 2*ib+1}
                    
                    # base_u[r][a] where a=0,1,2,3 corresponds to (i_offset, j_offset) in {0,1}x{0,1}
                    # a=0: (0,0), a=1: (0,1), a=2: (1,0), a=3: (1,1)
                    for a in range(4):
                        if base_u[r][a] != 0:
                            i_off = a // 2  # 0 or 1
                            j_off = a % 2   # 0 or 1
                            i_global = 2 * ib + i_off
                            j_global = 2 * jb + j_off
                            U[i_global * 4 + j_global][col] = base_u[r][a]
                    
                    # base_v[r][b] where b=0,1,2,3 corresponds to (j_offset, k_offset) in {0,1}x{0,1}
                    for b in range(4):
                        if base_v[r][b] != 0:
                            j_off = b // 2  # 0 or 1
                            k_off = b % 2   # 0 or 1
                            j_global = 2 * jb + j_off
                            k_global = 2 * kb + k_off
                            V[j_global * 6 + k_global][col] = base_v[r][b]
                    
                    # base_w[r][c] where c=0,1,2,3 corresponds to (k_offset, i_offset) in {0,1}x{0,1}
                    for c in range(4):
                        if base_w[r][c] != 0:
                            k_off = c // 2  # 0 or 1
                            i_off = c % 2   # 0 or 1
                            k_global = 2 * kb + k_off
                            i_global = 2 * ib + i_off
                            W[k_global * 4 + i_global][col] = base_w[r][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
