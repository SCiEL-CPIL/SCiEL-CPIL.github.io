import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 6  # A is n x m, B is m x p, C is n x p
    
    # Use rank-3 decomposition of 2x2x2 tensor composed over blocks
    # A (2x4) split into A0 (2x2, cols 0-1) and A1 (2x2, cols 2-3)
    # B (4x6) split into B00,B01,B02 (2x2, rows 0-1, cols 2k-2k+1) 
    #            and B10,B11,B12 (2x2, rows 2-3, cols 2k-2k+1)
    # C = A0 @ B_top + A1 @ B_bottom
    # Each 2x2 @ 2x2 uses rank-3, so 6 blocks * 3 = 18 terms
    
    # Rank-3 decomposition of 2x2 matrix multiplication:
    # For A2 = [[a,b],[c,d]], B2 = [[e,f],[g,h]], C2 = A2*B2
    # M1 = a*e
    # M2 = (a+d)*h
    # M3 = (d-a)*(h-e)
    # C2[0,0] = M1
    # C2[0,1] = M2 - M3 - M1  ... let me verify
    # C2[0,1] = a*f + b*h
    # C2[1,0] = c*e + d*g
    # C2[1,1] = c*f + d*h
    
    # Actually, let me use a verified rank-3 scheme:
    # M1 = a*e
    # M2 = (a+d)*h  
    # M3 = (d-a)*(h-e)
    # Then:
    # C00 = M1
    # C01 = M2 - M3 - M1  = (a+d)h - (d-a)(h-e) - ae = ah+dh - (dh-de-ah+ae) - ae = ah+dh-dh+de+ah-ae-ae = 2ah+de-2ae... not right
    
    # Let me use the correct one from literature (Bini et al.):
    # M1 = a*e
    # M2 = (a+d)*f
    # M3 = (d-a)*g
    # C00 = M1
    # C01 = M2 - M1  = (a+d)f - ae = af+df-ae... not right either
    
    # I'll use a different, verified rank-3 decomposition:
    # M1 = a*e
    # M2 = (a+d)*h
    # M3 = (d-a)*(h-e)
    # C00 = M1
    # C10 = M3
    # C11 = M2
    # C01 = M2 - M3 - M1 + 2*M1... 
    
    # This is getting too complex. Let me just use Strassen (rank 7) but try to 
    # find a better blocking that reduces total rank below 42.
    
    # Alternative: use the fact that we can share computations across blocks.
    # For now, return the known rank-42 solution as baseline.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_a_idx(group, local_i, local_j):
        global_col = local_j + 2 * (group - 1)
        global_row = local_i
        return global_row * m + global_col
    
    def get_b_idx(group, local_i, local_j, block_col):
        global_row = local_i + 2 * (group - 1)
        global_col = local_j + 2 * block_col
        return global_row * p + global_col
    
    def get_c_idx(local_i, local_j, block_col):
        global_col = local_j + 2 * block_col
        global_row = local_i
        return global_col * n + global_row
    
    col = 0
    for group in [1, 2]:
        for block_col in [0, 1, 2]:
            # P1: (A11+A22)(B11+B22)
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # P2: (A21+A22)B11
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] -= 1
            col += 1
            
            # P3: A11(B12-B22)
            U[get_a_idx(group, 0, 0)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # P4: A22(B21-B11)
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            col += 1
            
            # P5: (A11+A12)B22
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 0, 1)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            col += 1
            
            # P6: (A21-A11)(B11+B12)
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 0, 0)][col] -= 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # P7: (A12-A22)(B21+B22)
            U[get_a_idx(group, 0, 1)][col] += 1
            U[get_a_idx(group, 1, 1)][col] -= 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
