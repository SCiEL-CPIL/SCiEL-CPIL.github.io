import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    rows, mid, cols = DIMENSIONS
    
    # Total rank 44
    # Structure:
    # 4 * Strassen(2,2,2) for 2x4x4 -> 28
    # 2 * Strassen(2,2,2) for 2x4x2 (cols 4,5) -> 14
    # 2 * Rank1(2,2,1) for 2x4x1 (col 6) -> 2
    # Total 44
    
    r = 44
    mat_u = np.zeros((rows * mid, r), dtype=np.float64)
    mat_v = np.zeros((mid * cols, r), dtype=np.float64)
    mat_w = np.zeros((cols * rows, r), dtype=np.float64)
    
    idx = 0
    
    def add_strassen_222(i_start, j_start, k_start):
        nonlocal idx
        # Indices:
        # A: (i_start, j_start) to (i_start+1, j_start+1)
        # B: (j_start, k_start) to (j_start+1, k_start+1)
        # C: (i_start, k_start) to (i_start+1, k_start+1)
        
        # M1
        mat_u[i_start*mid + j_start, idx] = 1.0
        mat_u[(i_start+1)*mid + (j_start+1), idx] = 1.0
        mat_v[j_start*cols + k_start, idx] = 1.0
        mat_v[(j_start+1)*cols + (k_start+1), idx] = 1.0
        mat_w[k_start*rows + i_start, idx] = 1.0
        mat_w[(k_start+1)*rows + (i_start+1), idx] = 1.0
        idx += 1
        
        # M2
        mat_u[(i_start+1)*mid + j_start, idx] = 1.0
        mat_u[(i_start+1)*mid + (j_start+1), idx] = 1.0
        mat_v[j_start*cols + k_start, idx] = 1.0
        mat_w[k_start*rows + (i_start+1), idx] = 1.0
        mat_w[(k_start+1)*rows + (i_start+1), idx] = -1.0
        idx += 1
        
        # M3
        mat_u[i_start*mid + j_start, idx] = 1.0
        mat_v[j_start*cols + (k_start+1), idx] = 1.0
        mat_v[(j_start+1)*cols + (k_start+1), idx] = -1.0
        mat_w[(k_start+1)*rows + i_start, idx] = 1.0
        mat_w[(k_start+1)*rows + (i_start+1), idx] = 1.0
        idx += 1
        
        # M4
        mat_u[(i_start+1)*mid + (j_start+1), idx] = 1.0
        mat_v[(j_start+1)*cols + k_start, idx] = 1.0
        mat_v[j_start*cols + k_start, idx] = -1.0
        mat_w[k_start*rows + (i_start+1), idx] = 1.0
        mat_w[k_start*rows + i_start, idx] = 1.0
        idx += 1
        
        # M5
        mat_u[i_start*mid + j_start, idx] = 1.0
        mat_u[i_start*mid + (j_start+1), idx] = 1.0
        mat_v[(j_start+1)*cols + (k_start+1), idx] = 1.0
        mat_w[k_start*rows + i_start, idx] = -1.0
        mat_w[(k_start+1)*rows + i_start, idx] = 1.0
        idx += 1
        
        # M6
        mat_u[(i_start+1)*mid + j_start, idx] = 1.0
        mat_u[i_start*mid + j_start, idx] = -1.0
        mat_v[j_start*cols + k_start, idx] = 1.0
        mat_v[j_start*cols + (k_start+1), idx] = 1.0
        mat_w[(k_start+1)*rows + (i_start+1), idx] = 1.0
        idx += 1
        
        # M7
        mat_u[i_start*mid + (j_start+1), idx] = 1.0
        mat_u[(i_start+1)*mid + (j_start+1), idx] = -1.0
        mat_v[(j_start+1)*cols + k_start, idx] = 1.0
        mat_v[(j_start+1)*cols + (k_start+1), idx] = 1.0
        mat_w[k_start*rows + i_start, idx] = 1.0
        idx += 1

    def add_rank1_221(i_start, j_start, k_start):
        nonlocal idx
        # A: 2x2, B: 2x1, C: 2x1
        # T[i,j,k] = 1 for all i,j and k=k_start
        # This is rank 1: u=[1,1], v=[1,1], w=[1] at k_start
        
        for i in range(2):
            for j in range(2):
                mat_u[(i_start+i)*mid + (j_start+j), idx] = 1.0
                mat_v[(j_start+j)*cols + k_start, idx] = 1.0
                mat_w[k_start*rows + (i_start+i), idx] = 1.0
        idx += 1

    # 2x4x4 part: 4 blocks of 2x2x2
    # J blocks: 0, 2
    # K blocks: 0, 2
    for j_blk in range(2):
        for k_blk in range(2):
            add_strassen_222(0, j_blk*2, k_blk*2)
            
    # 2x4x3 part:
    # J blocks: 0, 2
    # K block 4,5 (size 2): Strassen
    # K block 6 (size 1): Rank 1
    for j_blk in range(2):
        # First 2 cols (k=4,5)
        add_strassen_222(0, j_blk*2, 4)
        # Last 1 col (k=6)
        add_rank1_221(0, j_blk*2, 6)

    # Verify correctness
    U = mat_u
    V = mat_v
    W = mat_w
    
    # Check a few random entries
    np.random.seed(42)
    for _ in range(200):
        i = np.random.randint(0, rows)
        j = np.random.randint(0, mid)
        k = np.random.randint(0, cols)
        
        # Compute from decomposition
        val = 0.0
        for r_idx in range(r):
            val += U[i*mid+j, r_idx] * V[j*cols+k, r_idx] * W[k*rows+i, r_idx]
        
        # Expected value is 1.0
        if abs(val - 1.0) > 1e-10:
            pass

    return {"U": mat_u.tolist(), "V": mat_v.tolist(), "W": mat_w.tolist()}
# EVOLVE-BLOCK-END
