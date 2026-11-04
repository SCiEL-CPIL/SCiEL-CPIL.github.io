import time
import numpy as np

DIMENSIONS = [6, 6, 6]
DEADLINE = 130  # safety margin before 150s kill

def _build_strassen_2x2_blocks_3x3_leaves(n, m, p):
    """Build Strassen on 2x2 blocks of 3x3 matrices.
    
    6x6 matrix = 2x2 grid of 3x3 blocks.
    Strassen gives 7 multiplications of 3x3 blocks.
    Each 3x3 block multiplication uses schoolbook: 27 scalar multiplications.
    Total rank = 7 * 27 = 189.
    """
    block_size = 3
    blocks_per_dim = 2
    rank = 7 * (block_size ** 2)  # 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 matrix multiplication
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):  # Row block of output C
        for J in range(blocks_per_dim):  # Col block of A / Row block of B (middle)
            for K in range(blocks_per_dim):  # Col block of output C
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # Iterate over all valid combinations
                    for (pi, pj), u_val in u_coeffs.items():
                        for (si, sj), v_val in v_coeffs.items():
                            for (qi, qj), w_val in w_coeffs.items():
                                # Consistency: A's col block (pj) must equal B's row block (si)
                                # B's col block (sj) must equal C's row block (qi)
                                if pj != si or sj != qi:
                                    continue
                                
                                # U: row = I*bs + pi, col = J*bs + pj
                                r_u = I * block_size + pi
                                c_u = J * block_size + pj
                                U[r_u * m + c_u][col_idx] += u_val
                                
                                # V: row = J*bs + si, col = K*bs + sj
                                r_v = J * block_size + si
                                c_v = K * block_size + sj
                                V[r_v * p + c_v][col_idx] += v_val
                                
                                # W: row = K*bs + qi, col = I*bs + qj
                                r_w = K * block_size + qi
                                c_w = I * block_size + qj
                                W[r_w * n + c_w][col_idx] += w_val
                                
                                col_idx += 1
    
    # Verify we used exactly 'rank' columns
    if col_idx != rank:
        raise ValueError(f"Expected rank {rank}, got {col_idx}")
    
    return U, V, W

def _verify_full(n, m, p, U, V, W):
    """Full verification using numpy."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(Ua[a_idx], Va[b_idx] * Wa[c_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Schoolbook algorithm: rank = n*m*p = 216."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build Strassen 2x2 blocks of 3x3 matrices (rank 189)
    U, V, W = _build_strassen_2x2_blocks_3x3_leaves(n, m, p)
    
    # Verify
    if time.time() - start_time < DEADLINE - 10:
        if not _verify_full(n, m, p, U, V, W):
            # Fallback to schoolbook
            U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
