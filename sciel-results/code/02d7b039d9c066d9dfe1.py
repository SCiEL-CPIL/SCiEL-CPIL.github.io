import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=3 into m1=2 + m2=1
    # Block 1: 2x2x6 (A: 2x2, B: 2x6, C: 2x6) -> rank 21 (3 Strassen 2x2x2 blocks)
    # Block 2: 2x1x6 (A: 2x1, B: 1x6, C: 2x6) -> rank 6
    # Total rank = 27
    
    rank_block1 = 21  # 3 * 7
    rank_block2 = 6
    total_rank = rank_block1 + rank_block2
    
    U = np.zeros((n * m, total_rank))  # (6, 27)
    V = np.zeros((m * p, total_rank))  # (18, 27)
    W = np.zeros((p * n, total_rank))  # (12, 27)
    
    # Block 1: 2x2x6
    # A indices: i in 0..1, j in 0..1 (local), global A index = i*3 + j
    # B indices: j in 0..1 (local), k in 0..5, global B index = j*6 + k
    # C indices: i in 0..1, k in 0..5, global C index = k*2 + i (W[k*n+i])
    
    # Split k into 3 pairs: (0,1), (2,3), (4,5)
    # For each pair, use Strassen 2x2x2
    
    # Strassen 2x2x2 terms (local indices):
    # A: a00=0, a01=1, a10=2, a11=3
    # B: b00=0, b01=1, b10=2, b11=3
    # C: c00=0, c01=1, c10=2, c11=3
    
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11) -> c00, c11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (a10+a11)b00 -> c10, c11 (c11 with -1)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a00(b01-b11) -> c01, c11
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = a11(b10-b00) -> c00, c10
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a00+a01)b11 -> c00 (-1), c01
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (a10-a00)(b00+b01) -> c11
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (a01-a11)(b10+b11) -> c00
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for block_k in range(3):
        k_offset = block_k * 2
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
            col = block_k * 7 + t_idx
            
            # U: local A index -> global A index = i*3 + j
            for local_idx, val in u_dict.items():
                i = local_idx // 2
                j = local_idx % 2
                global_a = i * 3 + j
                U[global_a, col] = val
            
            # V: local B index -> global B index = j*6 + (k_offset + k_local)
            for local_idx, val in v_dict.items():
                j = local_idx // 2
                k_local = local_idx % 2
                global_b = j * 6 + (k_offset + k_local)
                V[global_b, col] = val
            
            # W: local C index -> global C index = (k_offset + k_local)*2 + i
            for local_idx, val in w_dict.items():
                i = local_idx // 2
                k_local = local_idx % 2
                global_c = (k_offset + k_local) * 2 + i
                W[global_c, col] = val
    
    # Block 2: 2x1x6
    # A indices: i in 0..1, j=2 (local), global A index = i*3 + 2
    # B indices: j=2 (local), k in 0..5, global B index = 2*6 + k = 12 + k
    # C indices: i in 0..1, k in 0..5, global C index = k*2 + i
    # Rank 6: for each k, one term
    for k in range(6):
        col = rank_block1 + k
        
        # U: both i=0 and i=1 contribute with coefficient 1
        U[0 * 3 + 2, col] = 1  # A[0,2]
        U[1 * 3 + 2, col] = 1  # A[1,2]
        
        # V: B[2,k]
        V[2 * 6 + k, col] = 1
        
        # W: C[i,k] for both i
        W[k * 2 + 0, col] = 1  # C[0,k]
        W[k * 2 + 1, col] = 1  # C[1,k]
    
    # Verify the decomposition
    def verify(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1) > tol:
                        return False
        # Check zero entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if not (j == j2 and i == i2 and k == k2):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > tol:
                            return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
