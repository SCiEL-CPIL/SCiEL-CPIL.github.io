import time
import random
import numpy as np

def _find_2x3x3_decomp(max_attempts=5000, target_rank=16):
    """Find a rank-r decomposition for 2x3 x 3x3 matrix multiplication.
    Returns (U, V, W) as lists of lists, or None if not found.
    U shape: (6, r), V shape: (9, r), W shape: (6, r)
    """
    n, m, p = 2, 3, 3
    for r in range(target_rank, 19):
        for attempt in range(max_attempts):
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            for c in range(r):
                num_nonzero = random.randint(2, 5)
                u_pos = random.sample(range(n*m), min(num_nonzero, n*m))
                v_pos = random.sample(range(m*p), min(num_nonzero, m*p))
                w_pos = random.sample(range(p*n), min(num_nonzero, p*n))
                for pos in u_pos:
                    U[pos, c] = random.choice([-1, 1])
                for pos in v_pos:
                    V[pos, c] = random.choice([-1, 1])
                for pos in w_pos:
                    W[pos, c] = random.choice([-1, 1])
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(r):
                            total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a rank 16 decomposition for 2x3x3
    decomp = _find_2x3x3_decomp(max_attempts=2000, target_rank=16)
    
    if decomp is not None:
        U_block, V_block, W_block = decomp
        block_rank = len(U_block[0])
        total_rank = 3 * block_rank
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for g in range(3):
            for c in range(block_rank):
                col = g * block_rank + c
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][col] = U_block[i*m+j][c]
                for j in range(m):
                    for k_local in range(3):
                        k_global = g * 3 + k_local
                        V[j*p + k_global][col] = V_block[j*3 + k_local][c]
                for k_local in range(3):
                    k_global = g * 3 + k_local
                    for i in range(n):
                        W[k_global*n + i][col] = W_block[k_local*n + i][c]
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(total_rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
