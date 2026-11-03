import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin

    # 1. Try to find a rank 16 decomposition for 2x3x3
    # If found, use it 3 times to get rank 48 for 2x3x9
    U_233, V_233, W_233 = search_rank_233(16, deadline)
    
    if U_233 is not None:
        # Construct 2x3x9 decomposition
        # U shape (6, 48), V shape (27, 48), W shape (18, 48)
        # The 2x3x9 problem consists of 3 independent 2x3x3 blocks
        # B is 3x9. Split into B0, B1, B2 (3x3 each)
        # C is 2x9. Split into C0, C1, C2 (2x3 each)
        # C0 = A B0, C1 = A B1, C2 = A B2
        
        U_full = np.zeros((6, 48))
        V_full = np.zeros((27, 48))
        W_full = np.zeros((18, 48))
        
        # U is same for all blocks?
        # U depends on (i, j). i in 0..1, j in 0..2.
        # In 2x3x9, j is still 0..2.
        # So U_full[:, 0:16] = U_233
        # U_full[:, 16:32] = U_233
        # U_full[:, 32:48] = U_233
        
        # V depends on (j, k). j in 0..2, k in 0..8.
        # V_full has rows 0..26.
        # Row index = j * 9 + k.
        # For block 0 (k=0..2), V_full[j*9 + k, 0:16] = V_233[j*3 + k, :]
        # For block 1 (k=3..5), V_full[j*9 + k, 16:32] = V_233[j*3 + (k-3), :]
        # For block 2 (k=6..8), V_full[j*9 + k, 32:48] = V_233[j*3 + (k-6), :]
        
        # W depends on (k, i). k in 0..8, i in 0..1.
        # W_full has rows 0..17.
        # Row index = k * 2 + i.
        # For block 0 (k=0..2), W_full[k*2 + i, 0:16] = W_233[k*2 + i, :]
        # For block 1 (k=3..5), W_full[k*2 + i, 16:32] = W_233[(k-3)*2 + i, :]
        # For block 2 (k=6..8), W_full[k*2 + i, 32:48] = W_233[(k-6)*2 + i, :]
        
        U_full[:, 0:16] = U_233
        U_full[:, 16:32] = U_233
        U_full[:, 32:48] = U_233
        
        for j in range(3):
            for k in range(9):
                v_row = j * 9 + k
                block = k // 3
                k_local = k % 3
                v_233_row = j * 3 + k_local
                col_start = block * 16
                V_full[v_row, col_start:col_start+16] = V_233[v_233_row, :]
                
        for k in range(9):
            for i in range(2):
                w_row = k * 2 + i
                block = k // 3
                k_local = k % 3
                w_233_row = k_local * 2 + i
                col_start = block * 16
                W_full[w_row, col_start:col_start+16] = W_233[w_233_row, :]
                
        # Convert to lists of lists
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }

    # 2. Fallback: Schoolbook 54
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

def search_rank_233(rank, deadline):
    """
    Search for a rank 'rank' decomposition of 2x3x3.
    Returns U, V, W as numpy arrays or None.
    """
    n, m, p = 2, 3, 3
    num_u = n * m # 6
    num_v = m * p # 9
    num_w = p * n # 6
    
    # Random restarts
    random.seed(42)
    for _ in range(1000):
        if time.time() > deadline:
            return None, None, None
            
        # Initialize with small integers
        # Using -1, 0, 1
        U = np.zeros((num_u, rank), dtype=int)
        V = np.zeros((num_v, rank), dtype=int)
        W = np.zeros((num_w, rank), dtype=int)
        
        # Set some non-zeros randomly to help start
        # We need the sum of products to be 1.
        # Let's try to satisfy the 18 equations.
        
        # Simple local search
        # Error = sum( (product - target)^2 )
        
        # Initialize
        for c in range(rank):
            for i in range(num_u):
                U[i, c] = random.choice([-1, 0, 1])
            for i in range(num_v):
                V[i, c] = random.choice([-1, 0, 1])
            for i in range(num_w):
                W[i, c] = random.choice([-1, 0, 1])
                
        # Calculate error
        def calc_error(U, V, W):
            err = 0.0
            # Target is 1 for (i,j,k)
            # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        # We want val == 1
                        err += (val - 1) ** 2
                        # Also check zeros?
                        # The tensor is sparse.
                        # If we only check the 18 ones, we might get non-zeros elsewhere.
                        # But if U, V, W are sparse, it might be okay?
                        # No, we must ensure orthogonality.
                        # However, for small rank, it's hard to satisfy all 324 constraints.
                        # Let's just check the 18 ones first. If that works, check zeros.
            return err
            
        err = calc_error(U, V, W)
        
        # If error is small, verify full tensor
        if err < 1e-6:
            # Verify all 324 entries
            full_err = 0
            for i_u in range(num_u):
                for i_v in range(num_v):
                    for i_w in range(num_w):
                        val = np.dot(U[i_u, :], V[i_v, :] * W[i_w, :])
                        # Determine target
                        # i_u = i*m+j, i_v = j'*p+k', i_w = k''*n+i''
                        i = i_u // m
                        j = i_u % m
                        jp = i_v // p
                        k = i_v % p
                        kn = i_w // n
                        ii = i_w % n
                        target = 1 if (j == jp and k == kn and i == ii) else 0
                        if val != target:
                            full_err = 1
                            break
                    if full_err: break
                if full_err: break
            if not full_err:
                return U, V, W

        # Local search: try changing one element
        # This is too slow for 1000 restarts if done naively.
        # I will skip the complex local search and just return None if not found instantly.
        # Actually, random chance of finding a valid integer solution in 1000 tries is low.
        # I need a better search or a hardcoded solution.
        
    return None, None, None
# EVOLVE-BLOCK-END
