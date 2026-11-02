"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
    Uses a block decomposition into four 2x3x2 blocks.
    Attempts to find a rank-11 decomposition for each 2x3x2 block.
    If successful, total rank is 44. Otherwise, falls back to schoolbook (rank 48).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    def verify_full(U, V, W):
        """Verify the full 2x3x8 decomposition."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def find_2x3x2_decomposition(target_rank):
        """
        Finds a rank-target_rank decomposition for 2x3x2 matrix multiplication.
        Returns U, V, W as lists of lists, or None if not found.
        U shape: (6, r) corresponding to i in [0,1], j in [0,2]
        V shape: (6, r) corresponding to j in [0,2], k in [0,1]
        W shape: (4, r) corresponding to k in [0,1], i in [0,1]
        """
        # Target tensor entries are all 1 for valid (i,j,k)
        # n=2, m=3, p=2
        # U indices: i*m + j -> 0..5
        # V indices: j*p + k -> 0..5
        # W indices: k*n + i -> 0..3
        
        def loss(params):
            U_flat = params[0:6*target_rank]
            V_flat = params[6*target_rank:12*target_rank]
            W_flat = params[12*target_rank:16*target_rank]
            
            U = np.array(U_flat).reshape(6, target_rank)
            V = np.array(V_flat).reshape(6, target_rank)
            W = np.array(W_flat).reshape(4, target_rank)
            
            loss_val = 0.0
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        u_row = U[i*3 + j]
                        v_row = V[j*2 + k]
                        w_row = W[k*2 + i]
                        prod = np.sum(u_row * v_row * w_row)
                        loss_val += (prod - 1.0)**2
            return loss_val

        # Try multiple random initializations
        np.random.seed(42)
        best_loss = float('inf')
        best_params = None
        
        for attempt in range(20):
            # Initialize with small random values
            init_scale = 0.1
            x0 = np.random.randn(16 * target_rank) * init_scale
            
            res = minimize(loss, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                
            if best_loss < 1e-6:
                break
                
        if best_loss > 1e-5:
            return None

        # Extract and round to nearest half-integer
        U_flat = best_params[0:6*target_rank]
        V_flat = best_params[6*target_rank:12*target_rank]
        W_flat = best_params[12*target_rank:16*target_rank]
        
        U = np.array(U_flat).reshape(6, target_rank)
        V = np.array(V_flat).reshape(6, target_rank)
        W = np.array(W_flat).reshape(4, target_rank)
        
        # Round to nearest 0.5
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify exactness
        valid = True
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    total = np.sum(U[i*3 + j] * V[j*2 + k] * W[k*2 + i])
                    if abs(total - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if not valid:
            return None
            
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        return U_list, V_list, W_list

    # Try to find a rank-11 decomposition for the 2x3x2 block
    block_decomp = None
    for target_rank in [11, 12]:
        if time.time() > deadline - 20:
            break
        block_decomp = find_2x3x2_decomposition(target_rank)
        if block_decomp:
            break

    if block_decomp:
        U_block, V_block, W_block = block_decomp
        block_rank = len(U_block[0])
        
        # Construct the full 2x3x8 decomposition by replicating the block 4 times
        # p=8 is split into 4 blocks of size 2
        total_rank = 4 * block_rank
        U_full = [[0] * total_rank for _ in range(n * m)] # 6 x total_rank
        V_full = [[0] * total_rank for _ in range(m * p)] # 24 x total_rank
        W_full = [[0] * total_rank for _ in range(p * n)] # 16 x total_rank
        
        for block_idx in range(4):
            base_r = block_idx * block_rank
            # U is independent of k, so we can just copy the block U?
            # Wait, U[i,j,r] in the block corresponds to the same i,j in the full matrix.
            # But the rank indices are different for each block.
            # So for each block, we add the block's U, V, W contributions.
            
            # U: i in 0..1, j in 0..2
            for i in range(n):
                for j in range(m):
                    for r in range(block_rank):
                        U_full[i * m + j][base_r + r] += U_block[i * m + j][r]
                        
            # V: j in 0..2, k in 0..1 (local k)
            # Global k = block_idx * 2 + local_k
            for j in range(m):
                for local_k in range(2):
                    global_k = block_idx * 2 + local_k
                    for r in range(block_rank):
                        V_full[j * p + global_k][base_r + r] += V_block[j * 2 + local_k][r]
                        
            # W: local_k in 0..1, i in 0..1
            # Global k = block_idx * 2 + local_k
            for local_k in range(2):
                global_k = block_idx * 2 + local_k
                for i in range(n):
                    for r in range(block_rank):
                        W_full[global_k * n + i][base_r + r] += W_block[local_k * n + i][r]
                        
        # Verify full decomposition
        if verify_full(U_full, V_full, W_full):
            return {"U": U_full, "V": V_full, "W": W_full}

    # Fallback to schoolbook
    U, V, W, rank = make_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
