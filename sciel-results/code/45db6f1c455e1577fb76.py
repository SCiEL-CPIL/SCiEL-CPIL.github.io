import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def _check_half_integer(arr, tol=1e-6):
    """Check if all entries are within tol of half-integers. Return rounded half-integers if valid."""
    doubled = arr * 2.0
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) < tol:
        return rounded / 2.0
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the n x m x p matmul tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i * m + j, j * p + k, k * n + i] = 1.0
    
    if not np.allclose(T, expected, atol=1e-5):
        return False
    return True

def _search_3x3x2_block(rank, max_attempts=5000, deadline=None):
    """
    Search for a rank-R decomposition of the 3x3x2 matmul tensor.
    U: 9 x R, V: 6 x R, W: 6 x R
    Returns (U, V, W) as numpy arrays or None.
    """
    n, m, p = 3, 3, 2
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Initialize with random integers
        # Density helps avoid trivial solutions
        density = 0.5
        U = (np.random.rand(n * m, rank) < density) * np.random.randint(-2, 3, size=(n * m, rank))
        V = (np.random.rand(m * p, rank) < density) * np.random.randint(-2, 3, size=(m * p, rank))
        
        # Solve for W
        W = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Construct M: m x rank
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # We need M @ w = ones(m)
                # w is the row of W corresponding to (k, i)
                try:
                    w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                # Check if w_vec is close to half-integers
                w_check = _check_half_integer(w_vec)
                if w_check is None:
                    valid = False
                    break
                W[k * n + i, :] = w_check
        
        if valid:
            # Verify the block decomposition
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U, V, W
    
    return None

def _build_3x3x6_from_blocks(block_U, block_V, block_W, n, m, p, p_split):
    """
    Build a 3x3x6 decomposition from p_split blocks of 3x3x(p/p_split).
    Here p=6, p_split=3, so blocks are 3x3x2.
    """
    p_per_block = p // p_split
    core_rank = len(block_U[0])
    total_rank = p_split * core_rank
    
    U_total = np.zeros((n * m, total_rank))
    V_total = np.zeros((m * p, total_rank))
    W_total = np.zeros((p * n, total_rank))
    
    for t in range(p_split):
        offset = t * core_rank
        # U is the same for all blocks (A is the same)
        U_total[:, offset:offset+core_rank] = block_U
        
        # V and W are shifted for each block
        # Block t corresponds to columns t*p_per_block to (t+1)*p_per_block - 1 of B
        # And rows t*p_per_block to (t+1)*p_per_block - 1 of C
        
        for j in range(m):
            for k_local in range(p_per_block):
                k_global = t * p_per_block + k_local
                V_total[j * p + k_global, offset:offset+core_rank] = block_V[j * p_per_block + k_local, :]
        
        for k_local in range(p_per_block):
            for i in range(n):
                k_global = t * p_per_block + k_local
                W_total[k_global * n + i, offset:offset+core_rank] = block_W[k_local * n + i, :]
                
    return U_total, V_total, W_total

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    best_result = None
    best_rank = n * m * p  # 54
    
    # Strategy 1: Search for 3x3x2 block decomposition
    # Try ranks from 12 up to 17
    for core_rank in range(12, 18):
        if time.time() > deadline:
            break
        total_rank = 3 * core_rank
        if total_rank >= best_rank:
            continue
            
        # Try multiple times to find a valid block
        block_found = None
        for attempt in range(5): # 5 restarts per rank
            if time.time() > deadline:
                break
            block = _search_3x3x2_block(core_rank, max_attempts=1000, deadline=deadline)
            if block is not None:
                block_found = block
                break
        
        if block_found is not None:
            U_block, V_block, W_block = block_found
            U_total, V_total, W_total = _build_3x3x6_from_blocks(U_block, V_block, W_block, n, m, p, 3)
            
            # Final verification
            if _verify_decomposition(U_total.tolist(), V_total.tolist(), W_total.tolist(), n, m, p):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {
                        "U": U_total.tolist(),
                        "V": V_total.tolist(),
                        "W": W_total.tolist()
                    }
                # Since we are iterating core_rank upwards, the first valid one is the best
                break
    
    # Strategy 2: Fallback to Schoolbook (Rank 54)
    if best_result is None:
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
        best_result = {"U": U, "V": V, "W": W}
        
    return best_result
