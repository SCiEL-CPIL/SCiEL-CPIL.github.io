"""Matrix multiplication 2x3 by 3x8 tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    T[a, b, c] = sum_d U[a, d] * V[b, d] * W[c, d]
    where a = i*m + j, b = j*p + k, c = k*n + i
    """
    rank = U.shape[1]
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b]) * 0 # Placeholder, need to compute tensor contraction
                # Correct contraction: T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                val = 0.0
                for d in range(rank):
                    val += U[a, d] * V[b, d] * W[c, d]
                if val != 1.0:
                    return False
                
                # Check spurious terms (should be 0)
                # Iterate over all possible (a', b', c') that are not the target (i,j,k)
                # This is expensive, so we only do a subset or rely on the fact that
                # if the tensor is sparse and we match the non-zeros, we might miss spurious terms.
                # However, for matrix multiplication, the tensor is sparse.
                # We must ensure T[a', b', c'] == 0 for all (a', b', c') != (i, j, k) corresponding to valid indices.
                # Actually, the tensor T is defined on the grid.
                # T[a, b, c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
                # Otherwise 0.
                # So we need to check all (a, b, c) in the grid.
    # Full check is O(n*m*p * n*m*p * p*n * rank) which is too slow for large grids.
    # But n=2, m=3, p=8.
    # Grid size: 6 * 24 * 16 = 2304 entries.
    # Rank ~ 40.
    # 2304 * 40 ~ 100k ops. This is fast enough.
    
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                val = 0.0
                for d in range(rank):
                    val += U[a, d] * V[b, d] * W[c, d]
                
                # Determine expected value
                if i_a == i_c and j_a == j_b and k_b == k_c:
                    if val != 1.0:
                        return False
                else:
                    if val != 0.0:
                        return False
    return True

def _random_search_decomposition(n, m, p, target_rank, max_iters=1000, timeout=10):
    """
    Try to find a rank-target_rank decomposition using random search.
    """
    start_time = time.time()
    np.random.seed(42)
    
    # Initialize with random small integers
    U = np.random.randint(-1, 2, size=(n * m, target_rank))
    V = np.random.randint(-1, 2, size=(m * p, target_rank))
    W = np.random.randint(-1, 2, size=(p * n, target_rank))
    
    best_rank = target_rank
    best_U, best_V, best_W = None, None, None
    
    for _ in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Random perturbation
        if random.random() < 0.5:
            # Flip a random entry
            idx_u = (random.randint(0, n * m - 1), random.randint(0, target_rank - 1))
            U[idx_u] = -U[idx_u] if U[idx_u] != 0 else 1
        elif random.random() < 0.5:
            idx_v = (random.randint(0, m * p - 1), random.randint(0, target_rank - 1))
            V[idx_v] = -V[idx_v] if V[idx_v] != 0 else 1
        else:
            idx_w = (random.randint(0, p * n - 1), random.randint(0, target_rank - 1))
            W[idx_w] = -W[idx_w] if W[idx_w] != 0 else 1
            
        if _verify_decomposition(U, V, W, n, m, p):
            return U.copy(), V.copy(), W.copy()
            
    return None, None, None

def _construct_2x3_3x2_rank11():
    """
    Attempt to construct a rank-11 decomposition for 2x3 * 3x2.
    If not found, fall back to rank 12.
    """
    n, m, p = 2, 3, 2
    # Try to find rank 11
    U, V, W = _random_search_decomposition(n, m, p, 11, max_iters=500, timeout=5)
    if U is not None:
        return U, V, W
    
    # Fall back to schoolbook rank 12
    rank = 12
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _construct_tiled():
    """
    Construct a decomposition by tiling 2x3 * 3x2 blocks.
    """
    n, m, p = 2, 3, 8
    U_sub, V_sub, W_sub = _construct_2x3_3x2_rank11()
    rank_sub = U_sub.shape[1]
    rank = 4 * rank_sub
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(4):
        col_offset = block * rank_sub
        for d in range(rank_sub):
            U[:, col_offset + d] = U_sub[:, d]
            
            for j in range(m):
                for k in range(2):
                    V[j * p + block * 2 + k, col_offset + d] = V_sub[j * 2 + k, d]
            
            for k in range(2):
                for i in range(n):
                    W[(block * 2 + k) * n + i, col_offset + d] = W_sub[k * n + i, d]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # 1. Try to find a global low-rank decomposition using random search
    # This is a structural change: direct search on the full problem space
    # instead of tiling.
    
    # Try ranks from 32 up to 47
    # Note: Random search is unlikely to find the exact solution quickly for high dimensions,
    # but we give it a chance. If it fails, we fall back to tiling.
    
    # However, random search over 6x24x16 grid with rank 32 is huge.
    # It will almost certainly fail within the timeout.
    # So we rely on the tiling strategy with the improved sub-block search.
    
    # Let's try to improve the sub-block rank first.
    U_tiled, V_tiled, W_tiled = _construct_tiled()
    rank_tiled = U_tiled.shape[1]
    
    # Verify tiled decomposition
    if _verify_decomposition(U_tiled, V_tiled, W_tiled, n, m, p):
        # We have a valid decomposition.
        # Can we do better?
        # Try to find a global decomposition with rank < rank_tiled
        # This is computationally expensive, so we only try if rank_tiled is high.
        
        # For now, return the tiled result.
        # If rank_tiled is 44 (from rank-11 sub-blocks), that's an improvement over 48.
        
        # Check if we can find a better global solution with a short timeout
        if rank_tiled > 40 and time.time() < deadline - 30:
            # Try rank rank_tiled - 4
            target_rank = rank_tiled - 4
            U_glob, V_glob, W_glob = _random_search_decomposition(n, m, p, target_rank, max_iters=100, timeout=10)
            if U_glob is not None:
                return {"U": U_glob.tolist(), "V": V_glob.tolist(), "W": W_glob.tolist()}
        
        return {"U": U_tiled.tolist(), "V": V_tiled.tolist(), "W": W_tiled.tolist()}
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
