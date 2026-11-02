"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomp(U, V, W, n, m, p):
    """Verify if U, V, W form a valid matrix multiplication tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] V[j'*p+k, c] W[k'*n+i', c] 
               == 1 if j==j' and k==k' and i==i', else 0.
    """
    rank = U.shape[1]
    # Use numpy for speed
    # Construct the full tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # This is a batched matrix product / tensor contraction.
    # T = np.einsum('ar,br,cr->abc', U, V, W)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Construct the target tensor
    # Target[a, b, c] = 1 if indices match, else 0
    # a = i*m + j
    # b = j'*p + k
    # c = k'*n + i'
    # Match if j == j' and k == k' and i == i'
    
    # We can check this by iterating or using masks.
    # Given small dimensions, iteration is fine, but numpy is faster.
    
    # Let's construct the target tensor explicitly.
    Target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                Target[a, b, c] = 1.0
                
    # Check if T matches Target
    if not np.allclose(T, Target, atol=1e-9):
        return False
    return True

def _find_decomp(n, m, p, target_rank, max_attempts, seed=None):
    """Search for a rank-target_rank decomposition for n x m x p matrix multiplication."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    for attempt in range(max_attempts):
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Random sparse initialization
        for c in range(target_rank):
            # Each term should have a few non-zero entries
            # Heuristic: number of non-zeros proportional to dimensions
            num_nonzero_u = random.randint(2, max(2, int(np.sqrt(n*m))))
            num_nonzero_v = random.randint(2, max(2, int(np.sqrt(m*p))))
            num_nonzero_w = random.randint(2, max(2, int(np.sqrt(p*n))))
            
            u_pos = random.sample(range(n*m), min(num_nonzero_u, n*m))
            v_pos = random.sample(range(m*p), min(num_nonzero_v, m*p))
            w_pos = random.sample(range(p*n), min(num_nonzero_w, p*n))
            
            for pos in u_pos:
                U[pos, c] = random.choice([-1, 1])
            for pos in v_pos:
                V[pos, c] = random.choice([-1, 1])
            for pos in w_pos:
                W[pos, c] = random.choice([-1, 1])
                
        if _verify_decomp(U, V, W, n, m, p):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Strategy 1: Try to find a lower rank full decomposition
    # Start from a reasonable guess. Schoolbook is 54.
    # Seed found 18. Try below 18.
    
    # We can try block decompositions first as they are structured.
    # Block size 3 (for p=9).
    # Try to find a 2x3x3 block decomposition of rank r_block.
    # Total rank = 3 * r_block.
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p  # Schoolbook
    
    # Try block ranks from 5 down to 1
    # If r_block=6, total rank=18.
    # If r_block=5, total rank=15.
    
    for r_block in range(5, 0, -1):
        if time.time() > deadline:
            break
        decomp = _find_decomp(2, 3, 3, r_block, max_attempts=1000)
        if decomp is not None:
            U_b, V_b, W_b = decomp
            total_rank = 3 * r_block
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            for g in range(3):
                for c in range(r_block):
                    col = g * r_block + c
                    # U is shared
                    U[:, col] = U_b[:, c]
                    # V block
                    for j in range(3):
                        for k_local in range(3):
                            k_global = g * 3 + k_local
                            V[j*p + k_global, col] = V_b[j*3 + k_local, c]
                    # W block
                    for k_local in range(3):
                        k_global = g * 3 + k_local
                        for i in range(2):
                            W[k_global*n + i, col] = W_b[k_local*2 + i, c]
                            
            # Verify full decomposition
            if _verify_decomp(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = total_rank
                break # Found a valid one, move to next lower rank? No, break to try even lower?
                # Actually, we want the MINIMAL rank. So we should try lower r_block first.
                # The loop goes 5, 4, 3... so if we find 5, we break and use it.
                # But maybe 4 exists? We should continue searching for lower ranks if time permits.
                # However, finding lower ranks is harder.
                # Let's just return the first valid one found (lowest r_block tried first).
                # Wait, the loop is `for r_block in range(5, 0, -1)`.
                # So it tries 5 first. If found, it breaks.
                # This gives rank 15.
                # If 5 fails, it tries 4 (rank 12).
                # This is good.

    if best_U is None:
        # Try full decomposition search for ranks < 18
        # This is harder.
        for r in range(17, 5, -1):
            if time.time() > deadline:
                break
            decomp = _find_decomp(n, m, p, r, max_attempts=500)
            if decomp is not None:
                best_U, best_V, best_W = decomp
                best_rank = r
                break

    if best_U is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
